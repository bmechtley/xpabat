// ─── Feature-space projection modal ───────────────────────────────────────────
// A 2-D scatter of all (filtered) calls.  Each axis can be a raw acoustic
// feature OR a PCA principal component.  PCA runs client-side on the
// standardized feature matrix (calls already hold every scalar feature, so no
// server round-trip is needed).  Points are coloured by species and clicking
// one jumps to that call in the main view.

const _PROJ_FEATURES = [
  { key: 'dur',     label: 'Duration (ms)' },
  { key: 'Fmin',    label: 'F min (kHz)' },
  { key: 'Fmax',    label: 'F max (kHz)' },
  { key: 'Fpeak',   label: 'F peak (kHz)' },
  { key: 'bw',      label: 'Bandwidth (kHz)' },
  { key: 'sweep',   label: 'Sweep' },
  { key: 'cf_frac', label: 'CF fraction' },
  { key: 'ar1',     label: 'Audio AR(2) a₁' },
  { key: 'ar2',     label: 'Audio AR(2) a₂' },
  { key: 'ar1c',    label: 'Audio AR(1) a₁' },
  { key: 'car1',    label: 'Contour AR(2) a₁' },
  { key: 'car2',    label: 'Contour AR(2) a₂' },
  { key: 'car1c',   label: 'Contour AR(1) a₁' },
];

// Features EXCLUDED from the PCA (toggled by the floating checkboxes).  Stored
// as the excluded set — persisted in localStorage — so a newly-added feature
// defaults to included.
const _projExcluded = new Set((() => {
  try { const s = JSON.parse(localStorage.getItem('projPcaExcluded')); return Array.isArray(s) ? s : []; }
  catch { return []; }
})());
function _projSaveExcluded() {
  try { localStorage.setItem('projPcaExcluded', JSON.stringify([..._projExcluded])); } catch {}
}
// Feature keys that feed the PCA (never empty — falls back to all).
function _projPcaKeys() {
  const ks = _PROJ_FEATURES.filter(f => !_projExcluded.has(f.key)).map(f => f.key);
  return ks.length ? ks : _PROJ_FEATURES.map(f => f.key);
}

// Fit AR(2) + AR(1) to a 1-D sequence (mirrors features.ar_features in Python).
// Returns [a1, a2, b1]; zeros for degenerate input.
function _arFit(vals) {
  const n = vals.length;
  if (n < 4) return [0, 0, 0];
  let mean = 0;
  for (let i = 0; i < n; i++) mean += vals[i];
  mean /= n;
  let r0 = 0, r1 = 0, r2 = 0;
  for (let i = 0; i < n; i++)     { const x = vals[i] - mean; r0 += x * x; }
  for (let i = 0; i < n - 1; i++) r1 += (vals[i] - mean) * (vals[i + 1] - mean);
  for (let i = 0; i < n - 2; i++) r2 += (vals[i] - mean) * (vals[i + 2] - mean);
  r0 /= n; r1 /= n; r2 /= n;
  if (r0 <= 1e-20) return [0, 0, 0];
  const b1 = r1 / r0;
  const denom = r0 * r0 - r1 * r1;
  if (Math.abs(denom) < 1e-20) return [0, 0, b1];
  return [(r1 * r0 - r2 * r1) / denom, (r2 * r0 - r1 * r1) / denom, b1];
}

// Compute the contour-frequency AR features (car1/car2/car1c) for every call
// from its currently-selected contour, caching them on the call object.
function _projComputeContourAR() {
  if (!S.calls) return;
  for (const c of S.calls) {
    const ct = (typeof getContour === 'function')
             ? getContour(c) : (c.contour || c.contour_cwt);
    if (ct && ct.length >= 4) {
      const freqs = new Array(ct.length);
      for (let i = 0; i < ct.length; i++) freqs[i] = ct[i][1];
      const [a1, a2, b1] = _arFit(freqs);
      c.car1 = a1; c.car2 = a2; c.car1c = b1;
    } else {
      c.car1 = 0; c.car2 = 0; c.car1c = 0;
    }
  }
}

// State cached between renders for the currently-open projection.
let _proj = null;   // { calls, Z, means, stds, n, d, pca:{order, vecs, varRatio}, px, py }

// ── UMAP (unsupervised, nonlinear) ───────────────────────────────────────────
// A faithful but compact UMAP: smooth-kNN fuzzy simplicial set + SGD layout with
// the standard (a,b) attraction/repulsion kernel.  Runs entirely client-side,
// progressively (kNN then optimization are chunked across animation frames so
// the UI never freezes and you watch the embedding converge).
//
// Quantization guard: STFT-derived features (Fmin/Fmax/Fpeak, duration, the
// contour-AR set) land on a discrete lattice.  A neighbour graph reads coincident
// lattice values as dense micro-clusters — spurious "classes".  Before building
// the graph each feature is dithered by ±½ its own quantization step (auto-
// detected as the smallest real gap between sorted values), smearing the lattice
// back into the continuum.  Continuous features (audio-AR) have a negligible step
// so their jitter is ~0.
const UMAP_MAX_POINTS  = 4000;   // subsample cap (exact kNN is O(m²·d))
const UMAP_N_NEIGHBORS = 30;     // larger than the default 15 → favours global
                                 // structure, suppresses quantization micro-clusters
const UMAP_NEG_RATE    = 5;      // negative samples per positive edge
const UMAP_A = 1.5769434603113077;   // (a,b) for min_dist=0.1, spread=1.0
const UMAP_B = 0.8950608779109733;
const UMAP_KNN_ROWS_PER_FRAME = 160;
const UMAP_EPOCHS_PER_FRAME   = 12;

let _umap = null;   // active run/result — see _projEnsureUmap

// Deterministic PRNG (mulberry32) so repeated builds land in the same place.
function _umapRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Quantization step of a feature = smallest gap between sorted values that exceeds
// float/round-off noise.  Continuous features → tiny step (≈0 jitter).
function _umapQuantStep(vals) {
  const a = Float64Array.from(vals).sort();
  const range = (a[a.length - 1] - a[0]) || 1;
  const tol = range * 1e-4;
  let step = Infinity, prev = a[0];
  for (let i = 1; i < a.length; i++) {
    const g = a[i] - prev;
    if (g > tol) { if (g < step) step = g; prev = a[i]; }
  }
  return isFinite(step) ? step : 0;
}

// Build the standardized + dithered feature matrix over a (sub)sample of calls.
function _umapFeatureMatrix(calls, idx, keys, rng) {
  const m = idx.length, d = keys.length;
  const X = new Float64Array(m * d);
  // Per-feature: gather raw values, detect step, jitter, then standardize.
  const col = new Float64Array(m);
  for (let j = 0; j < d; j++) {
    const k = keys[j];
    for (let r = 0; r < m; r++) col[r] = (+calls[idx[r]][k] || 0);
    const step = _umapQuantStep(col);
    let mean = 0;
    for (let r = 0; r < m; r++) { col[r] += (rng() - 0.5) * step; mean += col[r]; }
    mean /= m;
    let sd = 0;
    for (let r = 0; r < m; r++) { const v = col[r] - mean; sd += v * v; }
    sd = Math.sqrt(sd / Math.max(m - 1, 1)) || 1;
    for (let r = 0; r < m; r++) X[r * d + j] = (col[r] - mean) / sd;
  }
  return X;
}

// Exact k-nearest-neighbours for rows [r0,r1) of X.  Self excluded.
function _umapKnnRows(u, r0, r1) {
  const { X, m, d, k, knnIdx, knnDist } = u;
  const nd = new Float64Array(k), ni = new Int32Array(k);
  for (let i = r0; i < r1; i++) {
    for (let t = 0; t < k; t++) { nd[t] = Infinity; ni[t] = -1; }
    const oi = i * d;
    for (let j = 0; j < m; j++) {
      if (j === i) continue;
      const oj = j * d;
      let s = 0;
      for (let c = 0; c < d; c++) { const diff = X[oi + c] - X[oj + c]; s += diff * diff; }
      if (s >= nd[k - 1]) continue;
      // insertion into the sorted k-buffer
      let p = k - 1;
      while (p > 0 && nd[p - 1] > s) { nd[p] = nd[p - 1]; ni[p] = ni[p - 1]; p--; }
      nd[p] = s; ni[p] = j;
    }
    const o = i * k;
    for (let t = 0; t < k; t++) { knnIdx[o + t] = ni[t]; knnDist[o + t] = Math.sqrt(nd[t]); }
  }
}

// Smooth-kNN → fuzzy membership, then symmetrize into a directed COO edge list.
function _umapBuildFuzzy(u) {
  const { m, k, knnIdx, knnDist } = u;
  const target = Math.log2(k);
  const W = new Float64Array(m * k);   // membership per knn entry
  for (let i = 0; i < m; i++) {
    const o = i * k;
    // rho = nearest strictly-positive distance
    let rho = 0;
    for (let t = 0; t < k; t++) { if (knnDist[o + t] > 0) { rho = knnDist[o + t]; break; } }
    let lo = 0, hi = Infinity, mid = 1;
    for (let it = 0; it < 64; it++) {
      let psum = 0;
      for (let t = 0; t < k; t++) {
        const dd = knnDist[o + t] - rho;
        psum += dd > 0 ? Math.exp(-dd / mid) : 1;
      }
      if (Math.abs(psum - target) < 1e-5) break;
      if (psum > target) { hi = mid; mid = (lo + hi) / 2; }
      else { lo = mid; mid = hi === Infinity ? mid * 2 : (lo + hi) / 2; }
    }
    for (let t = 0; t < k; t++) {
      const dd = knnDist[o + t] - rho;
      W[o + t] = dd > 0 ? Math.exp(-dd / mid) : 1;
    }
  }
  // Symmetrize: p = a + b − a·b, over directed entries (i→j) and (j→i).
  // Accumulate into a map keyed "i,j" then emit both directions for the SGD.
  const sym = new Map();
  const key = (i, j) => i * m + j;
  for (let i = 0; i < m; i++) {
    const o = i * k;
    for (let t = 0; t < k; t++) {
      const j = knnIdx[o + t]; if (j < 0) continue;
      const a = W[o + t];
      const kk = key(i, j), rk = key(j, i);
      if (sym.has(rk)) {
        const b = sym.get(rk); sym.set(rk, b + a - a * b); // combine the reciprocal
      } else {
        sym.set(kk, (sym.get(kk) || 0) + a);
      }
    }
  }
  const head = [], tail = [], weight = [];
  for (const [kk, w] of sym) {
    const i = Math.floor(kk / m), j = kk % m;
    head.push(i, j); tail.push(j, i); weight.push(w, w);  // both directions
  }
  u.head = Int32Array.from(head);
  u.tail = Int32Array.from(tail);
  u.weight = Float64Array.from(weight);
}

function _umapInitOptimize(u) {
  const m = u.m, ne = u.head.length;
  let wmax = 1e-12;
  for (let e = 0; e < ne; e++) if (u.weight[e] > wmax) wmax = u.weight[e];
  // epochs-per-sample: strongest edge sampled every epoch; weak edges get a large
  // interval (effectively never sampled), so no explicit pruning is needed.
  const eps = new Float64Array(ne), nextS = new Float64Array(ne),
        epsNeg = new Float64Array(ne), nextNeg = new Float64Array(ne);
  for (let e = 0; e < ne; e++) {
    eps[e] = wmax / u.weight[e];
    nextS[e] = eps[e];
    epsNeg[e] = eps[e] / UMAP_NEG_RATE;
    nextNeg[e] = epsNeg[e];
  }
  u.eps = eps; u.nextS = nextS; u.epsNeg = epsNeg; u.nextNeg = nextNeg;
  // Random init, uniform(−10,10) — the scale the (a,b) kernel is tuned for.
  const emb = new Float64Array(m * 2);
  for (let i = 0; i < m * 2; i++) emb[i] = (u.rng() - 0.5) * 20;
  u.emb = emb;
  u.epoch = 0;
  u.nEpochs = m <= 2000 ? 500 : 200;
  u.alpha0 = 1.0;
}

function _umapOptimizeEpochs(u, nEpochs) {
  const { head, tail, weight, eps, nextS, epsNeg, nextNeg, emb, m } = u;
  const ne = head.length, rng = u.rng;
  const clip = x => (x > 4 ? 4 : x < -4 ? -4 : x);
  for (let pass = 0; pass < nEpochs && u.epoch < u.nEpochs; pass++) {
    const n = u.epoch;
    const alpha = u.alpha0 * (1 - n / u.nEpochs);
    for (let e = 0; e < ne; e++) {
      if (nextS[e] > n) continue;
      const i = head[e], j = tail[e];
      const oi = i * 2, oj = j * 2;
      let dx = emb[oi] - emb[oj], dy = emb[oi + 1] - emb[oj + 1];
      let d2 = dx * dx + dy * dy;
      // Attraction
      if (d2 > 0) {
        const gc = (-2 * UMAP_A * UMAP_B * Math.pow(d2, UMAP_B - 1)) /
                   (1 + UMAP_A * Math.pow(d2, UMAP_B));
        const gx = clip(gc * dx) * alpha, gy = clip(gc * dy) * alpha;
        emb[oi] += gx; emb[oi + 1] += gy;
        emb[oj] -= gx; emb[oj + 1] -= gy;
      }
      nextS[e] += eps[e];
      // Negative samples (repulsion)
      const nNeg = Math.floor((n - nextNeg[e]) / epsNeg[e]);
      for (let q = 0; q < nNeg; q++) {
        const t = (rng() * m) | 0;
        if (t === i) continue;
        const ot = t * 2;
        dx = emb[oi] - emb[ot]; dy = emb[oi + 1] - emb[ot + 1];
        d2 = dx * dx + dy * dy;
        if (d2 > 0) {
          const gc = (2 * UMAP_B) / ((0.001 + d2) * (1 + UMAP_A * Math.pow(d2, UMAP_B)));
          emb[oi] += clip(gc * dx) * alpha; emb[oi + 1] += clip(gc * dy) * alpha;
        } else {
          emb[oi] += 4 * alpha; emb[oi + 1] += 4 * alpha;
        }
      }
      nextNeg[e] += nNeg * epsNeg[e];
    }
    u.epoch++;
  }
}

// Status line shown in the axis bar during/after a UMAP build.
function _umapSetStatus(txt) {
  const el = document.getElementById('proj-umap-status');
  if (el) el.textContent = txt || '';
}

// Drive a UMAP build forward one animation frame.  Phases: knn → opt → done.
function _umapTick() {
  const u = _umap;
  if (!u || u.cancelled) return;
  if (u.phase === 'knn') {
    const r1 = Math.min(u.m, u.knnCursor + UMAP_KNN_ROWS_PER_FRAME);
    _umapKnnRows(u, u.knnCursor, r1);
    u.knnCursor = r1;
    _umapSetStatus(`Building neighbour graph… ${Math.round(100 * r1 / u.m)}%`);
    if (u.knnCursor >= u.m) {
      _umapBuildFuzzy(u);
      _umapInitOptimize(u);
      u.phase = 'opt';
    }
  } else if (u.phase === 'opt') {
    _umapOptimizeEpochs(u, UMAP_EPOCHS_PER_FRAME);
    _umapSetStatus(`Optimizing layout… ${Math.round(100 * u.epoch / u.nEpochs)}%`);
    _projForceRender();
    if (u.epoch >= u.nEpochs) u.phase = 'done';
  }
  if (u.phase === 'done') {
    u.raf = 0;
    _umapSetStatus(`UMAP · ${u.m.toLocaleString()}${u.subsampled ? ' (subsampled)' : ''} calls · ${u.keys.length} features`);
    _projForceRender();
    return;
  }
  u.raf = requestAnimationFrame(_umapTick);
}

// Build (or rebuild) the UMAP embedding for the current call population + feature
// selection.  Returns true if an embedding exists or is being built.
function _projEnsureUmap() {
  if (!_projEnsure()) return false;   // populate _proj (calls, contour-AR, etc.)
  const calls = _proj.calls, n = calls.length;
  const keys = _projPcaKeys();
  const buildKey = n + '|' + (S.contourMethod || '') + '|' + keys.join(',');
  if (_umap && _umap.buildKey === buildKey && !_umap.cancelled) return true;
  if (_umap && _umap.raf) cancelAnimationFrame(_umap.raf);

  // Deterministic subsample (cap) of the full population.
  const rng = _umapRng(0x9e3779b1 ^ (n * 2654435761));
  let idx;
  let subsampled = false;
  if (n > UMAP_MAX_POINTS) {
    subsampled = true;
    // Fisher–Yates partial shuffle to pick UMAP_MAX_POINTS distinct indices.
    const all = Int32Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < UMAP_MAX_POINTS; i++) {
      const j = i + ((rng() * (n - i)) | 0);
      const t = all[i]; all[i] = all[j]; all[j] = t;
    }
    idx = all.slice(0, UMAP_MAX_POINTS);
  } else {
    idx = Int32Array.from({ length: n }, (_, i) => i);
  }
  const m = idx.length;
  const k = Math.min(UMAP_N_NEIGHBORS, m - 1);
  const d = keys.length;
  const X = _umapFeatureMatrix(calls, idx, keys, rng);
  const rowByCall = new Int32Array(n).fill(-1);
  for (let r = 0; r < m; r++) rowByCall[idx[r]] = r;

  _umap = {
    buildKey, idx, rowByCall, X, m, d, k, keys, subsampled,
    knnIdx: new Int32Array(m * k), knnDist: new Float64Array(m * k),
    knnCursor: 0, phase: 'knn', emb: null, rng: _umapRng(0x1234567 ^ n),
    cancelled: false, raf: 0,
  };
  _umapSetStatus('Building neighbour graph… 0%');
  _umap.raf = requestAnimationFrame(_umapTick);
  return true;
}

// Playhead-highlight peak dot radius (CSS px).  The pulse/pulsation machinery
// (phTick / phIntensity / phEnsureAnim / _mixWhite) is shared with the
// spectrogram and lives in render.js.
const PROJ_PULSE_PEAK = 9;

// ── Linear algebra (small, dense, symmetric) ─────────────────────────────────

function _projStandardize(calls, keys) {
  const n = calls.length, d = keys.length;
  const means = new Array(d).fill(0);
  const stds  = new Array(d).fill(0);
  for (const c of calls) for (let j = 0; j < d; j++) means[j] += (+c[keys[j]] || 0);
  for (let j = 0; j < d; j++) means[j] /= Math.max(n, 1);
  for (const c of calls) for (let j = 0; j < d; j++) {
    const v = (+c[keys[j]] || 0) - means[j]; stds[j] += v * v;
  }
  for (let j = 0; j < d; j++) stds[j] = Math.sqrt(stds[j] / Math.max(n - 1, 1)) || 1;
  const Z = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    const c = calls[i];
    for (let j = 0; j < d; j++) Z[i * d + j] = ((+c[keys[j]] || 0) - means[j]) / stds[j];
  }
  return { Z, means, stds, n, d };
}

function _projCovariance(Z, n, d) {
  const C = new Float64Array(d * d);
  for (let i = 0; i < n; i++) {
    const off = i * d;
    for (let a = 0; a < d; a++) {
      const za = Z[off + a];
      for (let b = a; b < d; b++) C[a * d + b] += za * Z[off + b];
    }
  }
  const denom = Math.max(n - 1, 1);
  for (let a = 0; a < d; a++) for (let b = a; b < d; b++) {
    const v = C[a * d + b] / denom; C[a * d + b] = v; C[b * d + a] = v;
  }
  return C;
}

// Jacobi eigenvalue algorithm for a symmetric d×d matrix.
// Returns { values:[d], vecs: Float64Array(d*d) } where column j of vecs is the
// eigenvector for values[j].
function _projJacobi(Ain, d) {
  const A = Float64Array.from(Ain);
  const V = new Float64Array(d * d);
  for (let i = 0; i < d; i++) V[i * d + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += A[p * d + q] * A[p * d + q];
    if (off < 1e-14) break;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) {
      const apq = A[p * d + q];
      if (Math.abs(apq) < 1e-18) continue;
      const phi = 0.5 * Math.atan2(2 * apq, A[q * d + q] - A[p * d + p]);
      const c = Math.cos(phi), s = Math.sin(phi);
      for (let k = 0; k < d; k++) {
        const akp = A[k * d + p], akq = A[k * d + q];
        A[k * d + p] = c * akp - s * akq; A[k * d + q] = s * akp + c * akq;
      }
      for (let k = 0; k < d; k++) {
        const apk = A[p * d + k], aqk = A[q * d + k];
        A[p * d + k] = c * apk - s * aqk; A[q * d + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < d; k++) {
        const vkp = V[k * d + p], vkq = V[k * d + q];
        V[k * d + p] = c * vkp - s * vkq; V[k * d + q] = s * vkp + c * vkq;
      }
    }
  }
  const values = new Array(d);
  for (let i = 0; i < d; i++) values[i] = A[i * d + i];
  return { values, vecs: V };
}

function _projPCA(Z, n, d) {
  const C = _projCovariance(Z, n, d);
  const { values, vecs } = _projJacobi(C, d);
  const order = values.map((v, i) => i).sort((a, b) => values[b] - values[a]);
  const total = values.reduce((s, v) => s + Math.max(v, 0), 0) || 1;
  const varRatio = order.map(i => Math.max(values[i], 0) / total);
  return { order, values, vecs, varRatio };
}

// ── Call set ─────────────────────────────────────────────────────────────────
// PCA / standardization is computed over the FULL call population (every call,
// no filters) and kept fixed — so the axes never shift as the min-conf, species,
// or time-window filters change which points are displayed.  Display filtering
// happens per-render in _projRender.

// ── Axis helpers ─────────────────────────────────────────────────────────────

// axis is "feat:<key>" or "pca:<rank>"  (rank 0 = PC1)
function _projAxisLabel(axis) {
  if (axis.startsWith('umap:')) return `UMAP ${+axis.slice(5) + 1}`;
  if (axis.startsWith('feat:')) {
    const k = axis.slice(5);
    return (_PROJ_FEATURES.find(f => f.key === k) || {}).label || k;
  }
  const rank = +axis.slice(4);
  const pct = _proj ? Math.round(_proj.pca.varRatio[rank] * 100) : 0;
  return `PC${rank + 1} (${pct}% var)`;
}

function _projAxisValue(i, axis) {
  if (axis.startsWith('umap:')) {
    const row = _umap ? _umap.rowByCall[i] : -1;
    return row >= 0 && _umap.emb ? _umap.emb[row * 2 + (+axis.slice(5))] : NaN;
  }
  if (axis.startsWith('feat:')) {
    return (+_proj.calls[i][axis.slice(5)] || 0);
  }
  // PCA score: standardized row i · eigenvector for the ranked component
  const rank = +axis.slice(4);
  const col  = _proj.pca.order[rank];
  const d    = _proj.d, off = i * d;
  let acc = 0;
  for (let j = 0; j < d; j++) acc += _proj.Z[off + j] * _proj.pca.vecs[j * d + col];
  return acc;
}

// ── PCA build / tab lifecycle ─────────────────────────────────────────────────

// Build (or rebuild) the PCA over the full call population.  Cheap (~1 ms);
// rebuilt only when the number of loaded calls changes.
// Force the next _projEnsure() to rebuild (e.g. after contours load → the
// contour-AR features change).  Also dirties the UMAP build (contour-AR feeds it).
function _projInvalidate() {
  if (_proj) _proj._buildKey = '';
  if (_umap) { if (_umap.raf) cancelAnimationFrame(_umap.raf); _umap = null; }
}

function _projModeIsUmap() { return S.projMode === 'umap'; }

// Switch the Call-Plot projection method (PCA ↔ UMAP).  PCA exposes axis pickers
// + the biplot; UMAP hides them (its 2-D layout is fixed and nonlinear) and shows
// a build-status line.  The feature checkboxes feed whichever is active.
function setProjMode(mode) {
  mode = mode === 'umap' ? 'umap' : 'pca';
  S.projMode = mode;
  try { localStorage.setItem('projMode', mode); } catch {}
  const umap = mode === 'umap';
  for (const b of document.querySelectorAll('#proj-mode-btns .toggle-btn'))
    b.classList.toggle('clf-active', b.dataset.mode === mode);
  // PCA-only controls
  for (const el of document.querySelectorAll('.proj-pca-only'))
    el.style.display = umap ? 'none' : '';
  const st = document.getElementById('proj-umap-status');
  if (st) st.style.display = umap ? '' : 'none';
  const title = document.querySelector('#proj-feature-panel .pfp-title');
  if (title) title.textContent = umap ? 'UMAP features' : 'PCA features';
  if (S.activeTab === 'plot') _projRebuild();
}

// Build/refresh whichever projection the current mode needs, then render.
function _projRebuild() {
  if (_projModeIsUmap()) _projEnsureUmap();
  else _projEnsure();
  _projForceRender();
}

function _projEnsure() {
  const n = (S.calls && S.calls.length) || 0;
  const keys = _projPcaKeys();   // selected feature subset
  // Rebuild when the population grows, the contour method changes (contour-AR
  // derives from the displayed contour), or the selected feature set changes.
  const buildKey = n + '|' + (S.contourMethod || '') + '|' + keys.join(',');
  if (_proj && _proj._buildKey === buildKey) return _proj && n > 0;
  if (n === 0) { _proj = null; return false; }
  _projComputeContourAR();         // (re)derive car1/car2/car1c on each call
  const calls = S.calls.slice();   // full population, no filters
  const std   = _projStandardize(calls, keys);
  const pca   = _projPCA(std.Z, std.n, std.d);
  _proj = { calls, keys, ...std, pca, _buildKey: buildKey, vis: null, px: null, py: null, _key: '' };
  _projBuildAxisOptions();
  return true;
}

// Switch the main view tab.  Driven by the tab buttons.
function switchTab(name) {
  S.activeTab = name;
  for (const t of document.querySelectorAll('.view-tab'))
    t.classList.toggle('active', t.dataset.tab === name);
  document.getElementById('tab-spectrogram').classList.toggle('active', name === 'spectrogram');
  document.getElementById('tab-plot').classList.toggle('active', name === 'plot');

  if (name === 'plot') {
    if (_projModeIsUmap()) _projEnsureUmap(); else _projEnsure();
    _projResizeCanvas();
    _proj && (_proj._key = '');   // force a redraw
    _projRender();
    if (!S.isPlaying && typeof phAnyActive === 'function' && phAnyActive()) phEnsureAnim();
  } else {
    // Spectrogram tab: its canvas was hidden (0-size) while away — re-measure + draw.
    if (typeof resize === 'function') resize();
    else if (typeof scheduleRender === 'function') scheduleRender();
  }
}

// Called from the main render loop while the plot tab is active.  Repaints every
// frame while a playhead highlight is active (flash / pulsation); otherwise only
// when a tracked input changed.
function _projOnMainRender() {
  if (S.activeTab !== 'plot') return;
  if (_projModeIsUmap()) { if (!_projEnsureUmap()) return; }
  else if (!_projEnsure()) return;
  // While a UMAP build is running it drives its own per-frame renders.
  if (_umap && _umap.phase !== 'done' && _projModeIsUmap()) return;
  if (!phAnyActive()) {
    const key = [
      S.viewStart.toFixed(3), S.viewDur.toFixed(3), S.minConf.toFixed(3),
      S.classifier, S.soloedSpecies || '', [...S.hiddenSpecies].sort().join(','),
      S.selectedCall ? S.selectedCall.id : -1,
      S.playheadTime.toFixed(3),   // paused playhead moves → clear stale highlight
      S.projMode,
      document.getElementById('proj-x')?.value, document.getElementById('proj-y')?.value,
      document.getElementById('proj-biplot')?.checked,
    ].join('|');
    if (key === _proj._key) return;   // nothing relevant changed
    _proj._key = key;
  }
  _projRender();
}

function _projBuildAxisOptions() {
  const nPca = (_proj && _proj.keys ? _proj.keys.length : _PROJ_FEATURES.length);
  const mk = () => {
    let html = '<optgroup label="Features">';
    for (const f of _PROJ_FEATURES) html += `<option value="feat:${f.key}">${f.label}</option>`;
    html += '</optgroup><optgroup label="PCA components">';
    for (let r = 0; r < nPca; r++)
      html += `<option value="pca:${r}">${_projAxisLabel('pca:' + r)}</option>`;
    html += '</optgroup>';
    return html;
  };
  const xSel = document.getElementById('proj-x');
  const ySel = document.getElementById('proj-y');
  // Preserve current selection across rebuilds; default PC1 / PC2.
  const xPrev = xSel.value || 'pca:0';
  const yPrev = ySel.value || 'pca:1';
  xSel.innerHTML = mk(); ySel.innerHTML = mk();
  xSel.value = xPrev; ySel.value = yPrev;
  if (!xSel.value) xSel.value = 'pca:0';
  if (!ySel.value) ySel.value = 'pca:1';
}

function _projResizeCanvas() {
  const cv   = document.getElementById('proj-canvas');
  const wrap = document.getElementById('proj-plot');
  const dpr  = window.devicePixelRatio || 1;
  cv.width   = Math.max(50, Math.floor(wrap.clientWidth  * dpr));
  cv.height  = Math.max(50, Math.floor(wrap.clientHeight * dpr));
  cv.style.width  = wrap.clientWidth  + 'px';
  cv.style.height = wrap.clientHeight + 'px';
  cv._dpr = dpr;
}

const _PROJ_PAD = 44;   // px reserved for axis labels (device px)

// Coefficient of each standardized feature in an axis's value:
//   feat:k → ∂(raw value)/∂z_j = std_k for j=k, else 0
//   pca:r  → eigenvector loading of feature j on the ranked component
function _projAxisCoeffs(axis) {
  const d = _proj.d, a = new Array(d).fill(0);
  if (axis.startsWith('feat:')) {
    const j = _proj.keys.indexOf(axis.slice(5));   // index within the PCA feature set
    if (j >= 0) a[j] = _proj.stds[j];
  } else {
    const col = _proj.pca.order[+axis.slice(4)];
    for (let j = 0; j < d; j++) a[j] = _proj.pca.vecs[j * d + col];
  }
  return a;
}

function _projAxisZeroValue(axis) {
  // Value of the axis when every standardized feature is 0 (the centroid):
  //   pca → score 0;  feat:k → the feature's mean.
  if (axis.startsWith('feat:')) {
    const k = axis.slice(5);
    const j = _proj.keys.indexOf(k);
    if (j >= 0) return _proj.means[j];
    // Feature not in the PCA set: compute its mean directly.
    let m = 0; const cs = _proj.calls;
    for (const c of cs) m += (+c[k] || 0);
    return m / Math.max(cs.length, 1);
  }
  return 0;
}

function _projDrawBiplot(ctx, dpr, xAxis, yAxis, g) {
  const d = _proj.d;
  const aX = _projAxisCoeffs(xAxis);
  const aY = _projAxisCoeffs(yAxis);

  // Pixel displacement per unit standardized feature (y flipped on screen).
  const vx = new Array(d), vy = new Array(d), len = new Array(d);
  let maxLen = 1e-9;
  for (let j = 0; j < d; j++) {
    vx[j] =  aX[j] / (g.xmax - g.xmin) * g.plotW;
    vy[j] = -aY[j] / (g.ymax - g.ymin) * g.plotH;
    len[j] = Math.hypot(vx[j], vy[j]);
    if (len[j] > maxLen) maxLen = len[j];
  }
  // Scale so the longest arrow spans ~22 % of the smaller plot dimension.
  const target = 0.22 * Math.min(g.plotW, g.plotH);
  const k = target / maxLen;

  const ox = g.sx(_projAxisZeroValue(xAxis));
  const oy = g.sy(_projAxisZeroValue(yAxis));

  ctx.save();
  ctx.lineWidth = 1.5 * dpr;
  ctx.font = `${11 * dpr}px system-ui,sans-serif`;
  const minDraw = 10 * dpr;   // hide near-orthogonal features
  for (let j = 0; j < d; j++) {
    const L = len[j] * k;
    if (L < minDraw) continue;
    const ex = ox + vx[j] * k, ey = oy + vy[j] * k;
    const alpha = Math.min(1, 0.4 + (L / target) * 0.6);
    ctx.strokeStyle = `rgba(120,200,255,${alpha})`;
    ctx.fillStyle   = `rgba(120,200,255,${alpha})`;
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ex, ey); ctx.stroke();
    // Arrowhead
    const ang = Math.atan2(ey - oy, ex - ox), ah = 6 * dpr;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - ah * Math.cos(ang - 0.4), ey - ah * Math.sin(ang - 0.4));
    ctx.lineTo(ex - ah * Math.cos(ang + 0.4), ey - ah * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
    // Label at the tip (strip units for compactness)
    const right = ex >= ox;
    ctx.textAlign = right ? 'left' : 'right';
    ctx.textBaseline = ey >= oy ? 'top' : 'bottom';
    const feat = _PROJ_FEATURES.find(f => f.key === _proj.keys[j]);
    const lbl = (feat ? feat.label : _proj.keys[j]).replace(/\s*\(.*\)$/, '');
    ctx.fillText(lbl, ex + (right ? 3 : -3) * dpr, ey);
  }
  ctx.restore();
}

function _projRender() {
  if (!_proj) return;
  const cv  = document.getElementById('proj-canvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, dpr = cv._dpr || 1;
  const umapMode = _projModeIsUmap();
  const xAxis = umapMode ? 'umap:0' : document.getElementById('proj-x').value;
  const yAxis = umapMode ? 'umap:1' : document.getElementById('proj-y').value;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, W, H);

  // UMAP still building its neighbour graph (no coordinates yet) → progress text.
  if (umapMode && (!_umap || !_umap.emb)) {
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = `${13 * dpr}px system-ui,sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const pct = _umap ? Math.round(100 * _umap.knnCursor / _umap.m) : 0;
    ctx.fillText(`Computing UMAP — building neighbour graph… ${pct}%`, W / 2, H / 2);
    return;
  }

  // Displayed subset — the main view's species/confidence filters applied to the
  // fixed PCA.  ALL of these are drawn; the time-overview window doesn't filter
  // them, it only highlights: calls inside the window are drawn bright, the rest
  // dimmed.
  const minc   = S.minConf;
  const soloed = S.soloedSpecies;
  const hidden = S.hiddenSpecies;
  const vis = [];
  for (let i = 0; i < _proj.n; i++) {
    const c = _proj.calls[i];
    if ((c.conf ?? 1) < minc) continue;
    if (hidden.has(c.species)) continue;
    if (soloed && soloed !== c.species) continue;
    if (umapMode && _umap.rowByCall[i] < 0) continue;   // not in the embedded subsample
    vis.push(i);
  }
  const nv = vis.length;

  if (nv === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = `${13 * dpr}px system-ui,sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('No calls match the current filters.', W / 2, H / 2);
    _proj.vis = []; _proj.px = null; _proj.py = null;
    return;
  }

  // Axis values + data ranges over the visible subset
  const xs = new Float64Array(nv), ys = new Float64Array(nv);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let k = 0; k < nv; k++) {
    const i = vis[k];
    const x = _projAxisValue(i, xAxis), y = _projAxisValue(i, yAxis);
    xs[k] = x; ys[k] = y;
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
  }
  if (xmin === xmax) { xmin -= 1; xmax += 1; }
  if (ymin === ymax) { ymin -= 1; ymax += 1; }

  const padL = _PROJ_PAD, padB = _PROJ_PAD, padT = 12 * dpr, padR = 12 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const sx = v => padL + (v - xmin) / (xmax - xmin) * plotW;
  const sy = v => padT + (1 - (v - ymin) / (ymax - ymin)) * plotH;

  // Axes frame
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  // Screen coords for hit-testing — parallel to `vis`.
  const px = new Float32Array(nv), py = new Float32Array(nv);
  for (let k = 0; k < nv; k++) { px[k] = sx(xs[k]); py[k] = sy(ys[k]); }
  _proj.vis = vis; _proj.px = px; _proj.py = py;
  // callId → vis position, for the playhead-pulse pass.
  const posById = new Map();
  for (let k = 0; k < nv; k++) posById.set(_proj.calls[vis[k]].id, k);

  // Time-overview highlight: calls overlapping the current viewport window are
  // drawn bright; everything else is dimmed (the cloud stays visible as a ghost
  // so you can see where the selected time falls in feature space).
  const winT0 = S.viewStart, winT1 = S.viewStart + S.viewDur;
  const fullSpan = winT0 <= 0 && winT1 >= S.duration - 1e-6;   // whole recording selected
  const inWin = new Uint8Array(nv);
  let nIn = 0;
  for (let k = 0; k < nv; k++) {
    const c = _proj.calls[vis[k]];
    const w = fullSpan || !(c.t1 < winT0 || c.t0 > winT1);
    inWin[k] = w ? 1 : 0; if (w) nIn++;
  }

  // Group by colour (separate dim / bright buckets) to minimise fillStyle churn.
  const r  = Math.max(1.4 * dpr, 1.6);
  const rb = r * 1.25;
  const dimByColor = new Map(), brightByColor = new Map();
  for (let k = 0; k < nv; k++) {
    const col = _proj.calls[vis[k]].color || '#888';
    const m = inWin[k] ? brightByColor : dimByColor;
    let arr = m.get(col); if (!arr) { arr = []; m.set(col, arr); }
    arr.push(k);
  }
  const _drawDots = (groups, radius) => {
    for (const [col, ks] of groups) {
      ctx.fillStyle = col;
      ctx.beginPath();
      for (const k of ks) {
        ctx.moveTo(px[k] + radius, py[k]);
        ctx.arc(px[k], py[k], radius, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  };
  // Dim pass first (under), then bright on top.
  if (!fullSpan) { ctx.globalAlpha = 0.13; _drawDots(dimByColor, r); }
  ctx.globalAlpha = (fullSpan ? (nv > 8000 ? 0.5 : 0.8) : (nIn > 8000 ? 0.6 : 0.9));
  _drawDots(brightByColor, fullSpan ? r : rb);
  ctx.globalAlpha = 1;

  // Playhead highlight: every displayed call under the playhead grows + blends
  // toward white by its phIntensity() — a decaying flash during playback, a slow
  // pulsation while paused.  (Shared with the spectrogram; see render.js.)
  const peak = PROJ_PULSE_PEAK * dpr;
  for (let k = 0; k < nv; k++) {
    const c  = _proj.calls[vis[k]];
    const it = phIntensity(c);
    if (it <= 0.01) continue;
    const rad = r + (peak - r) * it;
    ctx.globalAlpha = 0.85 + 0.15 * it;
    ctx.fillStyle   = _mixWhite(c.color || '#888', 0.85 * it);
    ctx.beginPath(); ctx.arc(px[k], py[k], rad, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Rubber-band-selected calls: cyan ring on each (matches the spectrogram +
  // overview highlight).
  if (S.selectedCalls && S.selectedCalls.size) {
    ctx.strokeStyle = PROJ_SEL_COLOR; ctx.lineWidth = 1.5 * dpr;
    for (let k = 0; k < nv; k++) {
      if (!S.selectedCalls.has(_proj.calls[vis[k]])) continue;
      ctx.beginPath(); ctx.arc(px[k], py[k], r + 2.5 * dpr, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // Highlight the currently-selected call, if it's in the visible subset
  if (S.selectedCall) {
    const ci = _proj.calls.indexOf(S.selectedCall);
    const pos = ci >= 0 ? vis.indexOf(ci) : -1;
    if (pos >= 0) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * dpr;
      ctx.beginPath(); ctx.arc(px[pos], py[pos], r + 3 * dpr, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // Feature-axis overlay (biplot): each feature drawn as an arrow showing the
  // direction a point moves when that feature increases by one standard
  // deviation.  Length ∝ how much the feature aligns with the current 2-D view;
  // features orthogonal to both axes shrink to nothing.
  const biplotEl = document.getElementById('proj-biplot');
  if (!umapMode && biplotEl && biplotEl.checked) {   // biplot is linear-only
    _projDrawBiplot(ctx, dpr, xAxis, yAxis,
                    { sx, sy, xmin, xmax, ymin, ymax, padL, padT, plotW, plotH });
  }

  // Axis labels
  ctx.fillStyle = '#aaa';
  ctx.font = `${11 * dpr}px system-ui,sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText(_projAxisLabel(xAxis), padL + plotW / 2, H - 6 * dpr);
  ctx.save();
  ctx.translate(12 * dpr, padT + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText(_projAxisLabel(yAxis), 0, 0);
  ctx.restore();

  // Min/max tick values
  ctx.fillStyle = '#666';
  ctx.font = `${9 * dpr}px monospace`;
  ctx.textAlign = 'left';  ctx.textBaseline = 'top';
  ctx.fillText(xmin.toPrecision(3), padL + 2 * dpr, H - padB + 2 * dpr);
  ctx.textAlign = 'right';
  ctx.fillText(xmax.toPrecision(3), W - padR, H - padB + 2 * dpr);
  ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
  ctx.fillText(ymax.toPrecision(3), padL - 3 * dpr, padT + 8 * dpr);
  ctx.textBaseline = 'top';
  ctx.fillText(ymin.toPrecision(3), padL - 3 * dpr, padT + plotH - 8 * dpr);

  // Count — top-left (the feature panel occupies the top-right corner).
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = `${10 * dpr}px system-ui,sans-serif`;
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  let countLbl = fullSpan
    ? `${nv.toLocaleString()} calls`
    : `${nIn.toLocaleString()} of ${nv.toLocaleString()} in window`;
  if (S.selectedCalls && S.selectedCalls.size) countLbl += ` · ${S.selectedCalls.size.toLocaleString()} selected`;
  ctx.fillText(countLbl, padL + 4 * dpr, padT + 2 * dpr);

  // Rubber-band rectangle (live, while dragging).
  if (_projRubber && _projRubber.moved) {
    const xa = Math.min(_projRubber.x0, _projRubber.x1) * dpr;
    const xb = Math.max(_projRubber.x0, _projRubber.x1) * dpr;
    const ya = Math.min(_projRubber.y0, _projRubber.y1) * dpr;
    const yb = Math.max(_projRubber.y0, _projRubber.y1) * dpr;
    ctx.save();
    ctx.fillStyle = 'rgba(0,229,255,0.08)';
    ctx.fillRect(xa, ya, xb - xa, yb - ya);
    ctx.strokeStyle = PROJ_SEL_COLOR; ctx.lineWidth = 1 * dpr;
    ctx.setLineDash([4 * dpr, 3 * dpr]);
    ctx.strokeRect(xa + 0.5, ya + 0.5, xb - xa - 1, yb - ya - 1);
    ctx.restore();
  }
}

// ── Hit-testing (hover tooltip + click-to-jump) ──────────────────────────────

function _projNearest(mx, my, maxDistPx) {
  if (!_proj || !_proj.px) return -1;
  const dpr = (document.getElementById('proj-canvas')._dpr || 1);
  const x = mx * dpr, y = my * dpr;
  const lim = (maxDistPx * dpr) ** 2;
  let best = -1, bestD = lim;
  const px = _proj.px, py = _proj.py;
  for (let k = 0; k < px.length; k++) {
    const dx = px[k] - x, dy = py[k] - y, d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = k; }
  }
  // Map visible-subset position back to the call index.
  return best >= 0 ? _proj.vis[best] : -1;
}

// ── Rubber-band selection ─────────────────────────────────────────────────────
// Drag a rectangle to select every call inside it; the selection is highlighted
// in cyan here AND in the spectrogram + time overview (see render.js).  Shift- or
// Cmd-drag adds to the current selection; a plain drag replaces it (an empty drag
// clears it).  A press that doesn't move stays a click (jump-to-call).
const PROJ_SEL_COLOR = '#00e5ff';   // matches SEL_COLOR in render.js
const PROJ_DRAG_THRESH = 4;         // px before a press becomes a rubber-band
let _projRubber = null;             // { x0,y0,x1,y1, moved, additive } in CSS px
let _projSuppressClick = false;

function _projOnDown(ev) {
  if (ev.button !== 0 || !_proj) return;
  const cv = document.getElementById('proj-canvas');
  const rect = cv.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  _projRubber = { x0: x, y0: y, x1: x, y1: y, moved: false,
                  additive: ev.shiftKey || ev.metaKey || ev.ctrlKey, rect };
  document.getElementById('proj-tip').style.display = 'none';
  window.addEventListener('mousemove', _projOnDragMove);
  window.addEventListener('mouseup', _projOnUp);
}

function _projOnDragMove(ev) {
  const r = _projRubber; if (!r) return;
  const cv = document.getElementById('proj-canvas');
  r.x1 = Math.max(0, Math.min(cv.clientWidth,  ev.clientX - r.rect.left));
  r.y1 = Math.max(0, Math.min(cv.clientHeight, ev.clientY - r.rect.top));
  if (Math.hypot(r.x1 - r.x0, r.y1 - r.y0) > PROJ_DRAG_THRESH) r.moved = true;
  _projForceRender();
}

function _projOnUp() {
  window.removeEventListener('mousemove', _projOnDragMove);
  window.removeEventListener('mouseup', _projOnUp);
  const r = _projRubber;
  if (r && r.moved) { _projFinalizeSelection(r); _projSuppressClick = true; }
  _projRubber = null;
  _projForceRender();
}

function _projFinalizeSelection(r) {
  if (!_proj || !_proj.px) return;
  const cv = document.getElementById('proj-canvas');
  const dpr = cv._dpr || 1;
  const xa = Math.min(r.x0, r.x1) * dpr, xb = Math.max(r.x0, r.x1) * dpr;
  const ya = Math.min(r.y0, r.y1) * dpr, yb = Math.max(r.y0, r.y1) * dpr;
  if (!r.additive) S.selectedCalls.clear();
  const px = _proj.px, py = _proj.py, vis = _proj.vis;
  for (let k = 0; k < px.length; k++) {
    if (px[k] >= xa && px[k] <= xb && py[k] >= ya && py[k] <= yb)
      S.selectedCalls.add(_proj.calls[vis[k]]);
  }
  if (typeof scheduleRender === 'function') scheduleRender();   // refresh spectrogram + overview
}

function _projOnMove(ev) {
  if (_projRubber) return;   // drag in progress → no hover tooltip
  const cv = document.getElementById('proj-canvas');
  const rect = cv.getBoundingClientRect();
  const i = _projNearest(ev.clientX - rect.left, ev.clientY - rect.top, 8);
  const tip = document.getElementById('proj-tip');
  if (i < 0) { tip.style.display = 'none'; cv.style.cursor = 'crosshair'; return; }
  const c = _proj.calls[i];
  tip.style.display = 'block';
  tip.style.left = (ev.clientX + 12) + 'px';
  tip.style.top  = (ev.clientY + 12) + 'px';
  tip.innerHTML = `<b style="color:${c.color}">${c.short || c.species}</b> · #${c.id}<br>` +
    `${c.dur?.toFixed?.(1) ?? c.dur} ms · ${c.Fmin}–${c.Fmax} kHz · peak ${(+c.Fpeak).toFixed(1)}`;
  cv.style.cursor = 'pointer';
}

function _projOnClick(ev) {
  if (_projSuppressClick) { _projSuppressClick = false; return; }   // tail of a rubber-band drag
  const cv = document.getElementById('proj-canvas');
  const rect = cv.getBoundingClientRect();
  const i = _projNearest(ev.clientX - rect.left, ev.clientY - rect.top, 8);
  if (i < 0) return;
  const c = _proj.calls[i];
  // Select the call (its time will land inside the current window) and refresh
  // the highlight; stay on the plot tab so the user keeps their place.
  if (typeof jumpToCallId === 'function') jumpToCallId(c.id, false);
  if (_proj) _proj._key = '';
  _projRender();
}

// Axis / biplot change → redraw immediately.
function _projForceRender() {
  if (_proj) _proj._key = '';
  _projRender();
}

// ── Feature-selection panel (which features feed the PCA) ─────────────────────

function _projBuildFeaturePanel() {
  const list = document.getElementById('pfp-list');
  if (!list) return;
  list.innerHTML = '';
  for (const f of _PROJ_FEATURES) {
    const lbl = document.createElement('label');
    const cb  = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !_projExcluded.has(f.key);
    cb.dataset.key = f.key;
    cb.addEventListener('change', () => {
      if (cb.checked) _projExcluded.delete(f.key); else _projExcluded.add(f.key);
      _projSaveExcluded();
      _projInvalidate();      // rebuild PCA / UMAP with the new feature subset
      _projRebuild();
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + f.label));
    list.appendChild(lbl);
  }
}

// ── Wiring ───────────────────────────────────────────────────────────────────

(function _projInit() {
  function bind() {
    const x = document.getElementById('proj-x');
    const y = document.getElementById('proj-y');
    const cv = document.getElementById('proj-canvas');
    if (!x || !y || !cv) { setTimeout(bind, 100); return; }
    _projBuildFeaturePanel();
    setProjMode(S.projMode);   // sync toggle / control visibility to persisted mode
    x.addEventListener('change', _projForceRender);
    y.addEventListener('change', _projForceRender);
    const bp = document.getElementById('proj-biplot');
    if (bp) bp.addEventListener('change', _projForceRender);
    cv.addEventListener('mousedown', _projOnDown);
    cv.addEventListener('mousemove', _projOnMove);
    cv.addEventListener('mouseleave', () => {
      document.getElementById('proj-tip').style.display = 'none';
    });
    cv.addEventListener('click', _projOnClick);
    window.addEventListener('resize', () => {
      if (S.activeTab === 'plot') { _projResizeCanvas(); _projForceRender(); }
    });
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', bind);
  else bind();
})();
