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
];

// State cached between renders for the currently-open projection.
let _proj = null;   // { calls, Z, means, stds, n, d, pca:{order, vecs, varRatio}, px, py }

// ── Playhead pulse animation ──────────────────────────────────────────────────
// When the playhead enters a call's [t0, t1] during playback, that call's dot in
// the scatter pulses: it jumps to PROJ_PULSE_PEAK px and eases back to its base
// size over PROJ_PULSE_MS.
const PROJ_PULSE_MS   = 600;   // pulse lifetime (ms)
const PROJ_PULSE_PEAK = 9;     // peak dot radius at pulse start (CSS px)
const PROJ_PULSATE_MS = 1500;  // sustained-highlight pulsation period (ms)
const _projPulses = new Map();  // callId → pulse start time (performance.now ms)
let   _projActive = new Set();  // calls under the playhead on the previous frame
let   _projAnimId = null;       // requestAnimationFrame id for the pulse loop
let   _projSustainedActive = false;  // a paused call is under the playhead → keep animating

// Blend a #rrggbb colour toward white by fraction t (0 = colour, 1 = white).
function _projMixWhite(hex, t) {
  let h = String(hex).replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  const r = parseInt(h.slice(0, 2), 16) || 0;
  const g = parseInt(h.slice(2, 4), 16) || 0;
  const b = parseInt(h.slice(4, 6), 16) || 0;
  const mix = (v) => Math.round(v + (255 - v) * t);
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
}

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
  if (axis.startsWith('feat:')) {
    const k = axis.slice(5);
    return (_PROJ_FEATURES.find(f => f.key === k) || {}).label || k;
  }
  const rank = +axis.slice(4);
  const pct = _proj ? Math.round(_proj.pca.varRatio[rank] * 100) : 0;
  return `PC${rank + 1} (${pct}% var)`;
}

function _projAxisValue(i, axis) {
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
function _projEnsure() {
  const n = (S.calls && S.calls.length) || 0;
  if (_proj && _proj._builtN === n) return _proj && n > 0;
  if (n === 0) { _proj = null; return false; }
  const calls = S.calls.slice();   // full population, no filters
  const keys  = _PROJ_FEATURES.map(f => f.key);
  const std   = _projStandardize(calls, keys);
  const pca   = _projPCA(std.Z, std.n, std.d);
  _proj = { calls, ...std, pca, _builtN: n, vis: null, px: null, py: null, _key: '' };
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
    _projEnsure();
    _projResizeCanvas();
    _proj && (_proj._key = '');   // force a redraw
    _projActive = new Set();      // don't pulse calls already under a paused playhead
    _projRender();
    if (S.isPlaying) _projEnsureAnim();
  } else {
    // Spectrogram tab: its canvas was hidden (0-size) while away — re-measure + draw.
    if (typeof resize === 'function') resize();
    else if (typeof scheduleRender === 'function') scheduleRender();
  }
}

// Called from the main render loop while the plot tab is active.  Re-renders the
// scatter only when something it depends on changed (viewport / filters / sel).
function _projOnMainRender() {
  if (S.activeTab !== 'plot') return;
  if (!_projEnsure()) return;
  // During playback (or while pulses are decaying) the dedicated pulse loop
  // owns rendering — don't also render here.
  if (S.isPlaying || _projPulses.size > 0) { _projEnsureAnim(); return; }
  const key = [
    S.viewStart.toFixed(3), S.viewDur.toFixed(3), S.minConf.toFixed(3),
    S.classifier, S.soloedSpecies || '', [...S.hiddenSpecies].sort().join(','),
    S.selectedCall ? S.selectedCall.id : -1,
    S.playheadTime.toFixed(3),   // paused playhead moves → refresh sustained highlight
    document.getElementById('proj-x')?.value, document.getElementById('proj-y')?.value,
    document.getElementById('proj-biplot')?.checked,
  ].join('|');
  if (key !== _proj._key) { _proj._key = key; _projRender(); }
  // Paused with a call under the playhead → run the slow pulsation loop.
  if (_projSustainedActive) _projEnsureAnim();
}

// Start a pulse for every call the playhead has just entered (t0 ≤ playhead ≤ t1).
function _projDetectPulses() {
  if (!_proj) return;
  const ph = S.playheadTime;
  const cur = new Set();
  for (let i = 0; i < _proj.n; i++) {
    const c = _proj.calls[i];
    if (c.t0 <= ph && ph <= c.t1) cur.add(c.id);
  }
  const now = performance.now();
  for (const id of cur) if (!_projActive.has(id)) _projPulses.set(id, now);  // entry only
  _projActive = cur;
}

function _projExpirePulses() {
  const now = performance.now();
  for (const [id, t] of _projPulses) if (now - t >= PROJ_PULSE_MS) _projPulses.delete(id);
}

// Self-sustaining RAF loop: redraws the scatter every frame while the playhead is
// moving, pulses are decaying, or a paused call under the playhead is pulsating.
function _projEnsureAnim() {
  if (_projAnimId != null || S.activeTab !== 'plot') return;
  const step = () => {
    if (S.activeTab !== 'plot') { _projAnimId = null; return; }
    if (S.isPlaying) _projDetectPulses();
    _projExpirePulses();
    if (_proj) _proj._key = '';     // force the scatter to repaint
    _projRender();                  // sets _projSustainedActive
    if (S.isPlaying || _projPulses.size > 0 || _projSustainedActive)
      _projAnimId = requestAnimationFrame(step);
    else _projAnimId = null;
  };
  _projAnimId = requestAnimationFrame(step);
}

function _projBuildAxisOptions() {
  const mk = () => {
    let html = '<optgroup label="Features">';
    for (const f of _PROJ_FEATURES) html += `<option value="feat:${f.key}">${f.label}</option>`;
    html += '</optgroup><optgroup label="PCA components">';
    for (let r = 0; r < _PROJ_FEATURES.length; r++)
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
    const j = _PROJ_FEATURES.findIndex(f => f.key === axis.slice(5));
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
    const j = _PROJ_FEATURES.findIndex(f => f.key === axis.slice(5));
    return j >= 0 ? _proj.means[j] : 0;
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
    const lbl = _PROJ_FEATURES[j].label.replace(/\s*\(.*\)$/, '');
    ctx.fillText(lbl, ex + (right ? 3 : -3) * dpr, ey);
  }
  ctx.restore();
}

function _projRender() {
  if (!_proj) return;
  const cv  = document.getElementById('proj-canvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, dpr = cv._dpr || 1;
  const xAxis = document.getElementById('proj-x').value;
  const yAxis = document.getElementById('proj-y').value;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, W, H);

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

  const peak = PROJ_PULSE_PEAK * dpr;

  // Playhead pulse (during playback): calls the playhead has crossed pulse from
  // the peak radius back down to the base over PROJ_PULSE_MS (quadratic ease-out).
  if (_projPulses.size) {
    const now = performance.now();
    for (const [id, start] of _projPulses) {
      const k = posById.get(id);
      if (k === undefined) continue;                 // call not currently displayed
      const prog = Math.min(1, (now - start) / PROJ_PULSE_MS);
      const ease = (1 - prog) * (1 - prog);
      const rad  = r + (peak - r) * ease;
      const c    = _proj.calls[vis[k]];
      ctx.globalAlpha = 0.85 + 0.15 * ease;
      // Blend toward white: ~85% white at the peak, easing back to the call colour.
      ctx.fillStyle   = _projMixWhite(c.color || '#888', 0.85 * ease);
      ctx.beginPath(); ctx.arc(px[k], py[k], rad, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // Sustained highlight (when NOT playing): every displayed call the playhead is
  // currently over slowly pulsates between full highlight (peak size + ~85% white)
  // and its normal dot, continuously, until the playhead moves off it.
  _projSustainedActive = false;
  if (!S.isPlaying) {
    const ph    = S.playheadTime;
    const phase = (performance.now() % PROJ_PULSATE_MS) / PROJ_PULSATE_MS * 2 * Math.PI;
    const t     = (1 - Math.cos(phase)) / 2;   // smooth 0 → 1 → 0
    const rad   = r + (peak - r) * t;
    const mix   = 0.85 * t;
    ctx.globalAlpha = 1;
    for (let k = 0; k < nv; k++) {
      const c = _proj.calls[vis[k]];
      if (c.t0 <= ph && ph <= c.t1) {
        _projSustainedActive = true;
        ctx.fillStyle = _projMixWhite(c.color || '#888', mix);
        ctx.beginPath(); ctx.arc(px[k], py[k], rad, 0, Math.PI * 2); ctx.fill();
      }
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
  if (biplotEl && biplotEl.checked) {
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

  // Count — note how many fall inside the highlighted time window.
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = `${10 * dpr}px system-ui,sans-serif`;
  ctx.textAlign = 'right'; ctx.textBaseline = 'top';
  const countLbl = fullSpan
    ? `${nv.toLocaleString()} calls`
    : `${nIn.toLocaleString()} of ${nv.toLocaleString()} in window`;
  ctx.fillText(countLbl, W - padR - 2 * dpr, padT + 2 * dpr);
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

function _projOnMove(ev) {
  const cv = document.getElementById('proj-canvas');
  const rect = cv.getBoundingClientRect();
  const i = _projNearest(ev.clientX - rect.left, ev.clientY - rect.top, 8);
  const tip = document.getElementById('proj-tip');
  if (i < 0) { tip.style.display = 'none'; cv.style.cursor = 'default'; return; }
  const c = _proj.calls[i];
  tip.style.display = 'block';
  tip.style.left = (ev.clientX + 12) + 'px';
  tip.style.top  = (ev.clientY + 12) + 'px';
  tip.innerHTML = `<b style="color:${c.color}">${c.short || c.species}</b> · #${c.id}<br>` +
    `${c.dur?.toFixed?.(1) ?? c.dur} ms · ${c.Fmin}–${c.Fmax} kHz · peak ${(+c.Fpeak).toFixed(1)}`;
  cv.style.cursor = 'pointer';
}

function _projOnClick(ev) {
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

// ── Wiring ───────────────────────────────────────────────────────────────────

(function _projInit() {
  function bind() {
    const x = document.getElementById('proj-x');
    const y = document.getElementById('proj-y');
    const cv = document.getElementById('proj-canvas');
    if (!x || !y || !cv) { setTimeout(bind, 100); return; }
    x.addEventListener('change', _projForceRender);
    y.addEventListener('change', _projForceRender);
    const bp = document.getElementById('proj-biplot');
    if (bp) bp.addEventListener('change', _projForceRender);
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
