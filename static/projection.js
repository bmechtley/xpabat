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

// ── Filtered call set (matches the main view's species/conf filters) ─────────

function _projCollectCalls() {
  if (!S.calls) return [];
  return S.calls.filter(c =>
    c.conf >= S.minConf &&
    !S.hiddenSpecies.has(c.species) &&
    (!S.soloedSpecies || S.soloedSpecies === c.species));
}

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

// ── Modal lifecycle ──────────────────────────────────────────────────────────

function openProjection() {
  document.getElementById('proj-modal').classList.add('open');

  const calls = _projCollectCalls();
  const keys  = _PROJ_FEATURES.map(f => f.key);
  const std   = _projStandardize(calls, keys);
  const pca   = _projPCA(std.Z, std.n, std.d);
  _proj = { calls, ...std, pca, px: null, py: null };

  _projBuildAxisOptions();
  _projResizeCanvas();
  _projRender();
}

function closeProjection() {
  document.getElementById('proj-modal').classList.remove('open');
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

function _projRender() {
  if (!_proj) return;
  const cv  = document.getElementById('proj-canvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, dpr = cv._dpr || 1;
  const xAxis = document.getElementById('proj-x').value;
  const yAxis = document.getElementById('proj-y').value;
  const n = _proj.n;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, W, H);

  if (n === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = `${13 * dpr}px system-ui,sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('No calls match the current filters.', W / 2, H / 2);
    return;
  }

  // Compute axis values + data ranges
  const xs = new Float64Array(n), ys = new Float64Array(n);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = _projAxisValue(i, xAxis), y = _projAxisValue(i, yAxis);
    xs[i] = x; ys[i] = y;
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

  // Cache screen coords for hit-testing
  const px = new Float32Array(n), py = new Float32Array(n);
  for (let i = 0; i < n; i++) { px[i] = sx(xs[i]); py[i] = sy(ys[i]); }
  _proj.px = px; _proj.py = py;

  // Points — group by colour to minimise fillStyle churn; alpha for density.
  const r = Math.max(1.4 * dpr, 1.6);
  const byColor = new Map();
  for (let i = 0; i < n; i++) {
    const col = _proj.calls[i].color || '#888';
    let arr = byColor.get(col); if (!arr) { arr = []; byColor.set(col, arr); }
    arr.push(i);
  }
  ctx.globalAlpha = n > 8000 ? 0.45 : 0.75;
  for (const [col, idxs] of byColor) {
    ctx.fillStyle = col;
    ctx.beginPath();
    for (const i of idxs) {
      ctx.moveTo(px[i] + r, py[i]);
      ctx.arc(px[i], py[i], r, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Highlight the currently-selected call, if present in the set
  if (S.selectedCall) {
    const i = _proj.calls.indexOf(S.selectedCall);
    if (i >= 0) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * dpr;
      ctx.beginPath(); ctx.arc(px[i], py[i], r + 3 * dpr, 0, Math.PI * 2); ctx.stroke();
    }
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

  // Count
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = `${10 * dpr}px system-ui,sans-serif`;
  ctx.textAlign = 'right'; ctx.textBaseline = 'top';
  ctx.fillText(`${n.toLocaleString()} calls`, W - padR - 2 * dpr, padT + 2 * dpr);
}

// ── Hit-testing (hover tooltip + click-to-jump) ──────────────────────────────

function _projNearest(mx, my, maxDistPx) {
  if (!_proj || !_proj.px) return -1;
  const dpr = (document.getElementById('proj-canvas')._dpr || 1);
  const x = mx * dpr, y = my * dpr;
  const lim = (maxDistPx * dpr) ** 2;
  let best = -1, bestD = lim;
  const px = _proj.px, py = _proj.py;
  for (let i = 0; i < px.length; i++) {
    const dx = px[i] - x, dy = py[i] - y, d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
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
  closeProjection();
  if (typeof jumpToCallId === 'function') jumpToCallId(c.id, true);
}

// ── Wiring ───────────────────────────────────────────────────────────────────

(function _projInit() {
  function bind() {
    const x = document.getElementById('proj-x');
    const y = document.getElementById('proj-y');
    const cv = document.getElementById('proj-canvas');
    if (!x || !y || !cv) { setTimeout(bind, 100); return; }
    x.addEventListener('change', _projRender);
    y.addEventListener('change', _projRender);
    cv.addEventListener('mousemove', _projOnMove);
    cv.addEventListener('mouseleave', () => {
      document.getElementById('proj-tip').style.display = 'none';
    });
    cv.addEventListener('click', _projOnClick);
    window.addEventListener('resize', () => {
      if (document.getElementById('proj-modal').classList.contains('open')) {
        _projResizeCanvas(); _projRender();
      }
    });
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', bind);
  else bind();
})();
