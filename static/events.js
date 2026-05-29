// ─── Resize ──────────────────────────────────────────────────
function resize() {
  const cr = canvas.getBoundingClientRect();
  canvas.width  = Math.max(1, Math.round(cr.width));
  canvas.height = Math.max(1, Math.round(cr.height));
  S.tileWarpCache.clear();      // height changed → pre-warped tiles are stale
  S.maskTileWarpCache.clear();
  S.flatTileWarpCache.clear();
  // PSD canvas shares same height as main canvas
  const pcr = psdCanvas.getBoundingClientRect();
  psdCanvas.width  = Math.max(1, Math.round(pcr.width));
  psdCanvas.height = canvas.height;
  ovCanvas.width  = document.getElementById('overview-wrap').getBoundingClientRect().width;
  ovCanvas.height = OV_H;
  updateScrollbar();
  scheduleRender();
}

// ─── Events ──────────────────────────────────────────────────
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  // Normalise deltaY across deltaMode units, then cap at ±200 px-equivalents
  // so a single big trackpad flick doesn't teleport the view.
  let delta = e.deltaY;
  if (e.deltaMode === 1) delta *= 20;   // line mode → px
  if (e.deltaMode === 2) delta *= 400;  // page mode → px
  delta = Math.sign(delta) * Math.min(Math.abs(delta), 200);

  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;

  // ── Shift+scroll → pan frequency ──────────────────────────
  if (e.shiftKey) {
    // frac: 50 scroll-px = 1 full visible span.
    // _freqPan blends linear (logScale=0) and multiplicative/log (logScale=1)
    // so both edges shift by the same amount in whichever scale is active.
    const frac = delta / 50;
    const { fH, fL } = _freqPan(S.freqHigh, S.freqLow, frac);
    S.freqHigh = fH;
    S.freqLow  = fL;
    updateScrollbar();
    scheduleRender();
    return;
  }

  // ── Scroll over Y-axis (freq labels) → zoom frequency ─────
  if (mx < YAXIS_W) {
    const factor   = Math.pow(1.0025, delta);
    const relY     = (e.clientY - rect.top) / canvas.height;
    const fCursor  = yToF(e.clientY - rect.top);
    const freqSpan = S.freqHigh - S.freqLow;
    const newSpan  = Math.max(2, Math.min(S.nyquist - TILE_FREQ_LOW, freqSpan * factor));
    S.freqHigh = Math.min(S.nyquist, Math.max(newSpan + TILE_FREQ_LOW, fCursor + relY * newSpan));
    S.freqLow  = Math.max(TILE_FREQ_LOW, S.freqHigh - newSpan);
    updateScrollbar();
    scheduleRender();
    return;
  }

  // ── Default: zoom time ─────────────────────────────────────
  // Exponential zoom: 1.0025 per pixel gives ~1.65× per 200-px swipe — smooth.
  const factor  = Math.pow(1.0025, delta);
  const relX    = (mx - YAXIS_W) / (canvas.width - YAXIS_W);
  const tCursor = S.viewStart + relX * S.viewDur;
  S.viewDur     = Math.max(0.5, Math.min(S.duration, S.viewDur * factor));
  S.viewStart   = Math.max(0, Math.min(S.duration - S.viewDur, tCursor - relX * S.viewDur));
  scheduleRender();
}, { passive: false });

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  // "Zoom to selection" button on the fixed ruler info box
  if (S.rulerFixed && _rulerBtnRect) {
    const b = _rulerBtnRect;
    if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
      // Set time bounds exactly from ruler selection
      const t0 = xToT(Math.min(S.rulerX0, S.rulerX1));
      const t1 = xToT(Math.max(S.rulerX0, S.rulerX1));
      S.viewStart = Math.max(0, t0);
      S.viewDur   = Math.min(S.duration - S.viewStart, Math.max(0.1, t1 - t0));
      // Set frequency bounds exactly — smaller y is higher on canvas = higher freq
      S.freqHigh  = yToF(Math.min(S.rulerY0, S.rulerY1));
      S.freqLow   = yToF(Math.max(S.rulerY0, S.rulerY1));
      updateScrollbar();
      // Dismiss the selection box
      S.rulerFixed  = false;
      S.rulerLoopT0 = null;
      S.rulerLoopT1 = null;
      S.rulerLoopF0 = null;
      S.rulerLoopF1 = null;
      audioUpdateBPF();
      _rulerBtnRect = null;
      scheduleRender();
      return;  // don't start a new ruler
    }
  }

  // Cmd/Meta+drag → pan view in time and frequency
  if (e.metaKey) {
    _panDrag = true;
    _panX0   = e.clientX; _panY0 = e.clientY;
    _panVS0  = S.viewStart; _panVD0 = S.viewDur;
    _panFL0  = S.freqLow;   _panFH0 = S.freqHigh;
    canvas.style.cursor = 'grabbing';
    return;
  }

  if (mx < YAXIS_W) return;       // click on freq-axis column: ignore

  // Playhead drag: grab zone around the playhead line (incl. handle at top)
  const phX = tToX(S.playheadTime);
  if (Math.abs(mx - phX) <= PLAYHEAD_GRAB_PX) {
    _phDrag = true;
    S.isDraggingPlayhead = true;
    // Handle triangle (top strip) → grabbing; line elsewhere → ew-resize
    canvas.style.cursor = (my < 12 && Math.abs(mx - phX) <= 8) ? 'grabbing' : 'ew-resize';
    return;
  }

  S.isRuling   = true;
  S.rulerFixed = false;
  S.rulerX0 = S.rulerX1 = mx;
  S.rulerY0 = S.rulerY1 = Math.max(0, Math.min(canvas.height, my));
});

window.addEventListener('mousemove', e => {
  // PSD transport drag
  if (_psdDrag) {
    const dy  = e.clientY - _psdY0;
    const dF  = -dy / psdCanvas.height * ((psdViewHigh ?? S.nyquist) - psdViewLow);  // up = higher freq
    const MIN = 2;
    if (_psdDrag === 'top') {
      S.freqHigh = Math.max(_psdFL0 + MIN, Math.min(S.nyquist, _psdFH0 + dF));
    } else if (_psdDrag === 'bot') {
      S.freqLow  = Math.min(_psdFH0 - MIN, Math.max(TILE_FREQ_LOW, _psdFL0 + dF));
    } else {  // pan
      // Convert the linear PSD-pixel delta to a fraction of the visible span,
      // then apply the blended log/linear pan so both edges move equally in
      // whichever scale is active (S.logScale=0 → same kHz; =1 → same ratio).
      const span = _psdFH0 - _psdFL0;
      const frac = span > 0 ? dF / span : 0;
      const { fH, fL } = _freqPan(_psdFH0, _psdFL0, frac);
      S.freqHigh = fH;
      S.freqLow  = fL;
    }
    updateScrollbar(); scheduleRender();
    return;
  }

  // Cmd+drag pan
  if (_panDrag) {
    const dx = e.clientX - _panX0;
    const dy = e.clientY - _panY0;
    const specW = canvas.width - YAXIS_W;
    // Time: dragging right means content moves left = view moves earlier
    const dt = -dx / specW * _panVD0;
    S.viewStart = Math.max(0, Math.min(S.duration - _panVD0, _panVS0 + dt));
    S.viewDur   = _panVD0;
    // Frequency: dragging down = pan toward lower freq (positive dy → lower).
    // frac = dy/height means dragging the full canvas height = one full visible span.
    // _freqPan blends linear (logScale=0) and multiplicative (logScale=1) movement
    // so both edges shift by the same amount in whichever scale is active.
    const frac = dy / canvas.height;
    const { fH, fL } = _freqPan(_panFH0, _panFL0, frac);
    S.freqHigh = fH;
    S.freqLow  = fL;
    updateScrollbar();
    scheduleRender();
    return;
  }

  // Overview playhead handle drag
  if (_ovPhDrag) {
    const ox = e.clientX - ovCanvas.getBoundingClientRect().left;
    S.playheadTime = Math.max(0, Math.min(S.duration, ovXT(ox)));
    _scrubSeek();
    scheduleRender(); return;
  }

  // Overview transport drag
  if (_ovDrag) {
    const OW  = ovCanvas.width;
    const dx  = e.clientX - _ovX0;
    const dt  = dx / OW * (S.ovDur || S.duration);  // time delta scaled to overview zoom
    const MIN = 0.5;
    if (_ovDrag === 'pan') {
      S.viewStart = Math.max(0, Math.min(S.duration - _ovVD0, _ovVS0 + dt));
      S.viewDur   = _ovVD0;
    } else if (_ovDrag === 'left') {
      const viewEnd  = _ovVS0 + _ovVD0;
      const newStart = Math.max(0, Math.min(viewEnd - MIN, _ovVS0 + dt));
      S.viewStart    = newStart;
      S.viewDur      = viewEnd - newStart;
    } else if (_ovDrag === 'right') {
      const newEnd = Math.max(_ovVS0 + MIN, Math.min(S.duration, _ovVS0 + _ovVD0 + dt));
      S.viewStart  = _ovVS0;
      S.viewDur    = newEnd - _ovVS0;
    }
    scheduleRender(); return;
  }

  // Playhead drag
  if (_phDrag) {
    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    S.playheadTime = Math.max(0, Math.min(S.duration, xToT(mx)));
    _scrubSeek();
    scheduleRender();
    return;
  }

  // Ruler rubber-band drag (main canvas)
  if (S.isRuling) {
    const rect = canvas.getBoundingClientRect();
    S.rulerX1 = Math.max(YAXIS_W,     Math.min(canvas.width,  e.clientX - rect.left));
    S.rulerY1 = Math.max(0,            Math.min(canvas.height, e.clientY - rect.top));
    scheduleRender();
  }
  updateHover(e);
});

window.addEventListener('mouseup', e => {
  if (_psdDrag) {
    _psdDrag = null;
    _psdHoverY = null;
    psdCanvas.style.cursor = 'ns-resize';
    drawPSD();
    return;
  }
  if (_panDrag) {
    _panDrag = false;
    canvas.style.cursor = S.hoveredCall ? 'pointer' : 'crosshair';
    return;
  }
  if (_phDrag) {
    _phDrag = false;
    S.isDraggingPlayhead = false;
    clearTimeout(_scrubTimer);
    canvas.style.cursor = S.hoveredCall ? 'pointer' : 'crosshair';
    // Final precise seek (not scrub: use full 1 s prebuffer for clean playback)
    if (S.isPlaying && typeof audioSeek === 'function') audioSeek(S.playheadTime);
    scheduleRender();
    return;
  }
  if (_ovPhDrag) {
    _ovPhDrag = false;
    S.isDraggingPlayhead = false;
    clearTimeout(_scrubTimer);
    ovCanvas.style.cursor = 'default';
    if (S.isPlaying && typeof audioSeek === 'function') audioSeek(S.playheadTime);
    else scheduleRender();
    return;
  }
  if (_ovDrag) {
    _ovDrag = null;
    ovCanvas.style.cursor = 'default';
    return;
  }
  if (!S.isRuling) return;
  S.isRuling = false;
  const moved = Math.hypot(S.rulerX1 - S.rulerX0, S.rulerY1 - S.rulerY0);
  if (moved < 5) {
    S.rulerFixed  = false;   // tiny drag → treat as click, discard ruler
    S.rulerLoopT0 = null;
    S.rulerLoopT1 = null;
    S.rulerLoopF0 = null;
    S.rulerLoopF1 = null;
    audioUpdateBPF();
    handleClick(e);
  } else {
    S.rulerFixed   = true;    // real drag → leave ruler on screen
    S.rulerLoopT0  = xToT(Math.min(S.rulerX0, S.rulerX1));
    S.rulerLoopT1  = xToT(Math.max(S.rulerX0, S.rulerX1));
    S.rulerLoopF0  = yToF(Math.max(S.rulerY0, S.rulerY1));   // kHz, lower freq
    S.rulerLoopF1  = yToF(Math.min(S.rulerY0, S.rulerY1));   // kHz, upper freq
    audioUpdateBPF();
  }
  canvas.style.cursor = S.hoveredCall ? 'pointer' : 'crosshair';
  scheduleRender();
});

canvas.addEventListener('mouseleave', () => {
  S.mouseX = -1; S.mouseY = -1;
  if (S.hoveredCall) { S.hoveredCall = null; hideTooltip(); }
  scheduleRender();
});

// Double-click on spectrogram → place playhead at that time
canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  if (mx < YAXIS_W) return;
  const t = xToT(mx);
  if (typeof audioSeek === 'function') audioSeek(t);
  else { S.playheadTime = Math.max(0, Math.min(S.duration, t)); scheduleRender(); }
});

// ─── Hit-testing ─────────────────────────────────────────────
// Returns the call whose bounding box is closest to (mx, my) in canvas pixels,
// as long as that distance is ≤ S.pickRadius.  Uses binary search on t0 to
// avoid scanning all 10k+ calls on every mousemove.
function _hitTest(mx, my) {
  const N      = S.pickRadius;
  const specW  = canvas.width - YAXIS_W;
  // Convert N pixels → seconds so we can bound the binary search window.
  const tol_t  = N * S.viewDur / Math.max(specW, 1);
  const t      = xToT(mx);

  let found     = null;
  let foundDist = N + 1;   // one beyond threshold — any dist ≤ N beats this

  // Start a little before (t - tol_t) to catch long calls that started earlier.
  const si = Math.max(0, callsLowerBound(t - tol_t - 0.15));
  for (let i = si; i < S.calls.length; i++) {
    const c = S.calls[i];
    // c.t0 in pixel space is > mx + N → can't be within N px → stop.
    if (c.t0 > t + tol_t + 0.01) break;
    if (S.hiddenSpecies.has(c.species)) continue;

    // Bounding box in canvas pixels
    const cx0 = tToX(c.t0), cx1 = tToX(c.t1);
    const cy0 = fToY(c.Fmax), cy1 = fToY(c.Fmin);  // y0=top (high freq)

    // Distance from point to axis-aligned rect: 0 if inside, else Euclidean
    // to nearest edge.  max(a, 0, b) = positive part of the outside gap.
    const dx   = Math.max(cx0 - mx, 0, mx - cx1);
    const dy   = Math.max(cy0 - my, 0, my - cy1);
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < foundDist) { found = c; foundDist = dist; }
  }
  return found;
}

function updateHover(e) {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;

  if (mx < YAXIS_W || mx > canvas.width || my < 0 || my > SPEC_H()) {
    S.mouseX = -1; S.mouseY = -1;
    if (S.hoveredCall) { S.hoveredCall = null; hideTooltip(); }
    scheduleRender();
    return;
  }

  S.mouseX = mx; S.mouseY = my;

  const found = _hitTest(mx, my);
  if (found !== S.hoveredCall) {
    S.hoveredCall = found;
    if (found) showTooltip(found, e.clientX, e.clientY);
    else hideTooltip();
  }

  // Cursor priority: ruler lock-out → playhead handle → playhead line → call → default
  if (!S.isRuling) {
    const phX = tToX(S.playheadTime);
    if (my < 12 && Math.abs(mx - phX) <= 8) {
      canvas.style.cursor = 'grab';
    } else if (Math.abs(mx - phX) <= PLAYHEAD_GRAB_PX) {
      canvas.style.cursor = 'ew-resize';
    } else {
      canvas.style.cursor = found ? 'pointer' : 'crosshair';
    }
  }
  scheduleRender();
}

function handleClick(e) {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;
  const found = _hitTest(mx, my);
  S.selectedCall = (found === S.selectedCall) ? null : found;
  // Advance playhead to selected call
  if (S.selectedCall) {
    S.playheadTime = S.selectedCall.t0;
    if (S.isPlaying && typeof audioSeek === 'function') audioSeek(S.selectedCall.t0);
  }
  renderDetail(S.selectedCall);
  scheduleRender();
}

// ─── Playhead drag ────────────────────────────────────────────
const PLAYHEAD_GRAB_PX = 9;   // px hit zone around the playhead line
let _phDrag = false;

// Scrub-seek: debounce seeks while the user drags so we don't hammer audioSeek
// on every pixel.  200 ms of no movement triggers a scrub seek; mouseup always
// does a final precise seek regardless.
let _scrubTimer = null;
function _scrubSeek() {
  if (!S.isPlaying) return;
  clearTimeout(_scrubTimer);
  _scrubTimer = setTimeout(() => {
    if (typeof audioSeek === 'function') audioSeek(S.playheadTime, /*scrub=*/true);
  }, 200);
}

// ─── Overview transport drag ──────────────────────────────────
// All positions are in the overview's own fixed coordinate system
// (ox / OW * duration = time), fully independent of viewStart/viewDur.
let _ovDrag   = null;   // 'left' | 'right' | 'pan' | 'jump' | null
let _ovPhDrag = false;  // dragging the playhead handle in the overview
let _ovX0 = 0, _ovVS0 = 0, _ovVD0 = 0;
const OV_EDGE_PX = 7;  // px grab zone for each edge handle
let _rulerBtnRect = null;  // bounding box of the ruler "Zoom to selection" button
let _bpfAttPos    = null;  // {x, y, w} canvas-space position for the BPF attenuation overlay

function ovHitTest(ox) {
  const vx0 = ovTX(S.viewStart);
  const vx1 = ovTX(S.viewStart + S.viewDur);
  if (Math.abs(ox - vx0) <= OV_EDGE_PX) return 'left';
  if (Math.abs(ox - vx1) <= OV_EDGE_PX) return 'right';
  if (ox > vx0 && ox < vx1)             return 'pan';
  return 'jump';
}

ovCanvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  e.preventDefault();
  const rect = ovCanvas.getBoundingClientRect();
  const ox   = e.clientX - rect.left;
  const oy   = e.clientY - rect.top;

  // Cmd+click on overview → seek playhead without panning the view
  if (e.metaKey) {
    const t = ovXT(ox);
    if (typeof audioSeek === 'function') audioSeek(t);
    else { S.playheadTime = Math.max(0, Math.min(S.duration, t)); scheduleRender(); }
    return;
  }

  // Playhead triangle handle (bottom strip of the overview canvas)
  const phOX = ovTX(S.playheadTime);
  if (Math.abs(ox - phOX) <= 10 && oy <= 14) {
    _ovPhDrag = true;
    S.isDraggingPlayhead = true;
    ovCanvas.style.cursor = 'grabbing';
    return;
  }

  _ovDrag   = ovHitTest(ox);
  _ovX0     = e.clientX;
  _ovVS0    = S.viewStart;
  _ovVD0    = S.viewDur;
  if (_ovDrag === 'jump') {
    const t = ovXT(ox);
    S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, t - S.viewDur / 2));
    _ovDrag = 'pan';
    _ovVS0  = S.viewStart;
    scheduleRender();
  }
  ovCanvas.style.cursor = _ovDrag === 'pan' ? 'grabbing' : 'ew-resize';
});

ovCanvas.addEventListener('mousemove', e => {
  if (_ovDrag || _ovPhDrag) return;  // cursor already set
  const rect = ovCanvas.getBoundingClientRect();
  const ox   = e.clientX - rect.left;
  const oy   = e.clientY - rect.top;
  // Hovering over the playhead triangle handle?
  const phOX = ovTX(S.playheadTime);
  if (Math.abs(ox - phOX) <= 10 && oy <= 14) {
    ovCanvas.style.cursor = 'grab';
    return;
  }
  const hit = ovHitTest(ox);
  ovCanvas.style.cursor = (hit === 'left' || hit === 'right') ? 'ew-resize'
                        : hit === 'pan' ? 'grab' : 'default';
});

// Keyboard
window.addEventListener('keydown', e => {
  // Space → play/pause (skip when focus is in a text input)
  if (e.key === ' ' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
    e.preventDefault();
    if (typeof audioPlayPause === 'function') audioPlayPause();
    return;
  }
  const step = S.viewDur * 0.25;
  if (e.key === 'ArrowRight') S.viewStart = Math.min(S.duration - S.viewDur, S.viewStart + step);
  if (e.key === 'ArrowLeft')  S.viewStart = Math.max(0, S.viewStart - step);
  if (e.key === '+'||e.key==='=') zoomBy(0.7);
  if (e.key === '-')           zoomBy(1.4);
  if (e.key === 'Escape')      { S.rulerFixed = false; S.rulerLoopT0 = null; S.rulerLoopT1 = null; S.rulerLoopF0 = null; S.rulerLoopF1 = null; audioUpdateBPF(); }
  scheduleRender();
});

// ─── Overview: scroll-wheel zoom + double-click reset ────────────────
ovCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  let delta = e.deltaY;
  if (e.deltaMode === 1) delta *= 20;
  if (e.deltaMode === 2) delta *= 400;
  delta = Math.sign(delta) * Math.min(Math.abs(delta), 200);
  const factor  = Math.pow(1.0025, delta);
  const ox      = e.clientX - ovCanvas.getBoundingClientRect().left;
  const tCursor = ovXT(ox);
  const relX    = ox / ovCanvas.width;
  S.ovDur   = Math.max(2.0, Math.min(S.duration, (S.ovDur || S.duration) * factor));
  S.ovStart = Math.max(0, Math.min(S.duration - S.ovDur, tCursor - relX * S.ovDur));
  scheduleRender();
}, { passive: false });

ovCanvas.addEventListener('dblclick', () => {
  // Double-click resets the overview zoom to the full recording
  S.ovStart = 0;
  S.ovDur   = S.duration;
  scheduleRender();
});

document.getElementById('btn-zoom-in').onclick  = () => zoomBy(0.6);
document.getElementById('btn-zoom-out').onclick = () => zoomBy(1.6);
document.getElementById('btn-fit').onclick      = () => { S.viewStart = 0; S.viewDur = S.duration; scheduleRender(); };
document.getElementById('btn-prev-call').onclick = () => navigateCall(-1);
document.getElementById('btn-next-call').onclick = () => navigateCall(+1);
document.getElementById('input-call-id').addEventListener('keydown', e => {
  if (e.key === 'Enter') { jumpToCallId(parseInt(e.target.value)); e.target.blur(); }
  if (e.key === 'ArrowUp')   { e.preventDefault(); navigateCall(-1); }
  if (e.key === 'ArrowDown') { e.preventDefault(); navigateCall(+1); }
});
document.getElementById('chk-contour').onchange = e => { S.showContour = e.target.checked; scheduleRender(); };
document.getElementById('chk-boxes').onchange   = e => { S.showBoxes   = e.target.checked; scheduleRender(); };
document.getElementById('slider-contour-alpha').oninput = e => {
  S.contourAlpha = e.target.value / 100;
  document.getElementById('contour-alpha-val').textContent = e.target.value + '%';
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-crossfade').oninput = e => {
  const wasZero = S.crossfade === 0;
  S.crossfade = e.target.value / 100;
  if (wasZero && S.crossfade > 0) ensureTiles();
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-flatness').oninput = e => {
  const wasZero = S.flatness === 0;
  S.flatness = e.target.value / 100;
  if (wasZero && S.flatness > 0) ensureTiles();
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-log').oninput = e => {
  S.logScale = e.target.value / 100;
  S.tileWarpCache.clear();
  S.maskTileWarpCache.clear();
  S.flatTileWarpCache.clear();
  updateTrack(e.target);
  scheduleRender();
  drawPSD();   // PSD uses same log blend — update immediately
};
document.getElementById('slider-sat').oninput = e => {
  S.saturation = e.target.value / 100;
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-min-conf').oninput = e => {
  S.minConf = e.target.value / 100;
  document.getElementById('min-conf-val').textContent = e.target.value + '%';
  updateTrack(e.target);
  scheduleRender();
};

// ─── Playback controls ────────────────────────────────────────
// Rate slider is initialised by _initRateSlider() (player.js) which is
// called from init() in ui.js after the DOM is ready.

// ─── Frequency scrollbar ──────────────────────────────────────
// Coordinate system: y=0 = Nyquist (top), y=trackH = 0 Hz (bottom).
// This is INDEPENDENT of S.freqLow/freqHigh, fixing the feedback-loop bug.
const sbTrack  = document.getElementById('freq-sb-track');
const sbFill   = document.getElementById('freq-sb-fill');
const sbTop    = document.getElementById('freq-sb-top');
const sbBot    = document.getElementById('freq-sb-bot');

let _sbDrag = null;  // null | 'top' | 'bot' | 'pan'
let _sbY0 = 0, _sbHi0 = 0, _sbLo0 = 0;

function sbTrackH() { return sbTrack.getBoundingClientRect().height; }

function updateScrollbar() {
  const h   = sbTrackH();
  if (h === 0) return;
  const ny  = S.nyquist;
  const top = (1 - S.freqHigh / ny) * h;   // y of max-freq handle
  const bot = (1 - S.freqLow  / ny) * h;   // y of min-freq handle
  sbFill.style.top    = top + 'px';
  sbFill.style.height = (bot - top) + 'px';
  sbTop.style.top     = '0px';   // relative to fill
  sbBot.style.top     = '100%';

  // Tick marks — rebuild only if nyquist changed (rare)
  if (!sbTrack._ticked) {
    sbTrack._ticked = true;
    [10,20,30,40,50,60,70,80,90].forEach(f => {
      if (f >= ny) return;
      const d = document.createElement('div');
      d.className = 'freq-sb-tick';
      d.style.top = ((1 - f / ny) * 100) + '%';
      d.title     = f + ' kHz';
      sbTrack.appendChild(d);
    });
  }
}

function sbStartDrag(type, e) {
  e.preventDefault(); e.stopPropagation();
  _sbDrag  = type;
  _sbY0    = e.clientY;
  _sbHi0   = S.freqHigh;
  _sbLo0   = S.freqLow;
}
sbTop.addEventListener('mousedown',  e => sbStartDrag('top', e));
sbBot.addEventListener('mousedown',  e => sbStartDrag('bot', e));
sbFill.addEventListener('mousedown', e => sbStartDrag('pan', e));

window.addEventListener('mousemove', e => {
  if (!_sbDrag) return;
  const h   = sbTrackH();
  const ny  = S.nyquist;
  const dy  = e.clientY - _sbY0;
  const df  = -dy / h * ny;           // upward mouse = higher freq
  const MIN_SPAN = 2;                  // kHz minimum window
  if (_sbDrag === 'top') {
    S.freqHigh = Math.max(_sbLo0 + MIN_SPAN, Math.min(ny, _sbHi0 + df));
  } else if (_sbDrag === 'bot') {
    S.freqLow  = Math.min(_sbHi0 - MIN_SPAN, Math.max(0,  _sbLo0 + df));
  } else {                             // pan: move both, keep span
    const span = _sbHi0 - _sbLo0;
    S.freqHigh = Math.min(ny,    Math.max(span, _sbHi0 + df));
    S.freqLow  = S.freqHigh - span;
  }
  updateScrollbar();
  scheduleRender();
});

window.addEventListener('mouseup', () => { _sbDrag = null; });

function zoomBy(factor) {
  const mid   = S.viewStart + S.viewDur / 2;
  S.viewDur   = Math.max(0.5, Math.min(S.duration, S.viewDur * factor));
  S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, mid - S.viewDur / 2));
}

