// ─── Resize ──────────────────────────────────────────────────
function resize() {
  const cr = canvas.getBoundingClientRect();
  canvas.width  = Math.max(1, Math.round(cr.width));
  canvas.height = Math.max(1, Math.round(cr.height));
  S.tileWarpCache.clear();      // height changed → pre-warped tiles are stale
  S.maskTileWarpCache.clear();
  S.flatTileWarpCache.clear();
  clearGLTextures();            // GL texture dimensions matched H; must rebuild
  // PSD overlay canvas — must match mainCanvas exactly so fToY() y-coords align
  psdCanvas.width  = canvas.width;
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
  S.viewDur     = Math.max(0.1, Math.min(S.duration, S.viewDur * factor));
  S.viewStart   = Math.max(0, Math.min(S.duration - S.viewDur, tCursor - relX * S.viewDur));
  scheduleRender();
}, { passive: false });

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (S.rulerFixed) {
    // ✕ clear button
    if (_rulerCloseRect) {
      const c = _rulerCloseRect;
      if (mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h) {
        S.rulerFixed = false;
        S.rulerLoopT0 = null; S.rulerLoopT1 = null;
        S.rulerLoopF0 = null; S.rulerLoopF1 = null;
        audioUpdateBPF();
        _rulerBtnRect = null; _rulerCloseRect = null;
        scheduleRender();
        return;
      }
    }
    // "Zoom to selection" button on the fixed ruler info box
    if (_rulerBtnRect) {
      const b = _rulerBtnRect;
      if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
        // Use loop bounds directly — they're always correct, even after panning
        S.viewStart = Math.max(0, S.rulerLoopT0);
        S.viewDur   = Math.min(S.duration - S.viewStart, Math.max(0.1, S.rulerLoopT1 - S.rulerLoopT0));
        S.freqHigh  = S.rulerLoopF1;
        S.freqLow   = S.rulerLoopF0;
        updateScrollbar();
        // Dismiss the selection box
        S.rulerFixed  = false;
        S.rulerLoopT0 = null; S.rulerLoopT1 = null;
        S.rulerLoopF0 = null; S.rulerLoopF1 = null;
        audioUpdateBPF();
        _rulerBtnRect = null; _rulerCloseRect = null;
        scheduleRender();
        return;  // don't start a new ruler
      }
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

  // Ruler resize handles (only when ruler is fixed, no Cmd held)
  if (S.rulerFixed) {
    const rHit = _rulerHitTest(mx, my);
    if (rHit) { _rulerDrag = rHit; return; }
  }

  // Freq-axis column: drag to pan frequency (no Cmd required)
  if (mx < YAXIS_W) {
    _freqAxisDrag = true;
    _faY0  = e.clientY;
    _faFH0 = S.freqHigh;
    _faFL0 = S.freqLow;
    canvas.style.cursor = 'grabbing';
    return;
  }

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
  // Freq-axis drag → pan frequency
  if (_freqAxisDrag) {
    const dy   = e.clientY - _faY0;
    const frac = dy / canvas.height;
    const { fH, fL } = _freqPan(_faFH0, _faFL0, frac);
    S.freqHigh = fH;
    S.freqLow  = fL;
    updateScrollbar();
    scheduleRender();
    return;
  }

  // Ruler resize drag — update whichever edges the handle controls
  if (_rulerDrag) {
    const rect = canvas.getBoundingClientRect();
    const mx = Math.max(YAXIS_W, Math.min(canvas.width,  e.clientX - rect.left));
    const my = Math.max(0,        Math.min(canvas.height, e.clientY - rect.top));
    // Derive current box edges from loop bounds (pixel coords may be stale after pan)
    let { x0, x1, y0, y1 } = _rulerPx();
    if (_rulerDrag.includes('n')) y0 = Math.min(y1 - 5, my);
    if (_rulerDrag.includes('s')) y1 = Math.max(y0 + 5, my);
    if (_rulerDrag.includes('w')) x0 = Math.min(x1 - 5, mx);
    if (_rulerDrag.includes('e')) x1 = Math.max(x0 + 5, mx);
    S.rulerX0 = x0; S.rulerY0 = y0; S.rulerX1 = x1; S.rulerY1 = y1;
    S.rulerLoopT0 = xToT(x0); S.rulerLoopT1 = xToT(x1);
    S.rulerLoopF0 = yToF(y1); S.rulerLoopF1 = yToF(y0);   // y1=bottom=lower freq
    audioUpdateBPF();
    const _RC = { n:'ns-resize',s:'ns-resize',e:'ew-resize',w:'ew-resize',
                  ne:'nesw-resize',sw:'nesw-resize',nw:'nwse-resize',se:'nwse-resize' };
    canvas.style.cursor = _RC[_rulerDrag];
    scheduleRender();
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

  // Cmd+drag on overview → slide the overview display window (S.ovStart) in time,
  // without touching the selected viewport.  Only meaningful when zoomed in.
  if (_ovCmdDrag) {
    const dx = e.clientX - _ovCmdDrag.x0;
    if (Math.abs(dx) > 3) _ovCmdDrag.moved = true;
    const ovD = S.ovDur || S.duration;
    if (ovD < S.duration) {                       // zoomed in → pan
      const dt = dx / ovCanvas.width * ovD;       // ovCanvas.width is in CSS px
      S.ovStart = Math.max(0, Math.min(S.duration - ovD, _ovCmdDrag.ov0 - dt));
      scheduleRender();
    }
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
  // Cmd on overview: if the mouse didn't move, it was a click → seek playhead;
  // otherwise it was a window-pan and we're done.
  if (_ovCmdDrag) {
    const wasDrag = _ovCmdDrag.moved;
    _ovCmdDrag = null;
    ovCanvas.style.cursor = 'default';
    if (!wasDrag) {
      const ox = e.clientX - ovCanvas.getBoundingClientRect().left;
      const t  = Math.max(0, Math.min(S.duration, ovXT(ox)));
      if (typeof audioSeek === 'function') audioSeek(t);
      else { S.playheadTime = t; scheduleRender(); }
    }
    return;
  }
  if (_freqAxisDrag) {
    _freqAxisDrag = false;
    canvas.style.cursor = 'crosshair';
    return;
  }
  if (_rulerDrag) {
    _rulerDrag = null;
    canvas.style.cursor = 'crosshair';
    scheduleRender();
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

// Single click on spectrogram → seek playhead
canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  if (mx < YAXIS_W) return;
  const t = xToT(mx);
  if (typeof audioSeek === 'function') audioSeek(t);
  else { S.playheadTime = Math.max(0, Math.min(S.duration, t)); scheduleRender(); }
});

// Double-click on a call → select it, zoom/centre on it (same as the call-ID
// text box), and place a marquee around its bounding box so audio playback
// loops over it and the BPF filter is centred on its frequency range.
// By the time dblclick fires, the two mouseup→handleClick() calls have toggled
// selectedCall on then off, so we re-hit-test and re-apply everything here.
canvas.addEventListener('dblclick', e => {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;
  if (mx < YAXIS_W) return;

  const c = _inRugZone(my) ? _rugHitTest(mx) : _hitTest(mx, my);
  if (!c) return;   // no call hit — playhead already seeked by the 'click' handler

  // Re-select (the second single-click deselected it)
  S.selectedCall = c;

  // Zoom and centre on the call — this updates S.viewStart/S.viewDur synchronously,
  // which is required before we compute ruler pixel coordinates below.
  zoomToCall(c);

  // Place the marquee around the call bounding box.
  // tToX/fToY use the post-zoom viewport so the box lands correctly.
  _applyCallMarquee(c);

  // Park the playhead at the call's start
  S.playheadTime = c.t0;
  if (S.isPlaying && typeof audioSeek === 'function') audioSeek(c.t0);

  renderDetail(c);
  scheduleRender();
});

// ─── Hit-testing ─────────────────────────────────────────────

// Returns true when my is inside the call-density rug strip.
// RUG_H and the geometry formula are defined in render.js.
function _inRugZone(my) {
  const rugTop = SPEC_H() - 14 - RUG_H - 2;
  return my >= rugTop && my <= rugTop + RUG_H;
}

// Rug hit-test: find the nearest visible call by pixel distance from its tick.
// The rug draws each call as a 1-px tick at tToX((t0+t1)/2), so we compare
// canvas-pixel distance directly — this avoids the time-vs-midpoint mismatch
// that caused the hover to fire ~10px to the left of the visible tick.
function _rugHitTest(mx) {
  const specW = canvas.width - YAXIS_W;
  const t     = xToT(mx);
  // Search window: 8-px in time so we don't miss any nearby call
  const tol_t = 8 * S.viewDur / Math.max(specW, 1);

  const TOL_PX = 4;   // pixel hit tolerance
  let found = null, foundDist = TOL_PX + 1;
  const si = Math.max(0, callsLowerBound(t - tol_t));
  for (let i = si; i < S.calls.length; i++) {
    const c = S.calls[i];
    if (c.t0 > t + tol_t) break;
    if (S.hiddenSpecies.has(c.species)) continue;
    if (c.conf < S.minConf) continue;
    if (S.soloedSpecies && S.soloedSpecies !== c.species) continue;
    // Match by pixel distance from the tick position — same formula as drawCallRug
    const tickX = Math.round(tToX((c.t0 + c.t1) / 2));
    const dist  = Math.abs(tickX - mx);
    if (dist < foundDist) { found = c; foundDist = dist; }
  }
  return found;
}

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
    if (c.conf < S.minConf) continue;
    if (S.soloedSpecies && S.soloedSpecies !== c.species) continue;

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
    // Show grab cursor when hovering the freq-axis column
    if (mx >= 0 && mx < YAXIS_W) canvas.style.cursor = 'ns-resize';
    scheduleRender();
    return;
  }

  S.mouseX = mx; S.mouseY = my;

  const found = _inRugZone(my) ? _rugHitTest(mx) : _hitTest(mx, my);
  if (found !== S.hoveredCall) {
    S.hoveredCall = found;
    if (found) showTooltip(found, e.clientX, e.clientY);
    else hideTooltip();
  }

  // Cursor priority: ruler lock-out → ruler resize → playhead handle → playhead line → call → default
  if (!S.isRuling) {
    const rHit = _rulerHitTest(mx, my);
    if (rHit) {
      const _RC = { n:'ns-resize',s:'ns-resize',e:'ew-resize',w:'ew-resize',
                    ne:'nesw-resize',sw:'nesw-resize',nw:'nwse-resize',se:'nwse-resize' };
      canvas.style.cursor = _RC[rHit];
    } else {
      const phX = tToX(S.playheadTime);
      if (my < 12 && Math.abs(mx - phX) <= 8) {
        canvas.style.cursor = 'grab';
      } else if (Math.abs(mx - phX) <= PLAYHEAD_GRAB_PX) {
        canvas.style.cursor = 'ew-resize';
      } else {
        canvas.style.cursor = found ? 'pointer' : 'crosshair';
      }
    }
  }
  scheduleRender();
}

function handleClick(e) {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;
  const found = _inRugZone(my) ? _rugHitTest(mx) : _hitTest(mx, my);
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
let _ovCmdDrag = null;  // {x0, ov0, moved} — Cmd+drag pans the overview window
let _ovX0 = 0, _ovVS0 = 0, _ovVD0 = 0;
const OV_EDGE_PX = 7;  // px grab zone for each edge handle
let _rulerBtnRect   = null;  // bounding box of the ruler "Zoom to selection" button
let _rulerCloseRect = null;  // bounding box of the ruler "✕ clear" button
let _bpfAttPos      = null;  // {x, y, w} canvas-space position for the BPF attenuation overlay
let _rulerDrag      = null;  // 'n'|'s'|'e'|'w'|'ne'|'nw'|'se'|'sw' — ruler-resize handle
let _freqAxisDrag = false; // dragging on the freq-axis column to pan frequency
let _faY0 = 0, _faFH0 = 0, _faFL0 = 0;
const _RULER_HIT = 8;  // px hit-zone radius for ruler resize handles

// When the ruler is fixed, compute its pixel rect from the time/freq loop bounds so
// it stays aligned after panning or zooming.  During active rubber-band drawing
// (isRuling), fall back to the raw canvas-pixel coords.
function _rulerPx() {
  if (S.rulerFixed && S.rulerLoopT0 != null) {
    const rx0 = tToX(S.rulerLoopT0), rx1 = tToX(S.rulerLoopT1);
    const ry0 = fToY(S.rulerLoopF1), ry1 = fToY(S.rulerLoopF0);
    return { x0: Math.min(rx0, rx1), x1: Math.max(rx0, rx1),
             y0: Math.min(ry0, ry1), y1: Math.max(ry0, ry1) };
  }
  return { x0: Math.min(S.rulerX0, S.rulerX1), x1: Math.max(S.rulerX0, S.rulerX1),
           y0: Math.min(S.rulerY0, S.rulerY1), y1: Math.max(S.rulerY0, S.rulerY1) };
}

// Hit-test ruler resize handles.  Returns 'n'/'s'/'e'/'w'/'ne'/'nw'/'se'/'sw'
// when mx,my is within _RULER_HIT of a corner or edge midpoint of the fixed
// ruler, null otherwise.
function _rulerHitTest(mx, my) {
  if (!S.rulerFixed) return null;
  const { x0, x1, y0, y1 } = _rulerPx();
  const H  = _RULER_HIT;
  const onL = Math.abs(mx - x0) <= H, onR = Math.abs(mx - x1) <= H;
  const onT = Math.abs(my - y0) <= H, onB = Math.abs(my - y1) <= H;
  const inX = mx > x0 - H && mx < x1 + H;
  const inY = my > y0 - H && my < y1 + H;
  if (onT && onL) return 'nw';
  if (onT && onR) return 'ne';
  if (onB && onL) return 'sw';
  if (onB && onR) return 'se';
  if (onT && inX) return 'n';
  if (onB && inX) return 's';
  if (onL && inY) return 'w';
  if (onR && inY) return 'e';
  return null;
}

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

  // Cmd on overview: drag slides the overview window in time (when zoomed in);
  // a click without dragging seeks the playhead.  Decided on mouseup.
  if (e.metaKey) {
    _ovCmdDrag = { x0: e.clientX, ov0: S.ovStart, moved: false };
    ovCanvas.style.cursor = 'grabbing';
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

// Double-click on the overview → jump the playhead to that time.
ovCanvas.addEventListener('dblclick', e => {
  e.preventDefault();
  const ox = e.clientX - ovCanvas.getBoundingClientRect().left;
  const t  = Math.max(0, Math.min(S.duration, ovXT(ox)));
  if (typeof audioSeek === 'function') audioSeek(t);
  else { S.playheadTime = t; scheduleRender(); }
});

ovCanvas.addEventListener('mousemove', e => {
  if (_ovDrag || _ovPhDrag || _ovCmdDrag) return;  // cursor already set
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
document.getElementById('btn-contour').onclick = () => {
  S.showContour = !S.showContour;
  document.getElementById('btn-contour').classList.toggle('clf-active', S.showContour);
  scheduleRender();
};
document.getElementById('contour-method').onchange = e => {
  S.contourMethod = e.target.value;
  scheduleRender();
  // Lazily fetch this method's contours if not yet loaded (defined in render.js;
  // merges data into S.calls by position and calls scheduleRender() when done).
  if (typeof ensureContourMethod === 'function') ensureContourMethod(e.target.value);
};
document.getElementById('btn-boxes').onclick = () => {
  S.showBoxes = !S.showBoxes;
  document.getElementById('btn-boxes').classList.toggle('clf-active', S.showBoxes);
  scheduleRender();
};
// GPU/WebGL is always enabled (toggle removed from UI)
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
  clearGLTextures();            // logScale changed → warp canvas content changed
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
  scheduleRender();   // re-renders the spectrogram, or the call plot if that tab is active
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
  S.viewDur   = Math.max(0.1, Math.min(S.duration, S.viewDur * factor));
  S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, mid - S.viewDur / 2));
}

