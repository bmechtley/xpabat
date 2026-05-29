function scheduleRender() {
  if (S.renderPending) return;
  S.renderPending = true;
  requestAnimationFrame(() => { S.renderPending = false; render(); });
  // Sync viewport → URL (defined in ui.js; guard handles load order)
  if (typeof _scheduleURLSync === 'function') _scheduleURLSync();
}

// ─── Classifier toggle ────────────────────────────────────────
// Copies the selected classifier's fields into the live c.species / c.color /
// c.short / c.conf fields so all rendering code works without modification.
function setClassifier(which) {
  S.classifier = which;
  const useV2 = (which === 'v2');
  for (const c of S.calls) {
    if (useV2) {
      c.species = c.species_v2 ?? c.species;
      c.conf    = c.conf_v2    ?? c.conf;
      c.color   = c.color_v2   ?? c.color;
      c.short   = c.short_v2   ?? c.short;
    } else {
      c.species = c.species_v1 ?? c.species;
      c.conf    = c.conf_v1    ?? c.conf;
      c.color   = c.color_v1   ?? c.color;
      c.short   = c.short_v1   ?? c.short;
    }
  }
  // Update toggle button appearance
  document.getElementById('clf-v1').classList.toggle('clf-active', !useV2);
  document.getElementById('clf-v2').classList.toggle('clf-active',  useV2);
  S.hiddenSpecies.clear();   // reset hide-state — species set may have changed
  buildLegend(S.colors);
  scheduleRender();
  // Re-render call detail so "v1 says"/"v2 says" label flips immediately.
  if (S.selectedCall) renderDetail(S.selectedCall);
}

function render() {
  _logWarpBudget = LOG_WARP_PER_FRAME;  // reset per-frame budget
  ensureTiles();
  const W = canvas.width, H = SPEC_H(), specW = W - YAXIS_W;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);

  // ── Spectrogram tiles (frequency-warped) ──
  const viewEnd  = S.viewStart + S.viewDur;
  const first = Math.max(0, Math.floor(S.viewStart / S.tileDur));
  const last  = Math.min(S.nTiles - 1, Math.ceil(viewEnd / S.tileDur));

  // Apply saturation filter to all tile draws; reset before contours/axes so
  // those remain fully saturated regardless of the spectrogram setting.
  if (S.saturation < 1) ctx.filter = `saturate(${S.saturation})`;

  for (let i = first; i <= last; i++) {
    const img = S.tileImgs.get(i);
    const tS  = i * S.tileDur;
    const tE  = Math.min((i + 1) * S.tileDur, S.duration);
    const tileDurActual = tE - tS;

    if (!img || !S.tileReady.get(i)) {
      const x1 = Math.max(YAXIS_W, tToX(tS));
      const x2 = Math.min(W, tToX(tE));
      ctx.fillStyle = '#151515';
      ctx.fillRect(x1, 0, x2 - x1, H);
      ctx.fillStyle = '#2a2a2a';
      ctx.font = '11px monospace';
      ctx.fillText('loading…', x1 + 4, H / 2);
      continue;
    }

    // Source X slice (time axis, always linear in the tile image)
    const srcX0 = Math.max(0, (S.viewStart - tS) / tileDurActual * img.naturalWidth);
    const srcX1 = Math.min(img.naturalWidth, (viewEnd - tS) / tileDurActual * img.naturalWidth);
    if (srcX1 <= srcX0) continue;
    const dstX0 = Math.max(YAXIS_W, tToX(tS));
    const dstX1 = Math.min(W, tToX(tE));
    if (dstX1 <= dstX0) continue;
    const srcW = srcX1 - srcX0, dstW = dstX1 - dstX0;

    // Both linear and log/blend: get the full-range warp canvas for this tile
    // and crop it to the current freq viewport.  Freq scrolling is free — only
    // the Y crop coordinates change, not the warp canvas itself.
    // imageSmoothingQuality 'medium' (bilinear) avoids the ringing artefacts
    // that 'high' (bicubic) produces near sharp spectrogram edges; those ringing
    // bands oscillate as the scale changes during zoom → shimmer.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'medium';
    {
      const warped = _getWarpedTile(i, img, H);
      const wY0 = _fullRangeFToY(S.freqHigh, H, S.logScale);
      const wY1 = _fullRangeFToY(S.freqLow,  H, S.logScale);
      if (warped) {
        if (wY1 > wY0)
          ctx.drawImage(warped, srcX0, wY0, srcW, wY1 - wY0, dstX0, 0, dstW, H);
      } else {
        // Log warp budget exceeded this frame — show linear fallback and keep
        // re-rendering until all visible tiles are fully warped.
        const ty0   = (TILE_FREQ_HIGH - S.freqHigh) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
        const ty1   = (TILE_FREQ_HIGH - S.freqLow)  / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
        const fbY0  = Math.max(0, ty0) * img.naturalHeight;
        const fbY1  = Math.min(1, ty1) * img.naturalHeight;
        if (fbY1 > fbY0)
          ctx.drawImage(img, srcX0, fbY0, srcW, fbY1 - fbY0, dstX0, 0, dstW, H);
        scheduleRender();  // come back next frame to warp remaining tiles
      }
    }

    // ── Flat tile overlay ──────────────────────────────────────
    if (S.flatness > 0) {
      const fImg = S.flatTileImgs.get(i);
      if (fImg && S.flatTileReady.get(i)) {
        const warpedFlat = _getWarpedTile(i, fImg, H, S.flatTileWarpCache);
        const wY0 = _fullRangeFToY(S.freqHigh, H, S.logScale);
        const wY1 = _fullRangeFToY(S.freqLow,  H, S.logScale);
        if (wY1 > wY0 && warpedFlat) {
          ctx.globalAlpha = S.flatness;
          ctx.drawImage(warpedFlat, srcX0, wY0, srcW, wY1 - wY0, dstX0, 0, dstW, H);
          ctx.globalAlpha = 1;
        }
      } else {
        loadFlatTile(i);
      }
    }

    // ── Mask overlay ───────────────────────────────────────────
    if (S.crossfade > 0) {
      const mImg = S.maskTileImgs.get(i);
      if (mImg && S.maskTileReady.get(i)) {
        const warpedMask = _getWarpedTile(i, mImg, H, S.maskTileWarpCache);
        const wY0 = _fullRangeFToY(S.freqHigh, H, S.logScale);
        const wY1 = _fullRangeFToY(S.freqLow,  H, S.logScale);
        if (wY1 > wY0 && warpedMask) {
          ctx.globalAlpha = S.crossfade;
          ctx.drawImage(warpedMask, srcX0, wY0, srcW, wY1 - wY0, dstX0, 0, dstW, H);
          ctx.globalAlpha = 1;
        }
      } else {
        loadMaskTile(i);
      }
    }
  }

  // Reset rendering state before drawing annotations
  ctx.filter = 'none';
  ctx.imageSmoothingQuality = 'low';  // annotations are path-drawn, no image smoothing needed

  // ── Grid lines (log-aware) ──
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth   = 1;
  for (const f of _freqTicks()) {
    if (f <= S.freqLow || f >= S.freqHigh) continue;
    const y = Math.round(fToY(f)) + 0.5;
    ctx.beginPath(); ctx.moveTo(YAXIS_W, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // ── Call overlays (always drawn: boxes appear on hover/select even when S.showBoxes is off) ──
  if (S.showBoxes || S.showContour || S.hoveredCall || S.selectedCall)
    drawCallOverlays(specW, H, viewEnd);

  // ── Freq axis ──
  drawFreqAxis(W, H);

  // ── Call density rug (above time axis) ──
  if (S.calls.length) drawCallRug(W, H, specW);

  // ── Time axis ──
  drawTimeAxis(W, H, specW);

  // ── Ruler (must be before crosshairs so crosshairs render on top) ──
  drawRuler(W, H);

  // ── Playhead ──
  drawPlayhead(W, H);

  // ── Crosshairs ──
  drawCrosshairs(W, H);

  // ── Overview ──
  drawOverview();

  // ── Time display ──
  document.getElementById('time-display').innerHTML =
    `View: ${fmt(S.viewStart)} – ${fmt(S.viewStart + S.viewDur)}<br>Duration: ${S.viewDur.toFixed(1)}s`;
  const _posEl = document.getElementById('playhead-pos');
  if (_posEl && typeof fmtHMS === 'function') _posEl.textContent = fmtHMS(S.playheadTime);

  // ── PSD sidebar ──
  drawPSD();
  schedulePSDFetch();
}

// Zoom-dependent base line width: thin when zoomed out (many calls, small pixels),
// progressively thicker when zoomed in.  Log2 of pixels-per-second keeps the
// growth gradual — doubles roughly every time the zoom doubles.
function _baseLineW() {
  const pxPerSec = (canvas.width - YAXIS_W) / Math.max(S.viewDur, 0.01);
  // ~0.8 px at 30 s view · ~1.5 px at 10 s · ~2.5 px at 3 s · ~3.5 px at 0.5 s
  return Math.max(0.5, Math.min(3.5, 1.0 + Math.log2(pxPerSec / 50) * 0.5));
}

function drawCall(c, specW, H) {
  const sel  = c === S.selectedCall;
  const hov  = c === S.hoveredCall;
  const col  = c.color;
  const base = _baseLineW();

  const x0 = tToX(c.t0),  x1 = tToX(c.t1);
  const y0 = fToY(c.Fmax), y1 = fToY(c.Fmin);
  const bw = x1 - x0,     bh = y1 - y0;

  // Bounding box: always visible when S.showBoxes is checked; otherwise only
  // on hover or selection so the spectrogram stays uncluttered by default.
  if (S.showBoxes || sel || hov) {
    const ca = S.contourAlpha;
    // Fill: scale base alphas by contourAlpha so the opacity slider works on boxes too.
    ctx.globalAlpha = sel ? 0.45 * ca : (hov ? 0.35 * ca : 0.18 * ca);
    ctx.fillStyle   = col;
    ctx.fillRect(x0, y0, bw, bh);
    // Stroke: same alpha logic as contour (boosted on hover/select).
    ctx.globalAlpha = sel ? Math.min(1, ca * 1.8) : (hov ? Math.min(1, ca * 1.4) : ca);
    ctx.strokeStyle = sel ? '#ffffff' : col;
    ctx.lineWidth   = sel ? Math.min(5, base * 2.2) : (hov ? Math.min(4, base * 1.6) : Math.max(0.5, base * 0.8));
    ctx.strokeRect(x0, y0, bw, bh);

    if (bw > 10) {
      // Label uses the same alpha as the stroke so it fades with the opacity slider.
      ctx.font      = 'bold 10px monospace';
      ctx.fillStyle = sel ? '#ffffff' : col;
      const ly      = y0 > 14 ? y0 - 3 : y0 + bh + 11;
      ctx.fillText(c.short, x0 + 2, ly);
    }
    ctx.globalAlpha = 1;
  }

  if (S.showContour && c.contour && c.contour.length > 1) {
    ctx.beginPath();
    // Selected contour turns white to match the box border; hovered and normal
    // keep the species colour, boosted in saturation so it punches through the
    // spectrogram background even at bright regions.
    ctx.strokeStyle = sel ? '#ffffff' : col;
    ctx.lineWidth   = sel ? Math.min(5, base * 2) : (hov ? Math.min(4, base * 1.6) : base);
    // Contour opacity: user-controlled slider; boosted on hover/select
    ctx.globalAlpha = sel ? Math.min(1, S.contourAlpha * 1.8)
                          : (hov ? Math.min(1, S.contourAlpha * 1.4)
                                 : S.contourAlpha);
    ctx.filter = sel ? 'none' : 'saturate(2.5) brightness(1.15)';
    let first = true;
    for (const [ct, cf] of c.contour) {
      const cx = tToX(ct), cy = fToY(cf);
      if (first) { ctx.moveTo(cx, cy); first = false; }
      else        ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.filter = 'none';
    ctx.globalAlpha = 1;

    if (sel || hov) {
      const pmid = c.contour[Math.floor(c.contour.length / 2)];
      ctx.beginPath();
      ctx.arc(tToX(pmid[0]), fToY(pmid[1]), Math.max(2, base * 0.9), 0, Math.PI * 2);
      ctx.fillStyle = sel ? '#ffffff' : col;
      ctx.fill();
    }
  }
}

function drawCrosshairs(W, H) {
  // Don't show crosshairs while actively drawing a ruler
  if (S.mouseX < 0 || S.isRuling) return;
  const mx = S.mouseX, my = S.mouseY;
  const t  = xToT(mx);
  const f  = yToF(my);

  ctx.save();
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = 'rgba(255,255,255,0.45)';
  ctx.lineWidth   = 1;
  // Vertical line (time)
  ctx.beginPath(); ctx.moveTo(mx, 0); ctx.lineTo(mx, H); ctx.stroke();
  // Horizontal line (frequency)
  ctx.beginPath(); ctx.moveTo(YAXIS_W, my); ctx.lineTo(W, my); ctx.stroke();
  ctx.setLineDash([]);

  ctx.font = '10px monospace';
  // Time label — just above the bottom time-axis strip (~20px from bottom)
  const tLabel = fmt(t);
  const tlw = ctx.measureText(tLabel).width;
  let tlx = mx + 4;
  if (tlx + tlw + 6 > W) tlx = mx - tlw - 8;
  ctx.fillStyle = 'rgba(0,0,0,0.78)';
  ctx.fillRect(tlx - 2, H - 32, tlw + 6, 14);
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.fillText(tLabel, tlx, H - 21);

  // Frequency label — just right of the freq-axis column
  const fLabel = f.toFixed(1) + ' kHz';
  const flw = ctx.measureText(fLabel).width;
  let fly = my - 5;
  if (fly < 12) fly = my + 14;
  ctx.fillStyle = 'rgba(0,0,0,0.78)';
  ctx.fillRect(YAXIS_W + 5, fly - 12, flw + 6, 14);
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.fillText(fLabel, YAXIS_W + 7, fly);

  ctx.restore();
}

// ── Playhead ──────────────────────────────────────────────────
// Vertical transport cursor shown in both main canvas and overview.
// Color: #00d488 (bright cyan-green); triangle handle at the top.
const PLAYHEAD_COLOR = '#00d488';

function drawPlayhead(W, H) {
  const x = tToX(S.playheadTime);
  if (x < YAXIS_W - 1 || x > W + 1) return;   // off-screen
  ctx.save();
  ctx.strokeStyle   = PLAYHEAD_COLOR;
  ctx.lineWidth     = 1.5;
  ctx.globalAlpha   = 0.88;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, H);
  ctx.stroke();
  // Triangle handle at top
  const hs = 6;   // half-base px
  ctx.fillStyle   = PLAYHEAD_COLOR;
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.moveTo(x - hs, 0);
  ctx.lineTo(x + hs, 0);
  ctx.lineTo(x, hs * 1.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawRuler(W, H) {
  if (!S.isRuling && !S.rulerFixed) return;
  const moved = Math.hypot(S.rulerX1 - S.rulerX0, S.rulerY1 - S.rulerY0);
  if (moved < 3) return;

  const x0 = Math.min(S.rulerX0, S.rulerX1);
  const x1 = Math.max(S.rulerX0, S.rulerX1);
  const y0 = Math.min(S.rulerY0, S.rulerY1);
  const y1 = Math.max(S.rulerY0, S.rulerY1);
  const rW = x1 - x0, rH = y1 - y0;

  const t0  = xToT(x0), t1 = xToT(x1);
  const fHi = yToF(y0), fLo = yToF(y1);   // y0=top=higher freq
  const dtMs = (t1 - t0) * 1000;
  const df   = fHi - fLo;

  ctx.save();

  // Translucent fill
  ctx.fillStyle   = 'rgba(242,142,43,0.08)';
  ctx.fillRect(x0, y0, rW, rH);

  // Dashed orange border
  ctx.setLineDash([5, 3]);
  ctx.strokeStyle = '#f28e2b';
  ctx.lineWidth   = 1.5;
  ctx.globalAlpha = 0.9;
  ctx.strokeRect(x0, y0, rW, rH);
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  // Corner dots
  ctx.fillStyle = '#f28e2b';
  for (const [cx, cy] of [[x0,y0],[x1,y0],[x0,y1],[x1,y1]]) {
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI*2); ctx.fill();
  }

  // Measurement label
  const dtStr = dtMs >= 1000 ? (dtMs/1000).toFixed(3)+'s' : dtMs.toFixed(1)+'ms';
  const lines = [
    `Δt   ${dtStr}`,
    `Δf   ${df.toFixed(1)} kHz`,
    `t   ${fmt(t0)} → ${fmt(t1)}`,
    `f   ${fLo.toFixed(1)} → ${fHi.toFixed(1)} kHz`,
  ];
  ctx.font = '11px monospace';
  const lw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 14;
  const lh = lines.length * 16 + 10;

  const btnH = 20;
  // Prefer label to the right of the box; fall back left if it would clip
  let lx = x1 + 8, ly = y0;
  if (lx + lw > W - 4)   lx = x0 - lw - 8;
  if (lx < YAXIS_W + 4)  lx = x0 + 4;
  if (ly + lh + btnH + 4 > H - 4)  ly = y1 - lh - btnH - 4;
  if (ly < 2)             ly = 2;

  ctx.fillStyle   = 'rgba(10,10,10,0.88)';
  ctx.fillRect(lx, ly, lw, lh);
  ctx.strokeStyle = '#f28e2b';
  ctx.lineWidth   = 1;
  ctx.strokeRect(lx, ly, lw, lh);
  ctx.fillStyle   = '#f28e2b';
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], lx + 7, ly + 16 + i * 16);
  }

  // "Zoom to selection" button directly below the info box
  if (S.rulerFixed) {
    const btnY = ly + lh + 3;
    _rulerBtnRect = { x: lx, y: btnY, w: lw, h: btnH };
    ctx.fillStyle = 'rgba(242,142,43,0.18)';
    ctx.fillRect(lx, btnY, lw, btnH);
    ctx.strokeStyle = '#f28e2b';
    ctx.strokeRect(lx, btnY, lw, btnH);
    ctx.fillStyle = '#f28e2b';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('⊕ Zoom to selection', lx + lw / 2, btnY + 13);
    ctx.textAlign = 'left';
  } else {
    _rulerBtnRect = null;
  }

  ctx.restore();
}

// Binary search: first index where calls[i].t0 >= target
function callsLowerBound(target) {
  let lo = 0, hi = S.calls.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (S.calls[mid].t0 < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function drawCallOverlays(specW, H, viewEnd) {
  // S.calls is sorted by t0.  Use binary search to skip the bulk of the array.
  // Back up 0.3 s from viewStart to catch calls that started just before the window.
  const startIdx = Math.max(0, callsLowerBound(S.viewStart - 0.3));
  const visible  = [];
  for (let i = startIdx; i < S.calls.length; i++) {
    const c = S.calls[i];
    if (c.t0 >= viewEnd) break;
    if (c.t1 > S.viewStart && !S.hiddenSpecies.has(c.species) && c.conf >= S.minConf) visible.push(c);
  }
  if (!visible.length) return;

  // LOD: view-duration threshold instead of per-call pixel width.
  // Using per-call size caused shimmer on zoom: individual calls crossed the 2px
  // boundary every frame as S.viewDur changed, toggling between drawCall() and
  // drawCallsBatched() for the same call on adjacent frames.
  // A single viewDur cutoff switches the ENTIRE set at once — at most one visual
  // transition per zoom gesture rather than N per-call transitions per frame.
  // Below the threshold (zoomed in): full contour/box detail for all calls.
  // Above the threshold (zoomed out): fast batched ticks for all calls.
  const LOD_DUR_THRESHOLD = 20;   // seconds — full detail when viewDur ≤ this
  const sel = S.selectedCall;
  const hov = S.hoveredCall;

  if (S.viewDur > LOD_DUR_THRESHOLD) {
    drawCallsBatched(visible, specW, H);
    // Selected/hovered always get full detail even when zoomed out
    if (sel && sel.t0 < viewEnd && sel.t1 > S.viewStart) drawCall(sel, specW, H);
    if (hov && hov !== sel && hov.t0 < viewEnd && hov.t1 > S.viewStart) drawCall(hov, specW, H);
    return;
  }

  const SPARSE_THRESHOLD = 400;
  if (visible.length <= SPARSE_THRESHOLD) {
    for (const c of visible) drawCall(c, specW, H);
    return;
  }

  // Dense view (zoomed in, many calls): batched rects + repaint sel/hov on top
  drawCallsBatched(visible, specW, H);
  if (sel && sel.t0 < viewEnd && sel.t1 > S.viewStart) drawCall(sel, specW, H);
  if (hov && hov !== sel && hov.t0 < viewEnd && hov.t1 > S.viewStart) drawCall(hov, specW, H);
}

function drawCallsBatched(visible, specW, H) {
  // Zoomed-out view: draw each call as a vertical tick at its centre time,
  // spanning Fmin→Fmax.  Tick width uses the same zoom-scaled _baseLineW() as
  // the individual contour renderer so there is no visible jump at the
  // sparse/dense threshold.
  //
  // We aggregate calls that share the same pixel column into one rect whose
  // y-extent is the union of all their freq ranges.  This achieves two things:
  //   1. No shimmering — the merged rect is the same every frame regardless of
  //      which call happens to be processed first as the view scrolls.
  //   2. Fewer path ops — at extreme zoom-out, thousands of calls compress into
  //      at most ~specW unique columns, so we go from O(n_calls) to O(n_pixels).
  // Use fractional (non-rounded) positions so ticks shift smoothly as the view
  // pans — Math.round causes 1-px integer snapping which creates shimmering as
  // adjacent calls alternately share / split a pixel column each frame.
  // A minimum tickW of 2 ensures sub-pixel ticks remain visible via antialiasing.
  const tickW = Math.max(2, _baseLineW());
  const half  = tickW / 2;

  const bySpecies = {};
  for (const c of visible) {
    if (!bySpecies[c.species]) bySpecies[c.species] = { col: c.color, calls: [] };
    bySpecies[c.species].calls.push(c);
  }

  for (const { col, calls } of Object.values(bySpecies)) {
    ctx.fillStyle   = col;
    ctx.globalAlpha = S.contourAlpha;   // respect the opacity slider in zoomed-out view
    ctx.beginPath();
    for (const c of calls) {
      const xc = tToX((c.t0 + c.t1) / 2);   // fractional — no Math.round
      const y0 = fToY(c.Fmax);
      const y1 = fToY(c.Fmin);
      ctx.rect(xc - half, y0, tickW, Math.max(tickW, y1 - y0));
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// Returns evenly-spaced frequency tick values (kHz) for the current view.
// Picks the finest interval from [1,2,5,10,20] that keeps ≤ 10 ticks visible.
function _freqTicks() {
  const range = S.freqHigh - S.freqLow;
  let interval = 20;
  for (const s of [1, 2, 5, 10, 20]) {
    interval = s;
    if (range / s <= 10) break;
  }
  const ticks = [];
  const first = Math.ceil(S.freqLow / interval) * interval;
  for (let f = first; f <= S.freqHigh + 0.001; f += interval) ticks.push(f);
  return ticks;
}

function drawFreqAxis(W, H) {
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, YAXIS_W, H);
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(YAXIS_W, 0); ctx.lineTo(YAXIS_W, H); ctx.stroke();

  ctx.fillStyle = '#777';
  ctx.font      = '10px monospace';
  ctx.textAlign = 'right';
  for (const f of _freqTicks()) {
    const y = Math.round(fToY(f));
    if (y < 0 || y > H) continue;
    ctx.fillStyle = '#666';
    ctx.fillText(`${f}k`, YAXIS_W - 5, y + 3);
    ctx.strokeStyle = '#2a2a2a';
    ctx.beginPath(); ctx.moveTo(YAXIS_W - 3, y + 0.5); ctx.lineTo(YAXIS_W, y + 0.5); ctx.stroke();
  }
  // Rotated label
  ctx.save();
  ctx.translate(10, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#444';
  ctx.font      = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Frequency (Hz)', 0, 0);
  ctx.restore();
  ctx.textAlign = 'left';

}

// Compact call-density rug drawn just above the time axis.
// Every visible call → 1-px vertical tick in its species colour.
// Gives an immediate sense of density and species composition when zoomed out.
const RUG_H = 11;
function drawCallRug(W, H, specW) {
  const rugTop = H - 14 - RUG_H - 2;   // 14px = time-axis height, 2px gap
  ctx.fillStyle = 'rgba(8,8,8,0.82)';
  ctx.fillRect(YAXIS_W, rugTop, specW, RUG_H);

  const viewEnd  = S.viewStart + S.viewDur;
  const startIdx = callsLowerBound(S.viewStart - 0.3);

  // Group by species for batched drawing.
  // Use a Set per species to deduplicate pixel columns — at high zoom-out,
  // many calls compress to the same 1-px column; drawing duplicates is waste.
  const bySpecies = {};
  for (let i = startIdx; i < S.calls.length; i++) {
    const c = S.calls[i];
    if (c.t0 > viewEnd) break;
    if (S.hiddenSpecies.has(c.species) || c.conf < S.minConf) continue;
    if (!bySpecies[c.species]) bySpecies[c.species] = { col: c.color, xs: new Set() };
    bySpecies[c.species].xs.add(Math.round(tToX((c.t0 + c.t1) / 2)));
  }

  for (const { col, xs } of Object.values(bySpecies)) {
    ctx.fillStyle   = col;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (const x of xs) ctx.rect(x, rugTop + 1, 1, RUG_H - 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Hairline border
  ctx.strokeStyle = '#1e1e1e';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(YAXIS_W, rugTop + 0.5);
  ctx.lineTo(YAXIS_W + specW, rugTop + 0.5);
  ctx.stroke();
}

function drawTimeAxis(W, H, specW) {
  const viewEnd = S.viewStart + S.viewDur;
  // Choose a sensible tick interval
  const targets = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60];
  const minPx   = 60;
  let interval  = targets.find(v => v / S.viewDur * specW >= minPx) || 60;
  const t0 = Math.ceil(S.viewStart / interval) * interval;
  ctx.fillStyle   = '#555';
  ctx.font        = '10px monospace';
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth   = 1;
  for (let t = t0; t <= viewEnd; t += interval) {
    const x = Math.round(tToX(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, H - 14); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillText(fmt(t), x + 2, H - 3);
  }
}

// Overview coordinate helpers (use S.ovStart/ovDur, not full S.duration)
function ovTX(t)  { return (t - S.ovStart) / S.ovDur * ovCanvas.width; }
function ovXT(ox) { return S.ovStart + ox / ovCanvas.width * S.ovDur; }

function drawOverview() {
  const OW = ovCanvas.width, OH = OV_H;
  const ovD = S.ovDur || S.duration || 1;   // guard against 0 before init
  octx.clearRect(0, 0, OW, OH);
  octx.fillStyle = '#0d0d0d';
  octx.fillRect(0, 0, OW, OH);

  // Individual call dots — y-position encodes peak frequency
  // Calls outside the current freq view are skipped (not plotted at midpoint)
  for (const c of S.calls) {
    if (c.Fpeak < S.freqLow || c.Fpeak > S.freqHigh) continue;
    const x = ovTX(c.t0);
    const w = Math.max(1, (c.t1 - c.t0) / ovD * OW);
    if (x + w < 0 || x > OW) continue;
    const fy = OH * (1 - (c.Fpeak - S.freqLow) / (S.freqHigh - S.freqLow));
    octx.fillStyle   = c.color;
    octx.globalAlpha = 0.7;
    octx.fillRect(x, Math.max(0, fy - 2), w, 4);
  }
  octx.globalAlpha = 1;

  // Viewport box
  const vx0 = ovTX(S.viewStart);
  const vx1 = ovTX(S.viewStart + S.viewDur);
  const vw  = Math.max(2, vx1 - vx0);
  octx.fillStyle = 'rgba(255,255,255,0.07)';
  octx.fillRect(vx0, 0, vw, OH);
  octx.strokeStyle = 'rgba(255,255,255,0.28)';
  octx.lineWidth = 1;
  octx.strokeRect(vx0, 0, vw, OH);

  // Draggable edge handles — brighter vertical bars
  const hw = 4;
  octx.fillStyle = _ovDrag ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.45)';
  octx.fillRect(vx0,      0, hw, OH);
  octx.fillRect(vx1 - hw, 0, hw, OH);

  // When zoomed: draw a thin full-recording ruler at the bottom (3 px strip)
  const zoomed = ovD < S.duration * 0.99;
  if (zoomed) {
    const rH = 3, rY = OH - rH;
    octx.fillStyle = '#1e1e1e';
    octx.fillRect(0, rY, OW, rH);
    // overview window within full recording
    const rx0 = S.ovStart / S.duration * OW;
    const rx1 = (S.ovStart + ovD) / S.duration * OW;
    octx.fillStyle = 'rgba(255,255,255,0.18)';
    octx.fillRect(rx0, rY, Math.max(2, rx1 - rx0), rH);
    // viewport within full recording
    const rvx0 = S.viewStart / S.duration * OW;
    const rvx1 = (S.viewStart + S.viewDur) / S.duration * OW;
    octx.fillStyle = 'rgba(255,255,255,0.5)';
    octx.fillRect(rvx0, rY, Math.max(1, rvx1 - rvx0), rH);
  }

  // Playhead line + draggable triangle handle at top
  const phOX = ovTX(S.playheadTime);
  if (phOX >= -8 && phOX <= OW + 8) {
    octx.save();
    octx.strokeStyle = PLAYHEAD_COLOR;
    octx.lineWidth   = 1.5;
    octx.globalAlpha = 0.85;
    octx.beginPath(); octx.moveTo(phOX, 0); octx.lineTo(phOX, OH); octx.stroke();
    // Triangle handle at the top (pointing downward) — drag target
    const hs = 7;
    octx.fillStyle   = PLAYHEAD_COLOR;
    octx.globalAlpha = _ovPhDrag ? 1 : 0.95;
    octx.beginPath();
    octx.moveTo(phOX - hs, 0);
    octx.lineTo(phOX + hs, 0);
    octx.lineTo(phOX, hs * 1.5);
    octx.closePath();
    octx.fill();
    octx.restore();
  }

  // Border
  octx.strokeStyle = zoomed ? '#3a3a3a' : '#222';
  octx.lineWidth   = 1;
  octx.strokeRect(0, 0, OW, OH);

  // ── Viewport time labels — mirror PSD freq-label style ──────
  if (S.recordingStart || true) {  // always show (use fmt fallback if no timestamp)
    const t0 = S.viewStart, t1 = S.viewStart + S.viewDur;
    const crossDate = _spansMidnight(t0, t1);
    const lbl0 = crossDate ? fmtAbsFull(t0) : fmtAbs(t0);
    const lbl1 = crossDate ? fmtAbsFull(t1) : fmtAbs(t1);
    octx.font      = 'bold 9px system-ui,sans-serif';
    octx.fillStyle = 'rgba(255,255,255,0.75)';
    // Start label: right-aligned just to the left of the left handle
    octx.textBaseline = 'top';
    octx.textAlign    = 'right';
    octx.fillText(lbl0, vx0 - hw - 1, 3);
    // End label: left-aligned just to the right of the right handle
    octx.textAlign    = 'left';
    octx.fillText(lbl1, vx1 + hw + 1, 3);
  }
}

// ─── PSD transport ───────────────────────────────────────────
// PSD viewport maps psdViewLow..psdViewHigh onto the full canvas height.
// Blends linear and log just like the main canvas fToY/yToF so that toggling
// the Log slider affects both displays consistently.
function psdFToY(f) {
  const lo = psdViewLow, hi = psdViewHigh ?? S.nyquist, a = S.logScale;
  const fc = Math.max(lo + 0.001, Math.min(hi, f));
  const linFrac = (fc - lo) / (hi - lo);
  const logFrac = lo > 0 ? Math.log(fc / lo) / Math.log(hi / lo) : linFrac;
  return psdCanvas.height * (1 - ((1 - a) * linFrac + a * logFrac));
}
function psdYToF(y) {
  const lo = psdViewLow, hi = psdViewHigh ?? S.nyquist, a = S.logScale;
  const frac = 1 - y / psdCanvas.height;
  if (a === 0 || lo <= 0) return Math.max(lo, Math.min(S.nyquist, lo + frac * (hi - lo)));
  if (a === 1) return Math.max(lo, Math.min(S.nyquist, lo * Math.exp(frac * Math.log(hi / lo))));
  let fLo = lo, fHi = hi;
  for (let i = 0; i < 40; i++) {
    const mid = (fLo + fHi) / 2;
    const linF = (mid - lo) / (hi - lo);
    const logF = Math.log(mid / lo) / Math.log(hi / lo);
    ((1 - a) * linF + a * logF < frac) ? fLo = mid : fHi = mid;
  }
  return Math.max(lo, Math.min(S.nyquist, (fLo + fHi) / 2));
}

function _psdHitTest(y) {
  const yHi = psdFToY(S.freqHigh);
  const yLo = psdFToY(S.freqLow);
  if (Math.abs(y - yHi) <= PSD_EDGE_PX) return 'top';
  if (Math.abs(y - yLo) <= PSD_EDGE_PX) return 'bot';
  if (y > yHi && y < yLo)               return 'pan';
  return 'jump';
}

function drawPSD() {
  const W = psdCanvas.width, H = psdCanvas.height;
  psdCtx.clearRect(0, 0, W, H);
  psdCtx.fillStyle = '#0d0d0d';
  psdCtx.fillRect(0, 0, W, H);

  // ── PSD curve — normalised to visible freq range ───────────
  // (bars always fill the full width at the visible peak)
  let peakDb = null, minDb = null;
  if (_psdData && _psdData.freqs.length) {
    const { freqs, powers, vmin, vmax } = _psdData;
    const CURVE_W = W - 4;

    // Find peak and floor in the PSD viewport range for normalisation + labels
    const _pvl = psdViewLow, _pvh = psdViewHigh ?? S.nyquist;
    let peakPow = 0, peakFreq = null, minPow = Infinity, minFreq = null;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= _pvl && freqs[i] <= _pvh) {
        if (powers[i] > peakPow) { peakPow = powers[i]; peakFreq = freqs[i]; }
        if (powers[i] < minPow)  { minPow  = powers[i]; minFreq  = freqs[i]; }
      }
    }
    const scale = peakPow > 0.01 ? 1 / peakPow : 1;
    if (vmin != null && vmax != null && peakPow > 0.01) {
      peakDb = { db: peakPow * (vmax - vmin) + vmin, freq: peakFreq };
      if (minFreq !== null)
        minDb = { db: minPow * (vmax - vmin) + vmin, freq: minFreq,
                  xTip: Math.min(minPow * scale, 1) * (W - 4) + 2 };
    }

    // Clip curve drawing to canvas bounds so out-of-viewport bins don't
    // leave diagonal artefacts at the top/bottom edges.
    psdCtx.save();
    psdCtx.beginPath();
    psdCtx.rect(0, 0, W, H);
    psdCtx.clip();

    // Filled area (high→low freq = top→bottom)
    psdCtx.beginPath();
    let started = false;
    for (let i = freqs.length - 1; i >= 0; i--) {
      const y = psdFToY(freqs[i]);
      const x = Math.min(powers[i] * scale, 1) * CURVE_W + 2;
      if (!started) { psdCtx.moveTo(2, y); started = true; }
      psdCtx.lineTo(x, y);
    }
    if (started) {
      psdCtx.lineTo(2, psdFToY(freqs[0]));
      psdCtx.closePath();
      const g = psdCtx.createLinearGradient(0, 0, W, 0);
      g.addColorStop(0, 'rgba(40,120,70,0.15)');
      g.addColorStop(1, 'rgba(80,200,110,0.45)');
      psdCtx.fillStyle = g;
      psdCtx.fill();
      // Stroke the curve
      psdCtx.beginPath(); started = false;
      for (let i = freqs.length - 1; i >= 0; i--) {
        const y = psdFToY(freqs[i]);
        const x = Math.min(powers[i] * scale, 1) * CURVE_W + 2;
        if (!started) { psdCtx.moveTo(x, y); started = true; }
        else psdCtx.lineTo(x, y);
      }
      psdCtx.strokeStyle = 'rgba(80,200,115,0.75)';
      psdCtx.lineWidth = 1.5;
      psdCtx.stroke();
    }
    psdCtx.restore();
  }

  // ── Transport overlay — matches time-transport (overview) style exactly ──
  const yHi = psdFToY(S.freqHigh);
  const yLo = psdFToY(S.freqLow);
  const hw   = 4;   // handle thickness, same as time transport

  // Window fill  (overview uses rgba(255,255,255,0.07))
  psdCtx.fillStyle = 'rgba(255,255,255,0.07)';
  psdCtx.fillRect(0, yHi, W, yLo - yHi);

  // Window border (overview uses rgba(255,255,255,0.28), lineWidth 1)
  psdCtx.strokeStyle = 'rgba(255,255,255,0.28)';
  psdCtx.lineWidth   = 1;
  psdCtx.strokeRect(0.5, yHi + 0.5, W - 1, yLo - yHi - 1);

  // Edge handles — brighter when dragging (overview: 0.45 inactive, 0.75 active)
  const ha = _psdDrag ? 0.75 : 0.45;
  psdCtx.fillStyle = `rgba(255,255,255,${ha})`;
  psdCtx.fillRect(0, yHi,      W, hw);   // top handle (into window from top)
  psdCtx.fillRect(0, yLo - hw, W, hw);   // bottom handle (into window from bottom)

  // Border (overview: '#222' when not zoomed)
  psdCtx.strokeStyle = '#222';
  psdCtx.lineWidth   = 1;
  psdCtx.strokeRect(0, 0, W, H);

  // ── Frequency labels — same style as overview time-transport labels ──────
  // bold 9px system-ui, rgba(255,255,255,0.75) — no hover dependency, no opacity slider.
  {
    psdCtx.font         = 'bold 9px system-ui,sans-serif';
    psdCtx.fillStyle    = 'rgba(255,255,255,0.75)';
    psdCtx.textBaseline = 'bottom';
    psdCtx.textAlign    = 'right';
    psdCtx.fillText(`${S.freqHigh.toFixed(1)} kHz`, W - 3, yHi - 1);
    psdCtx.textBaseline = 'top';
    psdCtx.fillText(`${S.freqLow.toFixed(1)} kHz`,  W - 3, yLo + 1);
  }

  // ── Peak / floor dB labels — same style ───────────────────────────────
  psdCtx.font         = 'bold 9px system-ui,sans-serif';
  psdCtx.textBaseline = 'middle';
  psdCtx.fillStyle    = 'rgba(255,255,255,0.75)';
  if (peakDb !== null) {
    psdCtx.textAlign = 'right';
    psdCtx.fillText(`${peakDb.db.toFixed(0)} dB`, W - 3, psdFToY(peakDb.freq));
  }
  if (minDb !== null) {
    psdCtx.textAlign = 'left';
    psdCtx.fillText(`${minDb.db.toFixed(0)} dB`, minDb.xTip + 3, psdFToY(minDb.freq));
  }
}

// ── PSD canvas events ─────────────────────────────────────────
psdCanvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  e.preventDefault();
  const y   = e.clientY - psdCanvas.getBoundingClientRect().top;
  const hit = _psdHitTest(y);
  _psdY0  = e.clientY;
  _psdFH0 = S.freqHigh;
  _psdFL0 = S.freqLow;
  if (hit === 'jump') {
    // Click outside the window → centre window on clicked freq
    const f    = psdYToF(y);
    const span = S.freqHigh - S.freqLow;
    S.freqHigh = Math.min(S.nyquist, Math.max(span, f + span / 2));
    S.freqLow  = Math.max(TILE_FREQ_LOW, S.freqHigh - span);
    updateScrollbar(); scheduleRender();
    // Continue as a pan drag from the new position
    _psdFH0 = S.freqHigh; _psdFL0 = S.freqLow;
    _psdDrag = 'pan';
  } else {
    _psdDrag = hit;
  }
  psdCanvas.style.cursor = (_psdDrag === 'pan') ? 'grabbing' : 'ns-resize';
});

psdCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  let delta = e.deltaY;
  if (e.deltaMode === 1) delta *= 20;
  if (e.deltaMode === 2) delta *= 400;
  delta = Math.sign(delta) * Math.min(Math.abs(delta), 200);
  const factor   = Math.pow(1.0025, delta);
  const y        = e.clientY - psdCanvas.getBoundingClientRect().top;
  const fCursor  = psdYToF(y);
  const vh = psdViewHigh ?? S.nyquist;
  const span     = vh - psdViewLow;
  const newSpan  = Math.max(2, Math.min(S.nyquist, span * factor));
  // Keep the frequency under the cursor fixed within the PSD viewport
  const relPos   = (vh - fCursor) / span;
  psdViewHigh = Math.min(S.nyquist, Math.max(newSpan, fCursor + relPos * newSpan));
  psdViewLow  = Math.max(TILE_FREQ_LOW, psdViewHigh - newSpan);
  drawPSD();   // PSD-only redraw — canvas view unchanged
}, { passive: false });

psdCanvas.addEventListener('mousemove', e => {
  const y   = e.clientY - psdCanvas.getBoundingClientRect().top;
  _psdHoverY = y;
  const hit  = _psdDrag || _psdHitTest(y);
  psdCanvas.style.cursor = (hit === 'pan') ? 'grab' :
                            (hit === 'top' || hit === 'bot') ? 'ns-resize' : 'ns-resize';
  drawPSD();
});

psdCanvas.addEventListener('mouseleave', () => {
  if (_psdDrag) return;   // keep labels visible while dragging
  _psdHoverY = null;
  psdCanvas.style.cursor = 'ns-resize';
  drawPSD();
});

// ── Local (ring-buffer) PSD — zero-latency during playback ───────────────────
// Welch parameters match config.py so the display is consistent with server mode.
const _L_NPERSEG = 1024;
const _L_STEP    = 256;                      // nperseg − noverlap  (1024 − 768)
const _L_NFREQS  = _L_NPERSEG / 2 + 1;      // 513 bins, DC … Nyquist

// Pre-compute Hann window and its squared power-sum (computed once at load).
const _lHann = (() => {
  const w = new Float32Array(_L_NPERSEG);
  for (let i = 0; i < _L_NPERSEG; i++)
    w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (_L_NPERSEG - 1)));
  return w;
})();
const _lWinPow = _lHann.reduce((s, w) => s + w * w, 0);

// Persistent FFT scratch buffers — avoids GC pressure at 20 fps.
const _lRe = new Float32Array(_L_NPERSEG);
const _lIm = new Float32Array(_L_NPERSEG);

// In-place radix-2 DIT FFT operating on the module-level _lRe / _lIm arrays.
function _lfft() {
  const n = _L_NPERSEG;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = _lRe[i]; _lRe[i] = _lRe[j]; _lRe[j] = t;
          t = _lIm[i]; _lIm[i] = _lIm[j]; _lIm[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = 2 * Math.PI / len;
    const wc = Math.cos(ang), ws = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let ur = 1, ui = 0;
      for (let j = 0; j < (len >> 1); j++) {
        const a = i + j, b = a + (len >> 1);
        const tr = ur * _lRe[b] - ui * _lIm[b];
        const ti = ur * _lIm[b] + ui * _lRe[b];
        _lRe[b] = _lRe[a] - tr; _lIm[b] = _lIm[a] - ti;
        _lRe[a] += tr;           _lIm[a] += ti;
        const nr = ur * wc - ui * ws; ui = ur * ws + ui * wc; ur = nr;
      }
    }
  }
}

// Rate-limit local PSD to ~20 fps (50 ms).  Each call costs ~2–5 ms of main-thread
// time (147 × 1024-pt FFT), so 20 fps ≈ 4–10 % CPU — well within budget.
let _localPsdAt = 0;
const _LOCAL_PSD_MS = 50;

// Try to compute PSD directly from the ring buffer.
// Returns true  → PSD was drawn (or is still fresh); caller should skip server fetch.
// Returns false → ring buffer unavailable; caller should fall back to server.
function _tryLocalPSD() {
  if (!S.isPlaying) return false;
  if (typeof audioGetFrames !== 'function') return false;

  const now = Date.now();
  if (now - _localPsdAt < _LOCAL_PSD_MS) return true;   // still fresh, suppress server fetch
  _localPsdAt = now;

  const srcSr  = typeof audioSrcSr === 'function' ? audioSrcSr() : S.nyquist * 2000;
  const halfN  = Math.round(0.1 * srcSr);   // ±100 ms of source audio
  const startF = Math.max(0, Math.round(S.playheadTime * srcSr) - halfN);
  const samples = audioGetFrames(startF, halfN * 2);
  if (!samples) return false;   // data not in ring yet / already evicted

  // Welch accumulation
  const accum = new Float64Array(_L_NFREQS);
  let nFrames = 0;
  for (let s = 0; s + _L_NPERSEG <= samples.length; s += _L_STEP) {
    _lRe.fill(0); _lIm.fill(0);
    for (let i = 0; i < _L_NPERSEG; i++) _lRe[i] = samples[s + i] * _lHann[i];
    _lfft();
    for (let i = 0; i < _L_NFREQS; i++)
      accum[i] += _lRe[i] * _lRe[i] + _lIm[i] * _lIm[i];
    nFrames++;
  }
  if (!nFrames) return false;

  // Convert to one-sided dB PSD, then build output arrays for the DISPLAY RANGE only
  // (TILE_FREQ_LOW..TILE_FREQ_HIGH kHz).  This matches the server's format exactly:
  //   • Sub-display bins (0–13 kHz) carry strong 1/f noise; including them causes
  //     drawPSD to mis-scale or draw artifacts at the edges of the canvas.
  //   • Above-display bins (> TILE_FREQ_HIGH) are excluded for the same reason.
  // vmin/vmax are anchored to the display range so the [0,1] normalisation covers
  // only the bat-call window — identical to how the server normalises per-window.
  const sc   = 1 / (srcSr * _lWinPow * nFrames);
  const dbs  = new Float32Array(_L_NFREQS);
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < _L_NFREQS; i++) {
    let p = accum[i] * sc;
    if (i > 0 && i < _L_NFREQS - 1) p *= 2;   // one-sided; double all bins except DC + Nyquist
    dbs[i] = 10 * Math.log10(Math.max(p, 1e-20));
    const fkHz = i * srcSr / _L_NPERSEG / 1000;
    if (fkHz >= TILE_FREQ_LOW && fkHz <= TILE_FREQ_HIGH) {
      if (dbs[i] < vmin) vmin = dbs[i];
      if (dbs[i] > vmax) vmax = dbs[i];
    }
  }
  const range  = Math.max(vmax - vmin, 1);
  // Build output arrays with display-range bins only (matches server output format).
  const freqs  = [];
  const powers = [];
  for (let i = 0; i < _L_NFREQS; i++) {
    const fkHz = i * srcSr / _L_NPERSEG / 1000;
    if (fkHz < TILE_FREQ_LOW || fkHz > TILE_FREQ_HIGH) continue;
    freqs.push(fkHz);
    powers.push((dbs[i] - vmin) / range);
  }
  if (!freqs.length) return false;

  _psdData = { freqs, powers, vmin, vmax };
  drawPSD();
  return true;
}

function _psdWindow() {
  if (S.psdMode === 'playhead') {
    return {
      t0: Math.max(0, S.playheadTime - 0.1),
      t1: Math.min(S.duration || 1e9, S.playheadTime + 0.1),
    };
  }
  return { t0: S.viewStart, t1: S.viewStart + S.viewDur };
}

// Real-time gate — triggers at most this often regardless of playback rate.
// 100 ms ≈ 10 PSD updates/s: snappy during playback without hammering the server.
const PSD_MIN_INTERVAL_MS = 100;
let _psdLastFetchAt = 0;

function schedulePSDFetch() {
  // Playhead mode + playing: compute from ring buffer — no network round-trip.
  if (S.psdMode === 'playhead' && _tryLocalPSD()) return;

  // Otherwise: rate-limited server fetch.
  const { t0, t1 } = _psdWindow();
  if (Math.abs(t0 - _psdT0) < 0.0005 && Math.abs(t1 - _psdT1) < 0.0005) return;
  if (_psdTimer) clearTimeout(_psdTimer);
  if (_psdPending) return;   // already in-flight; finally-block re-checks on completion
  const wait = PSD_MIN_INTERVAL_MS - (Date.now() - _psdLastFetchAt);
  if (wait <= 0) fetchPSD();
  else _psdTimer = setTimeout(fetchPSD, wait);
}

async function fetchPSD() {
  if (_psdPending) return;
  const { t0, t1 } = _psdWindow();
  _psdLastFetchAt = Date.now();
  _psdPending = true;
  try {
    const res = await fetch(`/api/psd?t0=${t0.toFixed(3)}&t1=${t1.toFixed(3)}&f=${S.fid}`);
    _psdData = await res.json();
    _psdT0 = t0; _psdT1 = t1;
    drawPSD();
  } catch (err) {
    console.warn('PSD fetch failed', err);
  } finally {
    _psdPending = false;
    // If the window moved while we were waiting for the server, kick off another
    // fetch immediately (respecting the rate-limit gate in schedulePSDFetch).
    const { t0: nt0, t1: nt1 } = _psdWindow();
    if (Math.abs(nt0 - t0) > 0.0005 || Math.abs(nt1 - t1) > 0.0005)
      schedulePSDFetch();
  }
}

