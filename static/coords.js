// ─── Coordinate helpers ──────────────────────────────────────
function tToX(t) {
  return YAXIS_W + (t - S.viewStart) / S.viewDur * (canvas.width - YAXIS_W);
}
function xToT(x) {
  return S.viewStart + (x - YAXIS_W) / (canvas.width - YAXIS_W) * S.viewDur;
}

// Frequency → canvas Y.
// Blends: frac = (1-α)*linFrac + α*logFrac, then y = H*(1-frac)
// This is a direct closed-form computation.
function fToY(f) {
  const lo = S.freqLow, hi = S.freqHigh, a = S.logScale;
  const fc = Math.max(lo + 0.001, Math.min(hi, f));
  const linFrac = (fc - lo) / (hi - lo);
  const logFrac = Math.log(fc / lo) / Math.log(hi / lo);
  return SPEC_H() * (1 - ((1 - a) * linFrac + a * logFrac));
}

// Canvas Y → frequency. Inverts fToY via binary search (only ~40 iters, negligible).
function yToF(y) {
  const lo = S.freqLow, hi = S.freqHigh, a = S.logScale;
  const frac = 1 - y / SPEC_H();
  if (a === 0) return lo + frac * (hi - lo);
  if (a === 1) return lo * Math.exp(frac * Math.log(hi / lo));
  let fLo = lo, fHi = hi;
  for (let i = 0; i < 40; i++) {
    const mid = (fLo + fHi) / 2;
    const linF = (mid - lo) / (hi - lo);
    const logF = Math.log(mid / lo) / Math.log(hi / lo);
    ((1 - a) * linF + a * logF < frac) ? fLo = mid : fHi = mid;
  }
  return (fLo + fHi) / 2;
}


// ─── Tile warp cache ──────────────────────────────────────────
// Pre-warp each tile image into a detached HTMLCanvasElement at the current
// canvas height so each render only needs ONE drawImage per tile instead of
// ~200 band slices.  We use a plain <canvas> (not OffscreenCanvas) because
// Firefox does not GPU-accelerate OffscreenCanvas on the main thread, making
// drawImage from it as slow as software rendering.  Detached HTMLCanvasElements
// are hardware-accelerated in Chrome, Safari, and Firefox alike.
// Inverse of the full-range blended log/lin mapping:
// maps canvas-Y (0=TILE_FREQ_HIGH, H=TILE_FREQ_LOW) → frequency (kHz).
// Used by _getWarpedTile to build a freq-independent warp canvas.
function _fullRangeYToF(y, H, a) {
  const lo = TILE_FREQ_LOW, hi = TILE_FREQ_HIGH;
  const frac = 1 - y / H;
  if (a <= 0) return lo + frac * (hi - lo);
  if (a >= 1) return lo * Math.exp(frac * Math.log(hi / lo));
  let fLo = lo, fHi = hi;
  for (let i = 0; i < 40; i++) {
    const mid  = (fLo + fHi) / 2;
    const linF = (mid - lo) / (hi - lo);
    const logF = Math.log(mid / lo) / Math.log(hi / lo);
    ((1 - a) * linF + a * logF < frac) ? fLo = mid : fHi = mid;
  }
  return (fLo + fHi) / 2;
}

// Forward mapping: frequency → Y on the full-range warp canvas.
// Used in the render loop to crop the warp to the current freq viewport.
function _fullRangeFToY(f, H, a) {
  const lo = TILE_FREQ_LOW, hi = TILE_FREQ_HIGH;
  const fc = Math.max(lo + 0.001, Math.min(hi, f));
  const linFrac = (fc - lo) / (hi - lo);
  const logFrac = Math.log(fc / lo) / Math.log(hi / lo);
  return H * (1 - ((1 - a) * linFrac + a * logFrac));
}

// Pan the frequency view by `frac` of the visible span, blended between
// linear (S.logScale=0) and multiplicative/log (S.logScale=1).
// frac > 0  →  move toward higher frequencies.
// Returns { fH, fL } clamped to [TILE_FREQ_LOW, S.nyquist], preserving the
// visible span (linear) or ratio fL/fH (log) at the boundary.
function _freqPan(fH0, fL0, frac) {
  const a    = S.logScale;
  const span = fH0 - fL0;

  // Linear component: same absolute kHz shift for both edges.
  const fHlin = fH0 + frac * span;
  const fLlin = fL0 + frac * span;

  // Log component: same multiplicative factor for both edges, so Δlog is equal.
  let fHlog = fH0, fLlog = fL0;
  if (a >= 0.001 && fL0 > 0) {
    const r = Math.pow(fH0 / fL0, frac);   // ratio = (span_ratio)^frac
    fHlog = fH0 * r;
    fLlog = fL0 * r;
  }

  let fH = (1 - a) * fHlin + a * fHlog;
  let fL = (1 - a) * fLlin + a * fLlog;

  // Clamp to [TILE_FREQ_LOW, nyquist]: at each wall, re-derive the other edge
  // using the blended invariant (linear → preserve span; log → preserve ratio).
  const ratio = (fL0 > 0) ? fL0 / fH0 : 0;
  if (fH > S.nyquist) {
    fH = S.nyquist;
    fL = (1 - a) * (fH - span) + a * (fH * ratio);
    fL = Math.max(TILE_FREQ_LOW, fL);
  }
  if (fL < TILE_FREQ_LOW) {
    fL = TILE_FREQ_LOW;
    const fHlog2 = (ratio > 0) ? fL / ratio : fL + span;
    fH = (1 - a) * (fL + span) + a * fHlog2;
    fH = Math.min(S.nyquist, fH);
  }

  return { fH, fL };
}

function _getWarpedTile(idx, img, H, warpCache = S.tileWarpCache) {
  // Cache key does NOT include freqLow/freqHigh: the warp covers the full
  // TILE_FREQ_LOW–TILE_FREQ_HIGH range so freq scrolling only changes which
  // Y slice of the warp canvas we copy — no re-warping needed.
  const key = `${idx}-${H}-${S.logScale.toFixed(3)}`;
  if (warpCache.has(key)) return warpCache.get(key);

  const osc  = document.createElement('canvas');
  osc.width  = img.naturalWidth;
  osc.height = H;
  const oc2  = osc.getContext('2d');
  // High-quality interpolation on the warp context so band-slice edges blend
  // smoothly rather than showing nearest-neighbour step artefacts.
  oc2.imageSmoothingEnabled = true;
  oc2.imageSmoothingQuality = 'high';

  const a = S.logScale;
  if (a < 0.001) {
    // ── Linear fast path: scale the full tile to canvas height in one shot ──
    // The render loop will crop this canvas to the freq viewport via
    // _fullRangeFToY — no per-frame work and no freq-dependent cache entry.
    oc2.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight,
                       0, 0, img.naturalWidth, H);
  } else {
    // ── Log/blended path: per-band warp over the full freq range ────────────
    // Enforce per-frame budget: if we've already warped LOG_WARP_PER_FRAME tiles
    // this render call, don't block further — return null so the caller can show
    // a linear fallback and re-schedule another render to continue warping.
    if (_logWarpBudget <= 0) {
      warpCache.delete(key);  // don't cache the empty canvas we just created
      return null;
    }
    _logWarpBudget--;

    const BANDS = Math.ceil(H / 2);
    for (let b = 0; b < BANDS; b++) {
      const cy  = b * 2;
      const f0  = _fullRangeYToF(cy,     H, a);
      const f1  = _fullRangeYToF(cy + 2, H, a);
      const ty0 = (TILE_FREQ_HIGH - f0) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
      const ty1 = (TILE_FREQ_HIGH - f1) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
      if (ty0 < 0 || ty1 > 1.01 || ty1 <= ty0) continue;
      const imgY0 = ty0 * img.naturalHeight;
      const imgH  = Math.max(0.5, (ty1 - ty0) * img.naturalHeight);
      oc2.drawImage(img, 0, imgY0, img.naturalWidth, imgH,
                         0, cy,   img.naturalWidth, 2);
    }
  }

  warpCache.set(key, osc);
  return osc;
}

// ─── Mip-aware tile blit helper ──────────────────────────────────────────────
// drawImage on a GPU canvas has no mipmaps, so at extreme downsampling ratios
// (e.g. 278:1 at full zoom-out) even 'high' quality degenerates to random
// sampling.  This wrapper pre-builds successive 2:1 halvings of the warp canvas,
// cached in warpCache as "${key}-m1", "-m2", …, and returns the level whose
// ratio to dstW is ≤ 2:1 — the sweet spot for bilinear area averaging.
// Returns { canvas, sx, sw } with src coords scaled to the mip, or null if the
// underlying warp isn't ready yet (caller falls back to linear img).
function _getWarpedTileBlit(idx, img, H, srcX, srcW, dstW, warpCache = S.tileWarpCache) {
  const base = _getWarpedTile(idx, img, H, warpCache);
  if (!base) return null;  // log warp budget exceeded — caller handles fallback

  const ratio = srcW / Math.max(1, dstW);
  if (ratio <= 2) return { canvas: base, sx: srcX, sw: srcW };

  // How many 2:1 halvings needed to bring ratio ≤ 2?
  const levelsNeeded = Math.ceil(Math.log2(ratio / 2));
  const baseKey = `${idx}-${H}-${S.logScale.toFixed(3)}`;

  let cur = base;
  for (let level = 1; level <= levelsNeeded; level++) {
    if (cur.width <= 1) break;
    const mipKey = `${baseKey}-m${level}`;
    let mip = warpCache.get(mipKey);
    if (!mip) {
      const mipW = Math.max(1, Math.floor(cur.width / 2));
      mip = document.createElement('canvas');
      mip.width  = mipW;
      mip.height = H;
      const mc = mip.getContext('2d');
      mc.imageSmoothingEnabled = true;
      mc.imageSmoothingQuality = 'high';
      mc.drawImage(cur, 0, 0, cur.width, cur.height, 0, 0, mipW, H);
      warpCache.set(mipKey, mip);
    }
    cur = mip;
  }

  // Scale source X coordinates proportionally to the selected mip level
  const f = cur.width / base.width;
  return { canvas: cur, sx: srcX * f, sw: srcW * f };
}

// ─── Tile loading ─────────────────────────────────────────────
function loadTile(idx) {
  // Capture the active Maps at call time so mode changes mid-load don't corrupt
  // the wrong cache (the closure holds references, not the S.* property names).
  const imgMap    = S.tileImgs;
  const readyMap  = S.tileReady;
  const warpCache = S.tileWarpCache;
  const endpoint  = S._tileEndpoint;

  if (imgMap.has(idx)) return;
  const img = new Image();
  imgMap.set(idx, img);
  readyMap.set(idx, false);
  img.onload = () => {
    readyMap.set(idx, true);
    // Pre-warp immediately so the next render() only needs 1 drawImage per tile.
    // Doing it here (async, after network load) keeps the render loop cheap.
    // Bypass the per-frame budget — this runs outside the render loop.
    const H = SPEC_H();
    if (H > 0) { _logWarpBudget = 999; _getWarpedTile(idx, img, H, warpCache); }
    scheduleRender();
  };
  img.src = `/api/${endpoint}/${idx}?v=${S.tileVersion}&f=${S.fid}`;
}

// ─── Switch between spectrogram tile modes ────────────────────────────────────
// Swaps S.tileImgs / S.tileReady / S.tileWarpCache to point at the backing Maps
// for the requested mode, then re-triggers tile loading.  render.js needs no
// changes because it always reads the live S.tileImgs reference.
function switchSpectrogramMode(mode) {
  if (mode === S.spectrogramMode) return;
  S.spectrogramMode = mode;
  if (mode === 'reassigned') {
    S.tileImgs      = S._reassignedTileImgs;
    S.tileReady     = S._reassignedTileReady;
    S.tileWarpCache = S._reassignedWarpCache;
    S._tileEndpoint = 'tile_reassigned';
  } else {
    S.tileImgs      = S._stftTileImgs;
    S.tileReady     = S._stftTileReady;
    S.tileWarpCache = S._stftWarpCache;
    S._tileEndpoint = 'tile';
  }
  // Update button active states
  const btnStft = document.getElementById('btn-spec-stft');
  const btnReas = document.getElementById('btn-spec-reassigned');
  if (btnStft) btnStft.classList.toggle('clf-active', mode === 'stft');
  if (btnReas) btnReas.classList.toggle('clf-active', mode === 'reassigned');
  ensureTiles();
  scheduleRender();
}

function loadMaskTile(idx) {
  if (S.maskTileImgs.has(idx)) return;
  const img = new Image();
  S.maskTileImgs.set(idx, img);
  S.maskTileReady.set(idx, false);
  img.onload = () => {
    S.maskTileReady.set(idx, true);
    const H = SPEC_H();
    if (H > 0) { _logWarpBudget = 999; _getWarpedTile(idx, img, H, S.maskTileWarpCache); }
    scheduleRender();
  };
  img.onerror = () => {
    S.maskTileImgs.delete(idx);
    S.maskTileReady.delete(idx);
    setTimeout(() => loadMaskTile(idx), 3000);
  };
  img.src = `/api/tile_mask/${idx}?v=${S.tileVersion}&f=${S.fid}`;
}

function loadFlatTile(idx) {
  if (S.flatTileImgs.has(idx)) return;
  const img = new Image();
  S.flatTileImgs.set(idx, img);
  S.flatTileReady.set(idx, false);
  img.onload = () => {
    S.flatTileReady.set(idx, true);
    const H = SPEC_H();
    if (H > 0) { _logWarpBudget = 999; _getWarpedTile(idx, img, H, S.flatTileWarpCache); }
    scheduleRender();
  };
  img.src = `/api/tile_flat/${idx}?v=${S.tileVersion}&f=${S.fid}`;
}

function ensureTiles() {
  const viewEnd = S.viewStart + S.viewDur;
  const first   = Math.max(0, Math.floor(S.viewStart / S.tileDur) - 1);
  const last    = Math.min(S.nTiles - 1, Math.ceil(viewEnd / S.tileDur));
  for (let i = first; i <= last; i++) {
    loadTile(i);
    if (S.crossfade > 0) loadMaskTile(i);
    if (S.flatness  > 0) loadFlatTile(i);
  }
  // Prefetch neighbours
  if (first > 0) {
    loadTile(first - 1);
    if (S.crossfade > 0) loadMaskTile(first - 1);
    if (S.flatness  > 0) loadFlatTile(first - 1);
  }
  if (last < S.nTiles - 1) {
    loadTile(last + 1);
    if (S.crossfade > 0) loadMaskTile(last + 1);
    if (S.flatness  > 0) loadFlatTile(last + 1);
  }
}

// ─── Rendering ───────────────────────────────────────────────
