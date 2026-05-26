import io, json, os
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from PIL import Image

from config import (
    TILE_DURATION, TILE_W, TILE_H,
    FREQ_LOW, FREQ_HIGH,
    D_NPERSEG, D_NOVERLAP,
    TILE_NORM_VERSION,
)
from state import (
    audio_lock, audio_fh, finfo, all_calls,
    tile_cache, tile_lock, TILE_DIR,
    mask_tile_cache, mask_tile_lock,
    flat_tile_cache, flat_tile_lock,
    _inferno,
    calls_ready,
)
import state


# ─────────────────────────────────────────────
# Tile generation
# ─────────────────────────────────────────────
def _init_tile_norm():
    """Compute global vmin/vmax from a sample of tiles and purge stale cache.

    Global normalization means every tile uses the same dB scale so brightness
    is consistent across the recording.  We sample ~30 tiles spread across the
    file, compute per-tile 2nd and 99.9th percentiles (bat calls are ~1% of
    pixels, so 99.9th is needed to land inside call energy), then take the
    median for vmin and 75th-percentile for vmax.

    Flat tiles use per-frequency per-tile normalization (see make_flat_tile),
    so they don't need global stats computed here.
    """
    norm_path = os.path.join(state.TILE_DIR, "norm.json")

    # Check whether cached stats already match the current version
    if os.path.exists(norm_path):
        try:
            with open(norm_path) as fh:
                ndata = json.load(fh)
            if ndata.get("version") == TILE_NORM_VERSION:
                state._global_vmin   = ndata["vmin"]
                state._global_vmax   = ndata["vmax"]
                state._global_vmin_f = np.array(ndata["vmin_f"]) if "vmin_f" in ndata else None
                state._global_vmax_f = np.array(ndata["vmax_f"]) if "vmax_f" in ndata else None
                n_f = len(state._global_vmin_f) if state._global_vmin_f is not None else 0
                print(f"  Tile norm: global mode (v{TILE_NORM_VERSION}), "
                      f"vmin={state._global_vmin:.1f}  vmax={state._global_vmax:.1f} dB  |  "
                      f"per-freq flat: {n_f} bins")
                return
            else:
                print("  Tile norm version changed — purging cached tiles…")
        except Exception:
            pass

    # Version mismatch or missing → purge stale tile PNGs
    if os.path.isdir(state.TILE_DIR):
        for fn in os.listdir(state.TILE_DIR):
            if (fn.startswith("tile_") or fn.startswith("mask_tile_")
                    or fn.startswith("flat_tile_")) and fn.endswith(".png"):
                try:
                    os.remove(os.path.join(state.TILE_DIR, fn))
                except Exception:
                    pass
    with tile_lock:
        tile_cache.clear()
    with mask_tile_lock:
        mask_tile_cache.clear()
    with flat_tile_lock:
        flat_tile_cache.clear()

    # Sample ~30 tiles evenly across the recording to compute global stats
    sr  = finfo["sr"]
    dur = finfo["duration_s"]
    ntiles = int(np.ceil(dur / TILE_DURATION))
    sample_idxs = np.linspace(0, ntiles - 1, min(30, ntiles), dtype=int)

    print(f"  Computing global tile normalization from {len(sample_idxs)} tiles…",
          flush=True)
    tile_vmins, tile_vmaxs = [], []
    tile_pct_los, tile_pct_his = [], []   # per-frequency arrays for flat normalization

    for tidx in sample_idxs:
        try:
            t0 = tidx * TILE_DURATION
            t1 = min(t0 + TILE_DURATION, dur)
            f0 = int(t0 * sr); f1 = int(t1 * sr)
            if f1 <= f0:
                continue
            with audio_lock:
                audio_fh.seek(f0)
                audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            f_s, _, Sxx = signal.spectrogram(
                mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
            bm  = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
            Sdb = 10 * np.log10(Sxx[bm, :] + 1e-12)   # (n_freq, n_time)

            # Global (scalar) stats for raw tiles:
            # bat calls cover ~1% of pixels, so 99.9th captures call energy.
            tile_vmins.append(float(np.percentile(Sdb,   2.0)))
            tile_vmaxs.append(float(np.percentile(Sdb, 99.9)))

            # Per-frequency stats for flat tiles: one number per bin per tile.
            # Stacking across sample tiles then taking inter-tile percentiles
            # gives time-invariant, frequency-specific normalization constants.
            tile_pct_los.append(np.percentile(Sdb, 2.0,  axis=1))   # (n_freq,)
            tile_pct_his.append(np.percentile(Sdb, 99.9, axis=1))   # (n_freq,)

        except Exception as exc:
            print(f"    tile {tidx} failed: {exc}")

    if tile_vmins:
        state._global_vmin = float(np.percentile(tile_vmins, 50))
        state._global_vmax = float(np.percentile(tile_vmaxs, 75))
    else:
        state._global_vmin = -100.0
        state._global_vmax =  -30.0

    if tile_pct_los:
        plo = np.vstack(tile_pct_los)     # (n_samples, n_freq)
        phi = np.vstack(tile_pct_his)
        state._global_vmin_f = np.percentile(plo, 50, axis=0)   # median noise floor per bin
        state._global_vmax_f = np.percentile(phi, 75, axis=0)   # 75th-pct signal ceiling per bin
    else:
        state._global_vmin_f = None
        state._global_vmax_f = None

    n_f = len(state._global_vmin_f) if state._global_vmin_f is not None else 0
    print(f"  Tile norm: global mode (v{TILE_NORM_VERSION}), "
          f"vmin={state._global_vmin:.1f}  vmax={state._global_vmax:.1f} dB  |  "
          f"per-freq flat: {n_f} bins — tiles will be regenerated")

    try:
        os.makedirs(state.TILE_DIR, exist_ok=True)
        ndata = {
            "version": TILE_NORM_VERSION,
            "mode":    "global",
            "vmin":    state._global_vmin,
            "vmax":    state._global_vmax,
        }
        if state._global_vmin_f is not None:
            ndata["vmin_f"] = state._global_vmin_f.tolist()
            ndata["vmax_f"] = state._global_vmax_f.tolist()
        with open(norm_path, "w") as fh:
            json.dump(ndata, fh)
    except Exception as exc:
        print(f"  Warning: could not write norm.json ({exc})")


def make_tile(tidx):
    # 1. RAM cache (fastest)
    with tile_lock:
        if tidx in tile_cache:
            return tile_cache[tidx]

    # 2. Disk cache (fast — avoids re-running STFT)
    if state.TILE_DIR:
        disk_path = os.path.join(state.TILE_DIR, f"tile_{tidx:04d}.png")
        if os.path.exists(disk_path):
            with open(disk_path, "rb") as fh:
                data = fh.read()
            with tile_lock:
                tile_cache[tidx] = data
            return data

    # 3. Generate from audio
    sr     = finfo["sr"]
    dur    = finfo["duration_s"]
    t0     = tidx * TILE_DURATION
    t1     = min(t0 + TILE_DURATION, dur)
    f0     = int(t0 * sr);  f1 = int(t1 * sr)

    with audio_lock:
        audio_fh.seek(f0)
        audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm   = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
    Sdb  = 10 * np.log10(Sxx[bm, :] + 1e-12)

    # Global normalization: same dB scale for every tile so call-isolated
    # crossfade makes sense (both spectrograms share one scale).
    # _global_vmin/_global_vmax are computed at startup from ~30 sampled tiles.
    arr  = np.clip((Sdb - state._global_vmin) / max(state._global_vmax - state._global_vmin, 1e-6), 0, 1)
    rgb  = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)

    pil  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    # 4. Save to disk cache (persist across server restarts)
    if state.TILE_DIR:
        try:
            os.makedirs(state.TILE_DIR, exist_ok=True)
            with open(disk_path, "wb") as fh:
                fh.write(data)
        except Exception:
            pass

    # 5. Store in RAM cache (no eviction — 253 tiles ≈ 25 MB, negligible)
    with tile_lock:
        tile_cache[tidx] = data
    return data


def _pregenerate_tiles():
    """Background thread: walk every tile so they're disk-cached before the user zooms out.
    Also pre-generates mask tiles once detection results are available.
    """
    ntiles = int(np.ceil(finfo["duration_s"] / TILE_DURATION))
    missing = [i for i in range(ntiles)
               if i not in tile_cache and
               not os.path.exists(os.path.join(state.TILE_DIR, f"tile_{i:04d}.png"))]
    if not missing:
        print("All raw tiles already cached on disk.")
    else:
        print(f"Pre-generating {len(missing)} raw tiles in background…", flush=True)
        for i in missing:
            try:
                make_tile(i)
            except Exception as exc:
                print(f"  tile {i} failed: {exc}")
        print(f"Raw tile pre-generation done ({ntiles} tiles total).", flush=True)

    # Pre-generate flat (frequency-compensated) tiles in parallel with detection
    flat_missing = [i for i in range(ntiles)
                    if i not in flat_tile_cache and
                    not os.path.exists(os.path.join(state.TILE_DIR, f"flat_tile_{i:04d}.png"))]
    if not flat_missing:
        print("All flat tiles already cached on disk.")
    else:
        print(f"Pre-generating {len(flat_missing)} flat tiles in background…", flush=True)
        for i in flat_missing:
            try:
                make_flat_tile(i)
            except Exception as exc:
                print(f"  flat tile {i} failed: {exc}")
        print(f"Flat tile pre-generation done ({ntiles} tiles total).", flush=True)

    # Wait for detection to finish, then pre-generate mask tiles
    calls_ready.wait()
    mask_missing = [i for i in range(ntiles)
                    if i not in mask_tile_cache and
                    not os.path.exists(os.path.join(state.TILE_DIR, f"mask_tile_{i:04d}.png"))]
    if not mask_missing:
        print("All mask tiles already cached on disk.")
        return
    print(f"Pre-generating {len(mask_missing)} mask tiles in background…", flush=True)
    for i in mask_missing:
        try:
            make_mask_tile(i)
        except Exception as exc:
            print(f"  mask tile {i} failed: {exc}")
    print(f"Mask tile pre-generation done ({ntiles} tiles total).", flush=True)


def _compute_call_mask(tidx):
    """Build a soft 2-D mask marking call regions for tile `tidx`.

    Returns a float32 array of shape (n_freq, n_time) in the same coordinate
    system as Sdb: row 0 = lowest frequency (FREQ_LOW), row n-1 = highest
    (FREQ_HIGH).  Values are in [0, 1]: 1 = call energy, 0 = background.

    Algorithm
    ---------
    For each detected call whose time range overlaps this tile:
      • Walk every contour point that falls inside the tile's time window.
      • At each time frame, add a Gaussian bump in the frequency direction
        centred on the contour frequency.  σ = max(bandwidth/3, 2 kHz).
      • For constant-frequency bats (cf_frac > 0.35, e.g. TABR) also add a
        70%-amplitude Gaussian at twice the fundamental frequency so the
        second harmonic is preserved when the mask is applied.
    After all calls are accumulated, smooth in time (σ ≈ 1.5 frames) to
    connect adjacent contour points and soften hard edges.
    """
    sr  = finfo["sr"]
    dur = finfo["duration_s"]
    t0_tile = tidx * TILE_DURATION
    t1_tile = min(t0_tile + TILE_DURATION, dur)

    # Compute the display STFT just for the frequency/time arrays (Sxx unused)
    f0 = int(t0_tile * sr); f1 = int(t1_tile * sr)
    with audio_lock:
        audio_fh.seek(f0)
        audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f_s, t_s, _ = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm    = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    f_arr = f_s[bm]          # Hz, ascending
    n_freq = len(f_arr)
    n_time = len(t_s)

    mask = np.zeros((n_freq, n_time), dtype=np.float32)

    if not all_calls:
        return mask

    tile_dur = t1_tile - t0_tile

    for c in all_calls:
        if c["t1"] < t0_tile or c["t0"] > t1_tile:
            continue
        contour = c.get("contour")
        if not contour:
            continue

        # σ in Hz: one-third of the call's measured bandwidth, minimum 2 kHz
        bw_hz    = max(c.get("bw", max(c["Fmax"] - c["Fmin"], 2.0)) * 1000, 2000)
        sigma_hz = max(bw_hz / 3.0, 2000.0)
        cf_frac  = c.get("cf_frac", 0.5)
        add_harmonic = cf_frac > 0.35

        for ct, cf_khz in contour:
            if ct < t0_tile or ct > t1_tile:
                continue
            # Map absolute time → tile time index
            ti = int((ct - t0_tile) / tile_dur * n_time)
            ti = max(0, min(ti, n_time - 1))

            cf_hz = cf_khz * 1000.0
            # Gaussian in frequency
            gauss = np.exp(-0.5 * ((f_arr - cf_hz) / sigma_hz) ** 2)
            mask[:, ti] = np.maximum(mask[:, ti], gauss)

            # Second harmonic for CF bats (adds harmonic ridge to the mask)
            if add_harmonic:
                cf2_hz = cf_hz * 2.0
                if FREQ_LOW <= cf2_hz <= FREQ_HIGH:
                    g2 = np.exp(-0.5 * ((f_arr - cf2_hz) / sigma_hz) ** 2) * 0.7
                    mask[:, ti] = np.maximum(mask[:, ti], g2)

    # Smooth in time to fill gaps between contour points
    if n_time > 1:
        mask = gaussian_filter1d(mask, sigma=1.5, axis=1)

    return np.clip(mask, 0.0, 1.0)


def make_mask_tile(tidx):
    """Generate an RGBA mask tile: R=G=B=0, A=(1−mask)×255.

    When composited on top of the raw spectrogram at opacity `α` (crossfade):
      • Call regions (mask≈1): A≈0 → transparent → raw tile shows through.
      • Background (mask≈0): A≈255 → black at globalAlpha=α → dims the background.
    """
    with mask_tile_lock:
        if tidx in mask_tile_cache:
            return mask_tile_cache[tidx]

    if state.TILE_DIR:
        disk_path = os.path.join(state.TILE_DIR, f"mask_tile_{tidx:04d}.png")
        if os.path.exists(disk_path):
            with open(disk_path, "rb") as fh:
                data = fh.read()
            with mask_tile_lock:
                mask_tile_cache[tidx] = data
            return data
    else:
        disk_path = None

    # Compute mask (n_freq × n_time), row 0 = lowest freq
    mask_arr = _compute_call_mask(tidx)

    # Flip vertically so row 0 = highest freq (same orientation as make_tile)
    mask_flipped = mask_arr[::-1, :]

    # Build RGBA: R=G=B=0, A=(1-mask)*255
    h, w = mask_flipped.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = np.round((1.0 - mask_flipped) * 255).astype(np.uint8)

    pil  = Image.fromarray(rgba, mode="RGBA").resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    if disk_path:
        try:
            os.makedirs(state.TILE_DIR, exist_ok=True)
            with open(disk_path, "wb") as fh:
                fh.write(data)
        except Exception:
            pass

    with mask_tile_lock:
        mask_tile_cache[tidx] = data
    return data


def _flat_gain_db(f_hz):
    """Per-frequency gain correction (dB) to compensate for the EM258 capsule rolloff.

    The Avisoft EM258 (and similar CCP electret ultrasonic capsules) have an
    approximately first-order (6 dB/octave) rolloff starting around 40 kHz.
    This function returns the dB boost needed to flatten the response so that
    energy at higher frequencies is displayed at equivalent brightness to
    lower-frequency energy of the same physical amplitude.

    Model: first-order high-pass shelf starting at f_ref = 40 kHz.
      boost(f) = max(0,  20 · log10(f / f_ref))  dB
    Gives:  13 kHz →  0.0 dB
            40 kHz →  0.0 dB  (rolloff start)
            60 kHz → +3.5 dB
            80 kHz → +6.0 dB  (one octave above f_ref)
            96 kHz → +7.6 dB

    Consult the actual EM258 datasheet (Avisoft-Bioacoustics) to calibrate
    f_ref and the rolloff order for your specific unit.
    """
    f_ref = 40_000.0  # Hz — frequency where capsule rolloff begins
    boost = np.maximum(0.0, 20.0 * np.log10(np.maximum(f_hz, f_ref) / f_ref))
    return boost   # shape matches f_hz


def make_flat_tile(tidx):
    """Per-frequency-normalised spectrogram tile (the "Flat" view).

    Why not a mic-response gain boost?
    -----------------------------------
    Applying a frequency-dependent dB gain (e.g. +7.6 dB at 96 kHz to
    compensate EM258 rolloff) is mathematically identical to multiplying the
    linear power spectrum by a frequency-dependent constant.  Both operations
    lift the *noise floor* by the same factor as the signal, so the background
    becomes progressively brighter at high frequencies — the "pink gradient"
    the user noticed.  No form of linear gain (additive in dB, multiplicative
    in power, or IIR/FIR in the time domain) can keep the noise floor dark
    while boosting bat calls, because it is frequency-blind: it doesn't know
    whether a given pixel is a bat call or background noise.

    Per-frequency normalisation (spectral whitening)
    -------------------------------------------------
    For each frequency bin k, we compute its local noise floor (2nd percentile
    over time) and its local signal ceiling (99.9th percentile over time) from
    this tile's own data, then map:

        arr[k, t] = clip( (Sdb[k,t] − lo[k]) / (hi[k] − lo[k]),  0, 1 )

    Result: the noise appears dark at every frequency; energy that stands
    above the local floor (bat calls, harmonics) appears bright — regardless
    of the recording chain's absolute frequency response.

    Physically this removes any spectrally-coloured noise source (mic rolloff,
    preamp shape, narrowband interference) and shows *deviation from the local
    noise floor*.  It is not a calibrated amplitude display, but for visually
    identifying calls at all frequencies it is the right tool.

    Bat calls occupy ~1% of pixels per tile, so the 99.9th-percentile captures
    call energy while the 2nd-percentile sits stably in the noise floor.
    """
    with flat_tile_lock:
        if tidx in flat_tile_cache:
            return flat_tile_cache[tidx]

    if state.TILE_DIR:
        disk_path = os.path.join(state.TILE_DIR, f"flat_tile_{tidx:04d}.png")
        if os.path.exists(disk_path):
            with open(disk_path, "rb") as fh:
                data = fh.read()
            with flat_tile_lock:
                flat_tile_cache[tidx] = data
            return data
    else:
        disk_path = None

    sr  = finfo["sr"]
    dur = finfo["duration_s"]
    t0  = tidx * TILE_DURATION
    t1  = min(t0 + TILE_DURATION, dur)
    f0  = int(t0 * sr); f1 = int(t1 * sr)

    with audio_lock:
        audio_fh.seek(f0)
        audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f_s, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm  = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    Sdb = 10 * np.log10(Sxx[bm, :] + 1e-12)   # (n_freq, n_time)

    # Per-frequency normalization using globally pre-computed constants so the
    # scale is time-invariant (same for every tile → no brightness jumps at
    # tile boundaries).  _global_vmin_f/_global_vmax_f are (n_freq,) arrays
    # computed from 30 sample tiles during startup.
    n_freq = Sdb.shape[0]
    if (state._global_vmin_f is not None and state._global_vmax_f is not None
            and len(state._global_vmin_f) == n_freq):
        lo = state._global_vmin_f[:, np.newaxis]   # (n_freq, 1) → broadcasts over time
        hi = state._global_vmax_f[:, np.newaxis]
    else:
        # Fallback if global stats are missing or frequency-bin count changed
        lo = np.percentile(Sdb, 2.0,  axis=1, keepdims=True)
        hi = np.percentile(Sdb, 99.9, axis=1, keepdims=True)
    arr = np.clip((Sdb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)

    pil  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    if disk_path:
        try:
            os.makedirs(state.TILE_DIR, exist_ok=True)
            with open(disk_path, "wb") as fh:
                fh.write(data)
        except Exception:
            pass

    with flat_tile_lock:
        flat_tile_cache[tidx] = data
    return data
