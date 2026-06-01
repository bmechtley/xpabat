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
from state import _inferno


# ─────────────────────────────────────────────
# Tile generation
# ─────────────────────────────────────────────

def _init_tile_norm(entry):
    """Compute global vmin/vmax from a sample of tiles and purge stale cache.

    Global normalization means every tile uses the same dB scale so brightness
    is consistent across the recording.  We sample ~30 tiles spread across the
    file, compute per-tile 2nd and 99.9th percentiles (bat calls are ~1% of
    pixels, so 99.9th is needed to land inside call energy), then take the
    median for vmin and 75th-percentile for vmax.
    """
    norm_path = os.path.join(entry.tile_dir, "norm.json")

    if os.path.exists(norm_path):
        try:
            with open(norm_path) as fh:
                ndata = json.load(fh)
            if ndata.get("version") == TILE_NORM_VERSION:
                entry.vmin    = ndata["vmin"]
                entry.vmax    = ndata["vmax"]
                entry.psd_p01 = ndata.get("psd_p01", entry.vmin)
                entry.psd_p99 = ndata.get("psd_p99", entry.vmax)
                entry.vmin_f  = np.array(ndata["vmin_f"]) if "vmin_f" in ndata else None
                entry.vmax_f  = np.array(ndata["vmax_f"]) if "vmax_f" in ndata else None
                n_f = len(entry.vmin_f) if entry.vmin_f is not None else 0
                print(f"  Tile norm ({entry.name}): v{TILE_NORM_VERSION}, "
                      f"vmin={entry.vmin:.1f}  vmax={entry.vmax:.1f} dB  |  "
                      f"per-freq flat: {n_f} bins")
                return
            else:
                print(f"  Tile norm version changed for {entry.name} — purging cached tiles…")
        except Exception:
            pass

    # Version mismatch or missing → purge stale tile PNGs
    if os.path.isdir(entry.tile_dir):
        for fn in os.listdir(entry.tile_dir):
            if (fn.startswith("tile_") or fn.startswith("mask_tile_")
                    or fn.startswith("flat_tile_")) and fn.endswith(".png"):
                try:
                    os.remove(os.path.join(entry.tile_dir, fn))
                except Exception:
                    pass
    with entry.tile_lock:
        entry.tile_cache.clear()
    with entry.mask_tile_lock:
        entry.mask_tile_cache.clear()
    with entry.flat_tile_lock:
        entry.flat_tile_cache.clear()

    sr     = entry.finfo["sr"]
    dur    = entry.finfo["duration_s"]
    ntiles = int(np.ceil(dur / TILE_DURATION))
    sample_idxs = np.linspace(0, ntiles - 1, min(30, ntiles), dtype=int)

    print(f"  Computing tile normalization for {entry.name} ({len(sample_idxs)} tiles)…",
          flush=True)
    tile_vmins, tile_vmaxs = [], []
    tile_pct_los, tile_pct_his = [], []
    tile_sdb_sub = []   # subsampled dB values for 1 %/99 % file-wide percentiles

    for tidx in sample_idxs:
        try:
            t0 = tidx * TILE_DURATION
            t1 = min(t0 + TILE_DURATION, dur)
            f0 = int(t0 * sr); f1 = int(t1 * sr)
            if f1 <= f0:
                continue
            with entry.audio_lock:
                entry.audio_fh.seek(f0)
                audio = entry.audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            f_s, _, Sxx = signal.spectrogram(
                mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
            bm  = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
            Sdb = 10 * np.log10(Sxx[bm, :] + 1e-12)

            tile_vmins.append(float(np.percentile(Sdb,   2.0)))
            tile_vmaxs.append(float(np.percentile(Sdb, 99.9)))
            tile_pct_los.append(np.percentile(Sdb, 2.0,  axis=1))
            tile_pct_his.append(np.percentile(Sdb, 99.9, axis=1))
            # Subsample every 16th time column to keep memory manageable
            tile_sdb_sub.append(Sdb[:, ::16].ravel())
        except Exception as exc:
            print(f"    tile {tidx} failed: {exc}")

    if tile_vmins:
        entry.vmin = float(np.percentile(tile_vmins, 50))
        entry.vmax = float(np.percentile(tile_vmaxs, 75))
    else:
        entry.vmin = -100.0
        entry.vmax =  -30.0

    if tile_sdb_sub:
        all_sdb = np.concatenate(tile_sdb_sub)
        entry.psd_p01 = float(np.percentile(all_sdb,  1))
        entry.psd_p99 = float(np.percentile(all_sdb, 99))
    else:
        entry.psd_p01 = -120.0
        entry.psd_p99 =  -40.0

    if tile_pct_los:
        plo = np.vstack(tile_pct_los)
        phi = np.vstack(tile_pct_his)
        entry.vmin_f = np.percentile(plo, 50, axis=0)
        entry.vmax_f = np.percentile(phi, 75, axis=0)
    else:
        entry.vmin_f = None
        entry.vmax_f = None

    n_f = len(entry.vmin_f) if entry.vmin_f is not None else 0
    print(f"  Tile norm ({entry.name}): v{TILE_NORM_VERSION}, "
          f"vmin={entry.vmin:.1f}  vmax={entry.vmax:.1f} dB  |  "
          f"per-freq flat: {n_f} bins — tiles will be regenerated")

    try:
        os.makedirs(entry.tile_dir, exist_ok=True)
        ndata = {
            "version": TILE_NORM_VERSION,
            "mode":    "global",
            "vmin":    entry.vmin,
            "vmax":    entry.vmax,
            "psd_p01": entry.psd_p01,
            "psd_p99": entry.psd_p99,
        }
        if entry.vmin_f is not None:
            ndata["vmin_f"] = entry.vmin_f.tolist()
            ndata["vmax_f"] = entry.vmax_f.tolist()
        with open(norm_path, "w") as fh:
            json.dump(ndata, fh)
    except Exception as exc:
        print(f"  Warning: could not write norm.json ({exc})")


def make_tile(entry, tidx):
    # 1. RAM cache
    with entry.tile_lock:
        if tidx in entry.tile_cache:
            return entry.tile_cache[tidx]

    # 2. Disk cache
    disk_path = os.path.join(entry.tile_dir, f"tile_{tidx:04d}.png")
    if os.path.exists(disk_path):
        with open(disk_path, "rb") as fh:
            data = fh.read()
        with entry.tile_lock:
            entry.tile_cache[tidx] = data
        return data

    # 3. Generate from audio
    sr  = entry.finfo["sr"]
    dur = entry.finfo["duration_s"]
    t0  = tidx * TILE_DURATION
    t1  = min(t0 + TILE_DURATION, dur)
    f0  = int(t0 * sr); f1 = int(t1 * sr)

    with entry.audio_lock:
        entry.audio_fh.seek(f0)
        audio = entry.audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm  = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
    Sdb = 10 * np.log10(Sxx[bm, :] + 1e-12)
    arr = np.clip((Sdb - entry.vmin) / max(entry.vmax - entry.vmin, 1e-6), 0, 1)
    rgb = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)

    pil  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    # 4. Save to disk cache
    try:
        os.makedirs(entry.tile_dir, exist_ok=True)
        with open(disk_path, "wb") as fh:
            fh.write(data)
    except Exception:
        pass

    # 5. Store in RAM cache
    with entry.tile_lock:
        entry.tile_cache[tidx] = data
    return data


def _pregenerate_mask_tiles(entry):
    """Background thread: generate mask tiles once detection is available.
    Polls entry.stop_event so startup can abort cleanly.
    """
    def stopped():
        return entry.stop_event.is_set()

    if stopped() or not entry.finfo:
        return

    mp     = entry.mask_progress
    ntiles = int(np.ceil(entry.finfo["duration_s"] / TILE_DURATION))

    mp["status"] = "waiting"
    while not entry.calls_ready.is_set():
        if stopped():
            mp["status"] = "idle"
            return
        entry.calls_ready.wait(timeout=0.5)

    if stopped():
        mp["status"] = "idle"
        return

    mask_missing = [i for i in range(ntiles)
                    if i not in entry.mask_tile_cache and
                    not os.path.exists(os.path.join(entry.tile_dir, f"mask_tile_{i:04d}.png"))]
    mp["total"] = ntiles
    mp["done"]  = ntiles - len(mask_missing)
    if not mask_missing:
        mp["status"] = "done"
        print(f"All mask tiles already cached ({entry.name}).")
        return
    mp["status"] = "running"
    print(f"Pre-generating {len(mask_missing)} mask tiles for {entry.name}…", flush=True)
    for i in mask_missing:
        if stopped():
            mp["status"] = "idle"
            return
        try:
            make_mask_tile(entry, i)
            mp["done"] += 1
        except Exception as exc:
            print(f"  mask tile {i} failed: {exc}")
    mp["status"] = "done"
    print(f"Mask tile pre-generation done ({entry.name}, {ntiles} total).", flush=True)


def _compute_call_mask(entry, tidx):
    """Build a soft 2-D mask marking call regions for tile `tidx`.

    Returns a float32 array of shape (n_freq, n_time) in the same coordinate
    system as Sdb: row 0 = lowest frequency (FREQ_LOW), row n-1 = highest
    (FREQ_HIGH).  Values are in [0, 1]: 1 = call energy, 0 = background.
    """
    sr      = entry.finfo["sr"]
    dur     = entry.finfo["duration_s"]
    t0_tile = tidx * TILE_DURATION
    t1_tile = min(t0_tile + TILE_DURATION, dur)

    f0 = int(t0_tile * sr); f1 = int(t1_tile * sr)
    with entry.audio_lock:
        entry.audio_fh.seek(f0)
        audio = entry.audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f_s, t_s, _ = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm    = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    f_arr = f_s[bm]
    n_freq = len(f_arr)
    n_time = len(t_s)

    mask = np.zeros((n_freq, n_time), dtype=np.float32)

    if not entry.all_calls:
        return mask

    tile_dur = t1_tile - t0_tile

    for c in entry.all_calls:
        if c["t1"] < t0_tile or c["t0"] > t1_tile:
            continue
        contour = c.get("contour")
        if not contour:
            continue

        bw_hz    = max(c.get("bw", max(c["Fmax"] - c["Fmin"], 2.0)) * 1000, 2000)
        sigma_hz = max(bw_hz / 3.0, 2000.0)
        cf_frac  = c.get("cf_frac", 0.5)
        add_harmonic = cf_frac > 0.35

        for ct, cf_khz in contour:
            if ct < t0_tile or ct > t1_tile:
                continue
            ti = int((ct - t0_tile) / tile_dur * n_time)
            ti = max(0, min(ti, n_time - 1))

            cf_hz = cf_khz * 1000.0
            gauss = np.exp(-0.5 * ((f_arr - cf_hz) / sigma_hz) ** 2)
            mask[:, ti] = np.maximum(mask[:, ti], gauss)

            if add_harmonic:
                cf2_hz = cf_hz * 2.0
                if FREQ_LOW <= cf2_hz <= FREQ_HIGH:
                    g2 = np.exp(-0.5 * ((f_arr - cf2_hz) / sigma_hz) ** 2) * 0.7
                    mask[:, ti] = np.maximum(mask[:, ti], g2)

    if n_time > 1:
        mask = gaussian_filter1d(mask, sigma=1.5, axis=1)

    return np.clip(mask, 0.0, 1.0)


def make_mask_tile(entry, tidx):
    """Generate an RGBA mask tile: R=G=B=0, A=(1−mask)×255.

    When composited on top of the raw spectrogram at opacity `α` (crossfade):
      • Call regions (mask≈1): A≈0 → transparent → raw tile shows through.
      • Background (mask≈0): A≈255 → black at globalAlpha=α → dims the background.
    """
    with entry.mask_tile_lock:
        if tidx in entry.mask_tile_cache:
            return entry.mask_tile_cache[tidx]

    disk_path = os.path.join(entry.tile_dir, f"mask_tile_{tidx:04d}.png")
    if os.path.exists(disk_path):
        with open(disk_path, "rb") as fh:
            data = fh.read()
        with entry.mask_tile_lock:
            entry.mask_tile_cache[tidx] = data
        return data

    mask_arr     = _compute_call_mask(entry, tidx)
    mask_flipped = mask_arr[::-1, :]

    h, w = mask_flipped.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = np.round((1.0 - mask_flipped) * 255).astype(np.uint8)

    pil  = Image.fromarray(rgba, mode="RGBA").resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    try:
        os.makedirs(entry.tile_dir, exist_ok=True)
        with open(disk_path, "wb") as fh:
            fh.write(data)
    except Exception:
        pass

    with entry.mask_tile_lock:
        entry.mask_tile_cache[tidx] = data
    return data


def _reassigned_scatter(entry, tidx):
    """Shared audio-load + scatter computation for reassigned-spectrogram tiles.

    Loads the tile's audio with boundary padding (to suppress STFT edge
    artefacts), runs the reassigned spectrogram, applies the SNR+range gate,
    and accumulates energy into a scatter array.

    Returns
    -------
    out   : (n_freq, n_time) float64 — lightly smoothed scatter power
    f_arr : (n_freq,)        float64 — frequency bin centres in Hz
    """
    sr  = entry.finfo["sr"]
    dur = entry.finfo["duration_s"]
    t0  = tidx * TILE_DURATION
    t1  = min(t0 + TILE_DURATION, dur)

    # ── Load with boundary padding ────────────────────────────────────────────
    # At tile edges the STFT window hangs over the segment boundary and sees
    # scipy's implicit zero-padding.  The derivative window amplifies this
    # heavily, producing garbage IF estimates that scatter broadband noise
    # across the full frequency axis (visible as a bright vertical stripe).
    # Fix: load one extra window of real audio on each side, run the full
    # reassignment, then crop the time axis back to [t0, t1].
    hop   = D_NPERSEG - D_NOVERLAP
    pad_s = D_NPERSEG / sr          # ≈ 5 ms at 192 kHz

    t0_pad = max(0.0, t0 - pad_s)
    t1_pad = min(dur,  t1 + pad_s)
    f0_pad = int(t0_pad * sr)
    f1_pad = int(t1_pad * sr)

    with entry.audio_lock:
        entry.audio_fh.seek(f0_pad)
        audio = entry.audio_fh.read(f1_pad - f0_pad, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    from contour import reassigned_spectrogram
    f_s, t_s, Sxx, IF = reassigned_spectrogram(mono, sr, D_NPERSEG, D_NOVERLAP)

    # Crop to the tile's time range (±half-hop tolerance at both edges)
    t_abs     = t_s + t0_pad
    half_hop  = 0.5 * hop / sr
    tile_mask = (t_abs >= t0 - half_hop) & (t_abs < t1 + half_hop)
    if not tile_mask.any():
        tile_mask = np.ones(len(t_s), dtype=bool)
    Sxx = Sxx[:, tile_mask]
    IF  = IF[:, tile_mask]

    bm     = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    f_arr  = f_s[bm]
    S_bat  = Sxx[bm, :].astype(np.float64)
    IF_raw = IF[bm, :]
    n_freq, n_time = S_bat.shape

    # ── SNR gate ───────────────────────────────────────────────────────────────
    # Low-power noise bins have essentially random IFs that scatter energy
    # uniformly, blurring the output.  Keep only bins ≥ 4× (6 dB) above the
    # per-frequency temporal median noise floor, and only when the IF is within
    # the bat-display frequency range.
    noise_floor = np.median(S_bat, axis=1, keepdims=True)
    snr_gate    = S_bat >= (noise_floor * 4.0)
    in_range    = (IF_raw >= FREQ_LOW) & (IF_raw <= FREQ_HIGH)

    # ── IF coherence gate ──────────────────────────────────────────────────────
    # Real bat calls have coherent instantaneous frequency: adjacent frequency
    # bins point to nearly the same IF (they share a tonal ridge).  Incoherent
    # noise bins (insects, wind) have random IFs that differ wildly between
    # neighbours → |IF[f] − IF[f±1]| >> one STFT bin width.
    # Threshold: 3 × df_bin.  This passes calls (delta ~ 0) and rejects the
    # diffuse noise that causes the sustained broadband spray in quiet tiles
    # where the per-tile SNR gate threshold is too low.
    if n_freq > 2:
        df_bin     = (f_arr[-1] - f_arr[0]) / max(n_freq - 1, 1)
        coh_thresh = 3.0 * df_bin                                # Hz
        delta      = np.abs(IF_raw[1:, :] - IF_raw[:-1, :])     # (n_freq-1, n_time)
        coh        = np.ones((n_freq, n_time), dtype=bool)
        coh[:-1, :] &= delta < coh_thresh
        coh[1:,  :] &= delta < coh_thresh
        gate = snr_gate & in_range & coh
    else:
        gate = snr_gate & in_range

    IF_clamped = np.clip(IF_raw, FREQ_LOW, FREQ_HIGH)
    IF_idx     = np.clip(np.searchsorted(f_arr, IF_clamped), 0, n_freq - 1)

    out = np.zeros((n_freq, n_time), dtype=np.float64)
    for ti in range(n_time):
        w = S_bat[:, ti] * gate[:, ti].astype(np.float64)
        out[:, ti] = np.bincount(IF_idx[:, ti], weights=w, minlength=n_freq)

    # Very mild time-axis smoothing; frequency axis kept sharp.
    out = gaussian_filter1d(out, sigma=0.4, axis=1)
    return out, f_arr


_REASS_NORM_VERSION = 2   # bump to force recomputation after algorithm changes


def _init_reassigned_norm(entry):
    """Compute and cache file-wide normalization stats for reassigned tiles.

    Samples 10 evenly-spaced tiles across the recording, computes the scatter
    power distribution, and derives:

        entry.reass_norm_max    – 90th-percentile tile-peak; used as the global
                                   ceiling so all tiles share the same brightness
                                   reference (prevents seams at tile boundaries).
        entry.reass_norm_max_f  – per-frequency 90th-percentile row-peak; used
                                   by the flat-reassigned tile to equalise across
                                   frequency bands (mic/species response).

    Results are persisted to ``{tile_dir}/reass_norm.json`` so recomputation
    only happens once per file; subsequent server restarts load in milliseconds.
    """
    # Fast path: already loaded
    if getattr(entry, '_reass_norm_done', False):
        return

    norm_path = os.path.join(entry.tile_dir, "reass_norm.json")

    # Try loading from disk
    if os.path.exists(norm_path):
        try:
            with open(norm_path) as fh:
                nd = json.load(fh)
            if nd.get("version") == _REASS_NORM_VERSION:
                entry.reass_norm_max   = nd["norm_max"]
                rmf = nd.get("norm_max_f")
                entry.reass_norm_max_f = np.array(rmf, dtype=np.float64) if rmf else None
                entry._reass_norm_done = True
                print(f"  Reassigned norm loaded ({entry.name}): "
                      f"max={entry.reass_norm_max:.2e}", flush=True)
                return
        except Exception:
            pass

    # Compute from a sample of tiles
    dur    = entry.finfo["duration_s"]
    ntiles = int(np.ceil(dur / TILE_DURATION))
    idxs   = np.linspace(0, ntiles - 1, min(10, ntiles), dtype=int)

    print(f"  Computing reassigned normalization for {entry.name} "
          f"({len(idxs)} sample tiles)…", flush=True)

    tile_maxes = []
    row_maxes  = []

    for tidx in idxs:
        try:
            out, _ = _reassigned_scatter(entry, tidx)
            m = float(out.max())
            if m > 1e-30:
                tile_maxes.append(m)
                row_maxes.append(out.max(axis=1))
        except Exception as exc:
            print(f"    reass_norm tile {tidx}: {exc}")

    if tile_maxes:
        entry.reass_norm_max = float(np.percentile(tile_maxes, 90))
    else:
        entry.reass_norm_max = 1.0

    if row_maxes:
        mat = np.vstack(row_maxes)                              # (n_samples, n_freq)
        p90 = np.percentile(mat, 90, axis=0)                   # (n_freq,)
        # Floor rows with negligible energy so they stay dark rather than saturate
        p90 = np.maximum(p90, 0.001 * entry.reass_norm_max)
        entry.reass_norm_max_f = p90.astype(np.float64)
    else:
        entry.reass_norm_max_f = None

    entry._reass_norm_done = True

    print(f"  Reassigned norm done ({entry.name}): "
          f"max={entry.reass_norm_max:.2e}", flush=True)

    # Persist to disk
    try:
        os.makedirs(entry.tile_dir, exist_ok=True)
        with open(norm_path, "w") as fh:
            json.dump({
                "version":    _REASS_NORM_VERSION,
                "norm_max":   entry.reass_norm_max,
                "norm_max_f": entry.reass_norm_max_f.tolist()
                               if entry.reass_norm_max_f is not None else None,
            }, fh)
    except Exception as exc:
        print(f"  Warning: could not write reass_norm.json ({exc})")

    # Purge any tiles that were cached with per-tile normalisation
    import glob
    for pat in ("reassigned_tile_*.png", "flat_reassigned_tile_*.png"):
        for p in glob.glob(os.path.join(entry.tile_dir, pat)):
            try:
                os.remove(p)
            except Exception:
                pass
    with entry.reassigned_tile_lock:
        entry.reassigned_tile_cache.clear()
    with entry.flat_reassigned_tile_lock:
        entry.flat_reassigned_tile_cache.clear()


def _encode_reassigned(arr, entry, disk_path, tile_lock, tile_cache, tidx):
    """Convert a normalised [0,1] array to PNG, cache it, and return bytes."""
    rgb  = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)
    pil  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()
    try:
        os.makedirs(entry.tile_dir, exist_ok=True)
        with open(disk_path, "wb") as fh:
            fh.write(data)
    except Exception:
        pass
    with tile_lock:
        tile_cache[tidx] = data
    return data


def make_reassigned_tile(entry, tidx):
    """Reassigned-STFT tile — energy mapped to instantaneous frequency.

    Global normalisation: peak call bin → 0 dB ceiling, 40 dB dynamic range.
    """
    with entry.reassigned_tile_lock:
        if tidx in entry.reassigned_tile_cache:
            return entry.reassigned_tile_cache[tidx]

    disk_path = os.path.join(entry.tile_dir, f"reassigned_tile_{tidx:04d}.png")
    if os.path.exists(disk_path):
        with open(disk_path, "rb") as fh:
            data = fh.read()
        with entry.reassigned_tile_lock:
            entry.reassigned_tile_cache[tidx] = data
        return data

    _init_reassigned_norm(entry)
    out, _ = _reassigned_scatter(entry, tidx)

    # File-wide normalisation anchored to the STFT noise floor (entry.vmin) as
    # the black point and the pre-computed scatter ceiling as the white point.
    # This ensures calls appear at full brightness (concentrated energy → high
    # scatter → near ceiling) while silent bins stay black, with no tile seams.
    ceiling = entry.reass_norm_max if (entry.reass_norm_max and entry.reass_norm_max > 1e-30) else 1.0
    ceiling_db = 10.0 * np.log10(ceiling)
    floor_db   = float(entry.vmin)          # STFT noise floor, e.g. −120 dB
    range_db   = max(ceiling_db - floor_db, 1.0)
    Sdb = 10.0 * np.log10(out + 1e-60)
    arr = np.clip((Sdb - floor_db) / range_db, 0.0, 1.0)

    return _encode_reassigned(arr, entry, disk_path,
                              entry.reassigned_tile_lock,
                              entry.reassigned_tile_cache, tidx)


def make_flat_reassigned_tile(entry, tidx):
    """Per-frequency-normalised reassigned spectrogram tile.

    Same scatter computation as make_reassigned_tile, but each frequency row
    is normalised by its own peak so all bat-call frequencies appear at
    comparable brightness regardless of mic response or species preference.
    Frequency rows with no significant signal (< 0.1 % of the global peak)
    keep the global normalisation and therefore appear dark/black.
    """
    with entry.flat_reassigned_tile_lock:
        if tidx in entry.flat_reassigned_tile_cache:
            return entry.flat_reassigned_tile_cache[tidx]

    disk_path = os.path.join(entry.tile_dir, f"flat_reassigned_tile_{tidx:04d}.png")
    if os.path.exists(disk_path):
        with open(disk_path, "rb") as fh:
            data = fh.read()
        with entry.flat_reassigned_tile_lock:
            entry.flat_reassigned_tile_cache[tidx] = data
        return data

    _init_reassigned_norm(entry)
    out, _ = _reassigned_scatter(entry, tidx)

    # File-wide per-frequency normalisation anchored to the STFT per-frequency
    # noise floor (entry.vmin_f) as the black point and the pre-computed
    # scatter per-frequency ceiling as the white point.
    n_freq = out.shape[0]
    if (entry.reass_norm_max_f is not None
            and len(entry.reass_norm_max_f) == n_freq):
        ceil_f_db = 10.0 * np.log10(entry.reass_norm_max_f + 1e-60)   # (n_freq,) dB
    else:
        ceiling = entry.reass_norm_max if (entry.reass_norm_max
                                           and entry.reass_norm_max > 1e-30) else 1.0
        ceil_f_db = np.full(n_freq, 10.0 * np.log10(ceiling), dtype=np.float64)

    if entry.vmin_f is not None and len(entry.vmin_f) == n_freq:
        floor_f_db = entry.vmin_f.astype(np.float64)                   # (n_freq,) dB
    else:
        floor_f_db = np.full(n_freq, float(entry.vmin), dtype=np.float64)

    range_f_db = np.maximum(ceil_f_db - floor_f_db, 1.0)              # (n_freq,) dB
    Sdb = 10.0 * np.log10(out + 1e-60)                                 # (n_freq, n_time)
    arr = np.clip((Sdb - floor_f_db[:, None]) / range_f_db[:, None], 0.0, 1.0)

    return _encode_reassigned(arr, entry, disk_path,
                              entry.flat_reassigned_tile_lock,
                              entry.flat_reassigned_tile_cache, tidx)


def make_flat_tile(entry, tidx):
    """Per-frequency-normalised spectrogram tile (the "Flat" view).

    For each frequency bin k, maps:
        arr[k, t] = clip( (Sdb[k,t] − lo[k]) / (hi[k] − lo[k]),  0, 1 )

    Uses globally pre-computed per-bin stats (entry.vmin_f / entry.vmax_f) so
    tile boundaries are seamless.
    """
    with entry.flat_tile_lock:
        if tidx in entry.flat_tile_cache:
            return entry.flat_tile_cache[tidx]

    disk_path = os.path.join(entry.tile_dir, f"flat_tile_{tidx:04d}.png")
    if os.path.exists(disk_path):
        with open(disk_path, "rb") as fh:
            data = fh.read()
        with entry.flat_tile_lock:
            entry.flat_tile_cache[tidx] = data
        return data

    sr  = entry.finfo["sr"]
    dur = entry.finfo["duration_s"]
    t0  = tidx * TILE_DURATION
    t1  = min(t0 + TILE_DURATION, dur)
    f0  = int(t0 * sr); f1 = int(t1 * sr)

    with entry.audio_lock:
        entry.audio_fh.seek(f0)
        audio = entry.audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f_s, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm  = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    Sdb = 10 * np.log10(Sxx[bm, :] + 1e-12)

    n_freq = Sdb.shape[0]
    if (entry.vmin_f is not None and entry.vmax_f is not None
            and len(entry.vmin_f) == n_freq):
        lo = entry.vmin_f[:, np.newaxis]
        hi = entry.vmax_f[:, np.newaxis]
    else:
        lo = np.percentile(Sdb, 2.0,  axis=1, keepdims=True)
        hi = np.percentile(Sdb, 99.9, axis=1, keepdims=True)
    arr = np.clip((Sdb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)

    pil  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    try:
        os.makedirs(entry.tile_dir, exist_ok=True)
        with open(disk_path, "wb") as fh:
            fh.write(data)
    except Exception:
        pass

    with entry.flat_tile_lock:
        entry.flat_tile_cache[tidx] = data
    return data
