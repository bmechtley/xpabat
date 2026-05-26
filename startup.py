import json, os, re, threading
from datetime import datetime
import numpy as np
import soundfile as sf

import config
from state import (
    audio_lock, finfo, all_calls, calls_ready, progress,
)
import state
from classify import classify_v1, classify_v2
from species import PROFILES, COLORS


def trim_call_contour(c):
    """Clean a call's frequency contour and tighten Fmin/Fmax.

    Three passes:

    Pass 1 — Floor filter
        Discard any contour point below 20 kHz (below which no western-NA bat
        species emits echolocation energy in this recording).

    Pass 2 — Harmonic separation
        Look for a large frequency gap (> 10 kHz) that splits the remaining
        points into a lower cluster (fundamental) and an upper cluster
        (harmonic).  If such a gap exists AND the lower cluster accounts for
        ≥ 15% of the post-floor points, keep only the lower cluster.

        Rationale: the raw argmax often jumps to the first harmonic (at ~2×
        the fundamental) when the harmonic is momentarily louder.  This is
        common in LACI calls where the harmonic can dominate.  The gap between
        fundamental and harmonic is always ≥ one fundamental-width (typically
        15–40 kHz) and is reliably larger than any legitimate intra-call FM
        sweep seen in this dataset.

    Pass 3 — Bounding box
        Recompute Fmin / Fmax from the cleaned contour points.
    """
    cnt = c.get("contour")
    if not cnt or len(cnt) < 2:
        return
    cnt.sort(key=lambda pt: pt[0])   # guarantee time-monotone order (in-place)
    freqs = np.array([pt[1] for pt in cnt])

    # ── Pass 1: floor filter ─────────────────────────────────────
    mask    = freqs >= 20.0
    trimmed_pts   = [pt for pt, m in zip(cnt, mask) if m]
    trimmed_freqs = freqs[mask]

    if len(trimmed_pts) == 0:
        return   # all floor-noise; leave untouched
    if len(trimmed_pts) == 1:
        f = float(trimmed_freqs[0])
        c["contour"] = [[cnt[0][0], f], [cnt[-1][0], f]]
        c["Fmin"]    = round(f - 2.0, 2)
        c["Fmax"]    = round(f + 2.0, 2)
        return

    # ── Pass 2: harmonic separation ──────────────────────────────
    # Criteria for a true fundamental/harmonic split:
    #   (a) gap ≥ 7 kHz between the two clusters, AND
    #   (b) upper-cluster centre / lower-cluster centre ≥ 1.55
    #       (≈ 2× fundamental; real harmonics land at 1.8–2.2×, FM-sweep
    #        intra-call components land at 1.2–1.4× and must NOT be split)
    HARMONIC_GAP_KHZ  = 7.0    # minimum gap that might indicate a harmonic split
    HARMONIC_RATIO    = 1.55   # minimum freq-ratio upper/lower to confirm harmonic
    MIN_FUND_FRACTION = 0.15   # lower cluster must have ≥ 15% of points to keep it

    sorted_f = np.sort(trimmed_freqs)
    gaps     = np.diff(sorted_f)
    if gaps.max() >= HARMONIC_GAP_KHZ:
        split_idx     = int(gaps.argmax())
        lower_cluster = sorted_f[:split_idx + 1]
        upper_cluster = sorted_f[split_idx + 1:]
        n_total       = len(sorted_f)
        lower_centre  = float(np.median(lower_cluster))
        upper_centre  = float(np.median(upper_cluster))
        ratio         = upper_centre / lower_centre if lower_centre > 0 else 0.0
        if (ratio >= HARMONIC_RATIO
                and len(lower_cluster) >= max(2, MIN_FUND_FRACTION * n_total)):
            # Keep only points in the lower (fundamental) cluster
            fund_ceil     = float(lower_cluster.max()) + 2.0
            keep          = trimmed_freqs <= fund_ceil
            trimmed_pts   = [pt for pt, k in zip(trimmed_pts, keep) if k]
            trimmed_freqs = trimmed_freqs[keep]

    if len(trimmed_pts) == 0:
        return
    if len(trimmed_pts) == 1:
        f = float(trimmed_freqs[0])
        c["contour"] = [[cnt[0][0], f], [cnt[-1][0], f]]
        c["Fmin"]    = round(f - 2.0, 2)
        c["Fmax"]    = round(f + 2.0, 2)
        return

    # ── Pass 3: update contour, bounding box, and derived features ──
    c["contour"] = trimmed_pts
    new_lo = float(trimmed_freqs.min())
    new_hi = float(trimmed_freqs.max())
    if new_hi - new_lo < 1.0:
        pad     = (1.0 - (new_hi - new_lo)) / 2
        new_lo -= pad;  new_hi += pad
    c["Fmin"] = round(new_lo, 2)
    c["Fmax"] = round(new_hi, 2)

    # Bandwidth: total frequency span of the cleaned contour (kHz)
    c["bw"] = round(new_hi - new_lo, 2)

    # CF fraction: proportion of contour frames within 2 kHz of the median
    # frequency.  High → mostly constant-frequency (CF); low → steep FM sweep.
    med_f   = float(np.median(trimmed_freqs))
    c["cf_frac"] = round(float(np.mean(np.abs(trimmed_freqs - med_f) <= 2.0)), 3)



def reclassify_calls(calls):
    """Run both classifiers on every call in-place using the current PROFILES.

    Stores v1 (4-criterion) and v2 (6-criterion with bw+cf_frac) results.
    c["species"] / c["color"] / c["short"] / c["conf"] always reflect v2 so
    the rest of the code works unchanged.  v1 results live in c["species_v1"]
    etc. for the UI comparison toggle.
    """
    counts_v1, counts_v2 = {}, {}
    short_map = {p["name"]: p["short"] for p in PROFILES}
    for c in calls:
        sp1, cf1 = classify_v1(c)
        sp2, cf2 = classify_v2(c)
        # v1
        c["species_v1"] = sp1
        c["conf_v1"]    = cf1
        c["color_v1"]   = COLORS.get(sp1, "#888888")
        c["short_v1"]   = short_map.get(sp1, "????")
        counts_v1[sp1]  = counts_v1.get(sp1, 0) + 1
        # v2 (default / active)
        c["species"] = sp2
        c["conf"]    = cf2
        c["color"]   = COLORS.get(sp2, "#888888")
        c["short"]   = short_map.get(sp2, "????")
        counts_v2[sp2]  = counts_v2.get(sp2, 0) + 1

    def _summary(counts):
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    print(f"reclassify v1: {_summary(counts_v1)}")
    print(f"reclassify v2: {_summary(counts_v2)}")


def try_load_cache():
    """Return True if valid cached results were loaded, False if detection must run."""
    global all_calls
    if not os.path.exists(config.CACHE_FILE):
        return False
    try:
        with open(config.CACHE_FILE) as fh:
            cache = json.load(fh)
        # Invalidate if detection threshold changed
        from config import BD2_THRESH
        if cache.get("bd2_thresh") != BD2_THRESH:
            print("Cache is stale (BD2_THRESH changed) — re-detecting.")
            return False
        all_calls.extend(cache["calls"])
        det = cache.get("detector", "cached")
        # Trim contour outliers in case the cache pre-dates the freq-gating fix
        for c in all_calls:
            trim_call_contour(c)
        # Re-run classifier so profile changes / priors take effect without re-detecting
        reclassify_calls(all_calls)
        progress["status"] = f"Loaded from cache — {len(all_calls)} calls  [{det}]"
        calls_ready.set()
        print(progress["status"])
        return True
    except Exception as exc:
        print(f"Cache load failed ({exc}) — re-detecting.")
        return False

def _parse_recording_start(filename):
    """Parse start timestamp from filenames like '2025-05-28 1942 …'.
    Returns ISO-8601 string or None."""
    m = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{4})', os.path.basename(filename))
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M").isoformat()
    except ValueError:
        return None


def _bit_depth(subtype: str) -> str:
    """Map soundfile subtype string to a human-readable bit-depth label."""
    return {"PCM_8": "8-bit", "PCM_16": "16-bit", "PCM_24": "24-bit",
            "PCM_32": "32-bit", "FLOAT": "32f", "DOUBLE": "64f"}.get(subtype, subtype)


def startup(redetect=False):
    print(f"Opening {config.AUDIO_FILE} …")
    state.audio_fh = sf.SoundFile(config.AUDIO_FILE)
    finfo.update({
        "sr":              state.audio_fh.samplerate,
        "nframes":         state.audio_fh.frames,
        "channels":        state.audio_fh.channels,
        "duration_s":      state.audio_fh.frames / state.audio_fh.samplerate,
        "bit_depth":       _bit_depth(state.audio_fh.subtype),
        "recording_start": _parse_recording_start(config.AUDIO_FILE),
    })
    print(f"  {finfo['duration_s']:.1f} s  ·  {finfo['sr']:,} Hz  ·  {finfo['channels']} ch")

    state.TILE_DIR = os.path.splitext(config.AUDIO_FILE)[0] + "_tiles"
    os.makedirs(state.TILE_DIR, exist_ok=True)
    print(f"  Tile cache → {state.TILE_DIR}")

    # Compute (or load) global spectrogram normalisation before any tiles are made
    from tiles import _init_tile_norm, _pregenerate_tiles
    _init_tile_norm()

    if not redetect and try_load_cache():
        # Detection loaded from cache — pre-generate tiles right away
        threading.Thread(target=_pregenerate_tiles, daemon=True).start()
        return

    progress["status"] = "Detection starting…"
    from detect import run_detection
    t = threading.Thread(target=run_detection, daemon=True)
    t.start()
