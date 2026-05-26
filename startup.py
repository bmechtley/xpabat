import json, os, re, subprocess, threading
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf

import config
from state import (
    audio_lock, finfo, all_calls, calls_ready, progress,
)
import state
from classify import classify_v1, classify_v2
from species import PROFILES, COLORS


# ─────────────────────────────────────────────
# WavPack / unsupported-format reader
# ─────────────────────────────────────────────

class _WavPackFile:
    """Read WavPack (or any ffmpeg-decodable) audio via a memory-mapped raw cache.

    On first open the file is decoded by ffmpeg to a companion `.f32raw` file
    (interleaved float32, same sample rate and channel count).  Subsequent
    opens mmap that file directly, so seeks and reads are near-instant.

    The interface mirrors the parts of soundfile.SoundFile that the app uses:
      .samplerate  .frames  .channels  .subtype
      .seek(n)  .read(n_frames, dtype, always_2d)  .close()
    """

    def __init__(self, path: str):
        path = str(path)
        self.name = path

        # ── Metadata via ffprobe ──────────────────────────────────
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-select_streams', 'a', path],
            capture_output=True, text=True, check=True)
        s = json.loads(r.stdout)['streams'][0]
        self.samplerate = int(s['sample_rate'])
        self.channels   = int(s['channels'])

        # Map ffprobe sample_fmt → subtype string (used by _bit_depth())
        bpr = int(s.get('bits_per_raw_sample', 0))
        fmt = s.get('sample_fmt', '')
        if fmt in ('fltp', 'flt'):
            self.subtype = 'FLOAT'
        elif fmt in ('dblp', 'dbl'):
            self.subtype = 'DOUBLE'
        elif bpr:
            self.subtype = f'PCM_{bpr}'
        else:
            self.subtype = 'PCM_32'

        # ── Decode to raw float32 cache (once) ───────────────────
        raw_path = Path(path).with_suffix('.f32raw')
        if not raw_path.exists():
            dur_s = float(s.get('duration', 0))
            print(f"  Decoding to {raw_path.name}  "
                  f"({dur_s:.0f} s · {self.samplerate//1000} kHz · "
                  f"{self.channels} ch) — this runs once…", flush=True)
            subprocess.run(
                ['ffmpeg', '-v', 'quiet', '-i', path,
                 '-f', 'f32le',
                 '-ar', str(self.samplerate),
                 '-ac', str(self.channels),
                 str(raw_path)],
                check=True)
            print(f"  Decode done → {raw_path.stat().st_size / 1e9:.2f} GB", flush=True)

        # ── Memory-map the raw file ───────────────────────────────
        size        = raw_path.stat().st_size
        self.frames = size // (self.channels * 4)   # 4 bytes per float32 sample
        self._mmap  = np.memmap(str(raw_path), dtype='float32', mode='r',
                                shape=(self.frames, self.channels))
        self._pos   = 0

    def seek(self, pos: int) -> None:
        self._pos = int(pos)

    def read(self, n_frames: int, dtype: str = 'float32',
             always_2d: bool = True) -> np.ndarray:
        start = self._pos
        end   = min(start + n_frames, self.frames)
        out   = np.array(self._mmap[start:end], dtype=dtype)
        self._pos = end
        if self.channels == 1 and not always_2d:
            return out.ravel()
        return out

    def close(self) -> None:
        del self._mmap


def _open_audio(path: str):
    """Open an audio file for reading.  Returns a soundfile.SoundFile when the
    format is supported, or a _WavPackFile wrapper for WavPack / other formats
    that soundfile's libsndfile cannot read."""
    try:
        return sf.SoundFile(path)
    except Exception:
        return _WavPackFile(path)


def trim_call_contour(c):
    """Clean a call's frequency contour and tighten Fmin/Fmax.

    Three passes:

    Pass 1 — Floor filter
        Discard any contour point below 20 kHz (below which no western-NA bat
        species emits echolocation energy in this recording).

    Pass 2 — Harmonic separation
        Look for a large frequency gap (> 7 kHz) that splits the remaining
        points into a lower cluster and an upper cluster, with a frequency
        ratio ≥ 1.55× (consistent with a harmonic relationship).  Two cases:

        (a) Large lower cluster (≥ 15% of post-floor points, ≥ 2 points):
            The lower cluster is a real fundamental; the tracker occasionally
            jumped to the harmonic.  Keep the lower cluster, drop the upper.
            Common in LACI calls where the harmonic is momentarily louder.

        (b) Tiny lower cluster (< 15% or < 2 points):
            The lower points are isolated noise frames or an overlapping call
            at a different frequency.  The real signal is in the upper cluster.
            Keep the upper cluster, drop the lower outliers.

        The gap between fundamental and harmonic is always ≥ one fundamental-
        width (typically 15–40 kHz) and is reliably larger than any legitimate
        intra-call FM sweep seen in this dataset.

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
        min_fund_pts = max(2, MIN_FUND_FRACTION * n_total)
        if ratio >= HARMONIC_RATIO and len(lower_cluster) >= min_fund_pts:
            # Strong fundamental: lower cluster is large enough to be real.
            # Keep only points in the lower (fundamental) cluster; drop the harmonic.
            fund_ceil     = float(lower_cluster.max()) + 2.0
            keep          = trimmed_freqs <= fund_ceil
            trimmed_pts   = [pt for pt, k in zip(trimmed_pts, keep) if k]
            trimmed_freqs = trimmed_freqs[keep]
        elif ratio >= HARMONIC_RATIO and len(lower_cluster) < min_fund_pts:
            # Tiny lower cluster: too few points to be a real fundamental.
            # These are isolated noise/outlier frames that happened to land at a
            # low frequency while the real call is in the upper cluster.
            # Remove the outlier points and keep the upper (real-signal) cluster.
            upper_floor   = float(upper_cluster.min()) - 2.0
            keep          = trimmed_freqs >= upper_floor
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
    state.audio_fh = _open_audio(config.AUDIO_FILE)
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


def reset_and_switch(new_path: str, redetect: bool = False) -> None:
    """Reset all server state and load a new audio file in-place.

    Safe to call from any thread.  Any in-progress detection is signalled
    to stop before the state is cleared.
    """
    print(f"\nSwitching → {Path(new_path).name}", flush=True)

    # ── 1. Signal detection thread to abort ──────────────────────
    state._stop_detection.set()

    # ── 2. Close old audio file (holds audio_lock briefly) ───────
    with state.audio_lock:
        if state.audio_fh is not None:
            try:
                state.audio_fh.close()
            except Exception:
                pass
            state.audio_fh = None

    # ── 3. Clear all mutable state ───────────────────────────────
    state.all_calls.clear()
    state.calls_ready.clear()
    state.finfo.clear()
    state.progress.update({"done": 0, "total": 1, "status": "Switching file…"})

    with state.tile_lock:
        state.tile_cache.clear()
    with state.mask_tile_lock:
        state.mask_tile_cache.clear()
    with state.flat_tile_lock:
        state.flat_tile_cache.clear()

    state._global_vmin   = -100.0
    state._global_vmax   =  -30.0
    state._global_vmin_f = None
    state._global_vmax_f = None

    # ── 4. Update config and clear the stop signal ───────────────
    config.AUDIO_FILE = new_path
    config.CACHE_FILE = os.path.splitext(new_path)[0] + ".calls.json"
    state._stop_detection.clear()

    # ── 5. Re-run startup with the new file ──────────────────────
    startup(redetect=redetect)
