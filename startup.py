import json, os, re, subprocess, threading
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf

import config
import state
import registry
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

        raw_path  = Path(path).with_suffix('.f32raw')
        meta_path = Path(path).with_suffix('.f32meta')

        # ── Metadata: sidecar JSON (no ffprobe) or ffprobe (first use) ──
        # The sidecar is written the first time ffprobe is run so that servers
        # without ffprobe installed can open the file after the .f32raw and
        # .f32meta have been rsynced from the machine that created them.
        _ffprobe_streams = None
        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    m = json.load(fh)
                self.samplerate = int(m['sample_rate'])
                self.channels   = int(m['channels'])
                self.subtype    = m.get('subtype', 'PCM_32')
            except Exception as exc:
                raise RuntimeError(f"Bad .f32meta ({exc}); delete it to regenerate") from exc
        else:
            r = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_streams', '-select_streams', 'a', path],
                capture_output=True, text=True, check=True)
            _ffprobe_streams = json.loads(r.stdout)['streams'][0]
            self.samplerate = int(_ffprobe_streams['sample_rate'])
            self.channels   = int(_ffprobe_streams['channels'])

            bpr = int(_ffprobe_streams.get('bits_per_raw_sample', 0))
            fmt = _ffprobe_streams.get('sample_fmt', '')
            if fmt in ('fltp', 'flt'):
                self.subtype = 'FLOAT'
            elif fmt in ('dblp', 'dbl'):
                self.subtype = 'DOUBLE'
            elif bpr:
                self.subtype = f'PCM_{bpr}'
            else:
                self.subtype = 'PCM_32'

        # ── Decode to raw float32 cache (once) ───────────────────
        if not raw_path.exists():
            dur_s = float(_ffprobe_streams.get('duration', 0)) if _ffprobe_streams else 0.0
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

        # ── Write sidecar so future opens (e.g. on a server without ffprobe)
        #    don't need to call ffprobe again. ────────────────────────────────
        if not meta_path.exists():
            try:
                with open(meta_path, 'w') as fh:
                    json.dump({'sample_rate': self.samplerate,
                               'channels':    self.channels,
                               'subtype':     self.subtype}, fh)
            except Exception as exc:
                print(f"  Warning: could not write {meta_path.name}: {exc}")

        # ── Memory-map the raw file ───────────────────────────────
        size        = raw_path.stat().st_size
        self.frames = size // (self.channels * 4)
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
        Discard any contour point below 20 kHz.

    Pass 2 — Harmonic separation
        Look for a large frequency gap (> 7 kHz) that splits the remaining
        points into a lower cluster and an upper cluster, with a frequency
        ratio ≥ 1.55× (consistent with a harmonic relationship).

    Pass 3 — Bounding box
        Recompute Fmin / Fmax from the cleaned contour points.
    """
    cnt = c.get("contour")
    if not cnt or len(cnt) < 2:
        return
    cnt.sort(key=lambda pt: pt[0])
    freqs = np.array([pt[1] for pt in cnt])

    # ── Pass 1: floor filter ─────────────────────────────────────
    mask          = freqs >= 20.0
    trimmed_pts   = [pt for pt, m in zip(cnt, mask) if m]
    trimmed_freqs = freqs[mask]

    if len(trimmed_pts) == 0:
        return
    if len(trimmed_pts) == 1:
        f = float(trimmed_freqs[0])
        c["contour"] = [[cnt[0][0], f], [cnt[-1][0], f]]
        c["Fmin"]    = round(f - 2.0, 2)
        c["Fmax"]    = round(f + 2.0, 2)
        return

    # ── Pass 2: harmonic separation ──────────────────────────────
    HARMONIC_GAP_KHZ  = 7.0
    HARMONIC_RATIO    = 1.55
    MIN_FUND_FRACTION = 0.15

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
        min_fund_pts  = max(2, MIN_FUND_FRACTION * n_total)
        if ratio >= HARMONIC_RATIO and len(lower_cluster) >= min_fund_pts:
            fund_ceil     = float(lower_cluster.max()) + 2.0
            keep          = trimmed_freqs <= fund_ceil
            trimmed_pts   = [pt for pt, k in zip(trimmed_pts, keep) if k]
            trimmed_freqs = trimmed_freqs[keep]
        elif ratio >= HARMONIC_RATIO and len(lower_cluster) < min_fund_pts:
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

    c["bw"] = round(new_hi - new_lo, 2)

    med_f   = float(np.median(trimmed_freqs))
    c["cf_frac"] = round(float(np.mean(np.abs(trimmed_freqs - med_f) <= 2.0)), 3)


def reclassify_calls(calls):
    """Run both classifiers on every call in-place using the current PROFILES."""
    counts_v1, counts_v2 = {}, {}
    short_map = {p["name"]: p["short"] for p in PROFILES}
    for c in calls:
        sp1, cf1 = classify_v1(c)
        sp2, cf2 = classify_v2(c)
        c["species_v1"] = sp1
        c["conf_v1"]    = cf1
        c["color_v1"]   = COLORS.get(sp1, "#888888")
        c["short_v1"]   = short_map.get(sp1, "????")
        counts_v1[sp1]  = counts_v1.get(sp1, 0) + 1
        c["species"] = sp2
        c["conf"]    = cf2
        c["color"]   = COLORS.get(sp2, "#888888")
        c["short"]   = short_map.get(sp2, "????")
        counts_v2[sp2]  = counts_v2.get(sp2, 0) + 1

    def _summary(counts):
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    print(f"reclassify v1: {_summary(counts_v1)}")
    print(f"reclassify v2: {_summary(counts_v2)}")


def try_load_cache(entry):
    """Return True if valid cached results were loaded into entry."""
    if not os.path.exists(entry.cache_file):
        return False
    try:
        with open(entry.cache_file) as fh:
            cache = json.load(fh)
        from config import BD2_THRESH
        if cache.get("bd2_thresh") != BD2_THRESH:
            print(f"Cache stale (BD2_THRESH changed) for {entry.name} — re-detecting.")
            return False
        entry.all_calls.extend(cache["calls"])
        det = cache.get("detector", "cached")
        for c in entry.all_calls:
            trim_call_contour(c)
        reclassify_calls(entry.all_calls)
        entry.detection_progress["status"] = (
            f"Loaded from cache — {len(entry.all_calls)} calls  [{det}]")
        entry.calls_ready.set()
        print(entry.detection_progress["status"])
        return True
    except Exception as exc:
        print(f"Cache load failed for {entry.name} ({exc}) — re-detecting.")
        return False


def _parse_recording_start(filename):
    """Parse start timestamp from filenames like '2025-05-28 1942 …' or
    '2025-06-06-1912-bats-192khz.wv' (hyphen separator after date).
    Returns ISO-8601 string or None."""
    m = re.search(r'(\d{4}-\d{2}-\d{2})[-\s](\d{4})', os.path.basename(filename))
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M").isoformat()
    except ValueError:
        return None


def _bit_depth(subtype: str) -> str:
    return {"PCM_8": "8-bit", "PCM_16": "16-bit", "PCM_24": "24-bit",
            "PCM_32": "32-bit", "FLOAT": "32-bit float", "DOUBLE": "64-bit float"}.get(subtype, subtype)


def _parse_location(path: str):
    """Look up a human-readable recording location by audio file stem."""
    stem = Path(path).stem
    return config.LOCATION_MAP.get(stem)


# ─────────────────────────────────────────────
# Per-entry loading
# ─────────────────────────────────────────────

def _load_entry(entry, redetect=False):
    """Open audio, compute norms, then load calls from cache or start detection."""
    print(f"Opening {entry.name} …")
    entry.audio_fh = _open_audio(entry.path)
    entry.finfo.update({
        "sr":              entry.audio_fh.samplerate,
        "nframes":         entry.audio_fh.frames,
        "channels":        entry.audio_fh.channels,
        "duration_s":      entry.audio_fh.frames / entry.audio_fh.samplerate,
        "bit_depth":       _bit_depth(entry.audio_fh.subtype),
        "recording_start": _parse_recording_start(entry.path),
        "location":        _parse_location(entry.path),
    })
    print(f"  {entry.finfo['duration_s']:.1f} s  ·  {entry.finfo['sr']:,} Hz  ·  "
          f"{entry.finfo['channels']} ch")

    os.makedirs(entry.tile_dir, exist_ok=True)

    from tiles import _init_tile_norm, _pregenerate_mask_tiles
    _init_tile_norm(entry)

    # Try loading BatDetect2 cache (synchronous in this thread)
    bd2_cached = not redetect and try_load_cache(entry)
    if bd2_cached:
        threading.Thread(target=_pregenerate_mask_tiles, args=(entry,), daemon=True).start()
    else:
        entry.detection_progress.update({"done": 0, "total": 1, "status": "Detection starting…"})
        from detect import run_detection
        threading.Thread(target=run_detection, args=(entry,), daemon=True).start()

    # Try loading Tadarida-D cache in background (non-blocking)
    try:
        from detect_tadarida import try_load_tadarida_cache
        threading.Thread(target=try_load_tadarida_cache, args=(entry,), daemon=True).start()
    except Exception as exc:
        print(f"  [startup] Tadarida cache check failed: {exc}")


# ─────────────────────────────────────────────
# Main startup
# ─────────────────────────────────────────────

def startup(redetect=False):
    """Scan the audio directory, register all files, and start the scheduler."""
    audio_dir   = Path(os.path.abspath(config.AUDIO_FILE)).parent
    active_path = os.path.abspath(config.AUDIO_FILE)
    exts = {'.flac', '.wav', '.wv', '.mp3', '.ogg', '.aif', '.aiff'}

    # ── Register and fully load the default file (synchronous) ──────
    default_entry = registry.register(active_path)
    registry.set_default(active_path)
    _load_entry(default_entry, redetect=redetect)

    # ── Tile scheduler (created once; reused forever) ────────────────
    if state.scheduler is None:
        from tile_scheduler import TileScheduler
        state.scheduler = TileScheduler()
        state.scheduler.start()
    state.scheduler.register_file(active_path)
    state.scheduler.set_active(active_path)

    # ── Register other audio files in background ──────────────────────
    def _bg():
        for p in sorted(audio_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                pstr = str(p)
                if pstr == active_path:
                    continue
                try:
                    e = registry.register(pstr)
                    _load_entry(e)            # opens audio, norms, cache/detection
                    state.scheduler.register_file(pstr)
                except Exception as exc:
                    print(f"  [startup] skip {Path(pstr).name}: {exc}")

    threading.Thread(target=_bg, daemon=True).start()
