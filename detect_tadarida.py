"""
Tadarida-D detection backend.

Wraps pytadarida (https://github.com/mbsantiago/pytadarida) to produce call
dicts in the same format as detect.py (BatDetect2 path).

Tadarida-D is a Linux-only C++ binary bundled inside the pytadarida package.
It runs detection-only (no classification); our own v1/v2 classifiers are
applied afterwards by startup.reclassify_calls().

Chunking: Tadarida-D rejects files longer than ~12.8 s in HF mode, so we
split the audio into CHUNK_S-second temporary WAV files, run the detector on
each, then reassemble with corrected absolute timestamps.
"""

import os, sys, tempfile, threading, time, json
import numpy as np

# Chunk size (< 12.8 s Tadarida-D HF limit, padded a bit for safety)
_CHUNK_S = 10.0

# How much Tadarida overlap to add (helps at chunk boundaries)
_OVERLAP_S = 0.5

# Minimum call duration seconds (matches config.MIN_CALL)
_MIN_CALL = 0.001
_MAX_CALL = 0.200


def _tadarida_available():
    """Return True iff pytadarida is importable (Linux only)."""
    if sys.platform != "linux":
        return False
    try:
        import pytadarida   # noqa: F401
        return True
    except ImportError:
        return False


TADARIDA_AVAILABLE = _tadarida_available()


def run_tadarida_detection(entry):
    """
    Run Tadarida-D on entry's audio file, populate entry calls, and cache.

    Called in a background thread (same pattern as detect.run_detection).
    Results are stored in entry.calls_by_detector["tadarida"] (a separate
    list from the BatDetect2 slot) and persisted to
    <stem>.tadarida.calls.json.
    """
    import threading as _threading
    from scipy import signal as _signal
    from startup import reclassify_calls, trim_call_contour

    det_key   = "tadarida"
    cache_path = os.path.splitext(entry.path)[0] + ".tadarida.calls.json"

    # Ensure per-detector slots exist
    if det_key not in entry.calls_by_detector:
        entry.calls_by_detector[det_key]    = []
        entry.ready_by_detector[det_key]    = _threading.Event()
        entry.progress_by_detector[det_key] = {"done": 0, "total": 1,
                                                "status": "starting"}

    calls_list = entry.calls_by_detector[det_key]
    ready_ev   = entry.ready_by_detector[det_key]
    progress   = entry.progress_by_detector[det_key]

    progress.update({"done": 0, "total": 1, "status": "Loading Tadarida-D…"})

    if not TADARIDA_AVAILABLE:
        progress["status"] = "Tadarida-D unavailable (Linux only / pytadarida not installed)"
        ready_ev.set()
        print(f"[Tadarida] Not available on this platform ({sys.platform})")
        return

    try:
        from pytadarida import run_tadarida as _run_tadarida
    except ImportError as exc:
        progress["status"] = f"pytadarida import failed: {exc}"
        ready_ev.set()
        return

    sr    = entry.finfo["sr"]
    nf    = entry.finfo["nframes"]
    dur_s = nf / sr

    chunk_frames   = int(_CHUNK_S    * sr)
    overlap_frames = int(_OVERLAP_S  * sr)
    total_ch       = int(np.ceil(nf / chunk_frames))
    progress.update({"done": 0, "total": total_ch,
                     "status": f"Detecting (Tadarida-D)… 0/{total_ch}"})

    print(f"\n[Tadarida] Detection starting  {entry.name}")
    print(f"  {dur_s:.1f} s  ·  {sr:,} Hz  ·  {total_ch} chunks ({_CHUNK_S:.0f} s each)")
    t_start = time.time()

    raw = []
    from config import A_NPERSEG, A_NOVERLAP, FREQ_LOW, FREQ_HIGH
    from detect import track_fundamental

    with tempfile.TemporaryDirectory(prefix="tadarida_") as tmpdir:
        offset = 0
        for ch_idx in range(total_ch):
            if entry.stop_event.is_set():
                print("[Tadarida] Aborted.")
                progress["status"] = "Aborted"
                return

            end = min(nf, offset + chunk_frames + overlap_frames)
            with entry.audio_lock:
                entry.audio_fh.seek(offset)
                audio = entry.audio_fh.read(end - offset, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            chunk_t0_s = offset / sr

            # Write temp WAV
            wav_path = os.path.join(tmpdir, f"chunk_{ch_idx:04d}.wav")
            _write_wav(wav_path, mono, sr)

            # Run Tadarida-D
            try:
                df, _status = _run_tadarida(
                    [wav_path],
                    threads=1,
                    time_expansion=1,
                    frequency_band=1,   # 1 = HF (8–250 kHz); 2 = LF (0.8–25 kHz)
                )
            except FileNotFoundError:
                # Tadarida found no detections in this chunk — normal, just skip.
                offset += chunk_frames
                ch_idx_done = ch_idx + 1
                progress.update({"done": ch_idx_done,
                                  "status": f"Detecting (Tadarida-D)… {ch_idx_done}/{total_ch}"})
                continue
            except Exception as exc:
                err_short = str(exc).split("\n")[0][:120]
                print(f"  [Tadarida] chunk {ch_idx} FAILED: {err_short}")
                offset += chunk_frames
                ch_idx_done = ch_idx + 1
                progress.update({"done": ch_idx_done,
                                  "status": f"Error chunk {ch_idx_done}: {err_short}"})
                continue

            if df is None or df.empty:
                offset += chunk_frames
                ch_idx_done = ch_idx + 1
                progress.update({"done": ch_idx_done,
                                  "status": f"Detecting (Tadarida-D)… {ch_idx_done}/{total_ch}"})
                continue

            # Compute spectrogram of this chunk for contour tracking
            f_arr, t_arr, Sxx = _signal.spectrogram(
                mono, fs=sr, nperseg=A_NPERSEG, noverlap=A_NOVERLAP, window="hann")
            bm = (f_arr >= FREQ_LOW) & (f_arr <= FREQ_HIGH)
            fb = f_arr[bm]; Sb = Sxx[bm, :]

            for _, row in df.iterrows():
                # Tadarida-D outputs StTime and Dur in **milliseconds** — convert to seconds
                t0_rel = float(row["StTime"]) / 1000.0
                dur    = float(row["Dur"])    / 1000.0   # seconds
                t1_rel = t0_rel + dur

                # Skip detections in the overlap region (will be covered by next chunk)
                if t0_rel >= _CHUNK_S and (offset + chunk_frames) < nf:
                    continue
                if not (_MIN_CALL <= dur <= _MAX_CALL):
                    continue

                t0_abs = chunk_t0_s + t0_rel
                t1_abs = chunk_t0_s + t1_rel

                # Tadarida returns Fmin/Fmax in kHz
                Fmin_k = float(row["Fmin"])   # already kHz
                Fmax_k = float(row["Fmax"])

                # Contour tracking
                i0 = max(0,           np.searchsorted(t_arr, t0_rel - 0.001))
                i1 = min(Sb.shape[1], np.searchsorted(t_arr, t1_rel + 0.001))

                if i1 - i0 < 2:
                    fpeak   = (Fmin_k + Fmax_k) / 2
                    swp     = 0.0
                    contour = [[t0_abs, fpeak], [t1_abs, fpeak]]
                else:
                    flo_hz  = max(FREQ_LOW  * 1000, Fmin_k * 1000 * 0.75)
                    fhi_hz  = min(FREQ_HIGH * 1000, Fmax_k * 1000 * 1.25)
                    bm_seg  = (fb >= flo_hz) & (fb <= fhi_hz)
                    if not bm_seg.any():
                        bm_seg = np.ones(len(fb), dtype=bool)
                    seg_f   = fb[bm_seg]
                    seg     = Sb[bm_seg, :][:, i0:i1]
                    fc_t    = t_arr[i0:i1] + chunk_t0_s
                    fc_hz   = track_fundamental(seg, seg_f,
                                                Fmin_k * 1000, Fmax_k * 1000, sr)
                    Fmax_k  = fc_hz.max() / 1000
                    Fmin_k  = fc_hz.min() / 1000
                    fpeak   = seg_f[seg.mean(axis=1).argmax()] / 1000
                    tms     = np.linspace(0, dur * 1000, len(fc_hz))
                    swp     = (abs(np.polyfit(tms, fc_hz / 1000, 1)[0])
                               if len(fc_hz) > 2 else 0.0)
                    contour = [[float(ct), float(cf / 1000)]
                               for ct, cf in zip(fc_t, fc_hz)]

                raw.append({
                    "t0":       t0_abs, "t1":    t1_abs,
                    "dur":      dur * 1000,
                    "Fmax":     Fmax_k, "Fmin":  Fmin_k, "Fpeak": fpeak,
                    "sweep":    swp,
                    "contour":  contour,
                    "det_prob": 1.0,   # Tadarida-D doesn't output per-event confidence
                })

            offset += chunk_frames
            ch_idx_done = ch_idx + 1
            progress.update({"done": ch_idx_done,
                              "status": f"Detecting (Tadarida-D)… {ch_idx_done}/{total_ch}"})
            elapsed = time.time() - t_start
            eta     = elapsed / ch_idx_done * (total_ch - ch_idx_done)
            print(f"  chunk {ch_idx_done:3d}/{total_ch}  ({100*ch_idx_done//total_ch:3d}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  calls so far: {len(raw)}",
                  flush=True)

    # Merge overlapping detections from adjacent chunks
    from classify import merge
    merged = merge(raw)
    for c in merged:
        trim_call_contour(c)

    # Apply both classifiers
    reclassify_calls(merged)
    from species import COLORS
    from species import PROFILES as _PROFILES
    short_map = {p["name"]: p["short"] for p in _PROFILES}
    for idx, c in enumerate(merged):
        c["id"]    = idx
        c["color"] = COLORS.get(c["species"], "#888888")
        c["short"] = short_map.get(c["species"], "????")

    calls_list.extend(merged)
    elapsed_total = time.time() - t_start
    progress["status"] = (f"Done (Tadarida-D) — {len(calls_list)} calls  "
                          f"[{elapsed_total:.0f} s]")
    print(f"\n[Tadarida] Done in {elapsed_total:.0f} s  —  {len(calls_list)} calls",
          flush=True)

    # Cache to disk
    try:
        cache = {
            "version":    5,
            "audio_file": entry.path,
            "detector":   "tadarida",
            "calls":      calls_list,
        }
        with open(cache_path, "w") as fh:
            json.dump(cache, fh)
        print(f"[Tadarida] Results cached → {cache_path}")
    except Exception as exc:
        print(f"[Tadarida] Warning: could not write cache ({exc})")

    ready_ev.set()
    from tiles import _pregenerate_mask_tiles
    _threading.Thread(target=_pregenerate_mask_tiles, args=(entry,), daemon=True).start()


def try_load_tadarida_cache(entry) -> bool:
    """Load Tadarida-D results from cache into entry if available. Returns True on success."""
    import threading as _threading
    from startup import reclassify_calls, trim_call_contour

    det_key    = "tadarida"
    cache_path = os.path.splitext(entry.path)[0] + ".tadarida.calls.json"

    if not os.path.exists(cache_path):
        return False

    _CACHE_VERSION = 5
    try:
        with open(cache_path) as fh:
            cache = json.load(fh)
        if cache.get("version", 0) < _CACHE_VERSION:
            print(f"[Tadarida] Cache stale (v{cache.get('version',0)} < v{_CACHE_VERSION}) — will re-detect")
            return False
        calls = cache.get("calls", [])
        for c in calls:
            trim_call_contour(c)
        reclassify_calls(calls)
        from species import COLORS, PROFILES as _PROFILES
        short_map = {p["name"]: p["short"] for p in _PROFILES}
        for c in calls:
            c["color"] = COLORS.get(c["species"], "#888888")
            c["short"] = short_map.get(c["species"], "????")

        if det_key not in entry.calls_by_detector:
            entry.calls_by_detector[det_key]    = []
            entry.ready_by_detector[det_key]    = _threading.Event()
            entry.progress_by_detector[det_key] = {"done": 0, "total": 1, "status": "idle"}

        entry.calls_by_detector[det_key].extend(calls)
        entry.ready_by_detector[det_key].set()
        entry.progress_by_detector[det_key]["status"] = (
            f"Loaded from cache — {len(calls)} calls  [tadarida]")
        print(f"[Tadarida] Loaded {len(calls)} calls from {cache_path}")
        return True
    except Exception as exc:
        print(f"[Tadarida] Cache load failed ({exc})")
        return False


def _write_wav(path: str, mono: np.ndarray, sr: int):
    """Write a mono 16-bit PCM WAV (most compatible with Tadarida-D)."""
    import soundfile as sf
    # Clip to [-1, 1] and convert to int16 range; Tadarida-D expects integer PCM.
    pcm = np.clip(mono, -1.0, 1.0)
    sf.write(path, pcm, sr, subtype="PCM_16")
