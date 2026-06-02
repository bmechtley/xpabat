import json, os, time, threading
import numpy as np
from scipy import signal
from scipy.ndimage import label, binary_dilation, binary_erosion

# MPS (Metal) is not thread-safe: concurrent bd2.process_audio calls from
# multiple detection threads cause a segfault on macOS Apple Silicon.
# This lock serialises all GPU inference while allowing numpy/scipy work
# (spectrogram, contour tracking) to proceed in parallel across threads.
_gpu_lock = threading.Lock()

import config
from config import (
    FREQ_LOW, FREQ_HIGH, A_NPERSEG, A_NOVERLAP,
    BD2_THRESH, BD2_CHUNK_S, BD2_OVERLAP_S,
    CHUNK_SECS, MIN_CALL, MAX_CALL,
    THRESH_SIGMA,
)
from classify import merge, classify
from species import PROFILES, COLORS


def track_fundamental(seg, seg_f, low_hz, high_hz, sr):
    """
    Extract a continuous frequency contour that tracks the call's fundamental,
    not its harmonics.

    Two-stage approach
    ------------------
    1.  Initialise by finding the best energy peak within BD2's predicted
        frequency range (averaged over the first few frames).
    2.  For each subsequent frame, only consider frequencies within
        MAX_JUMP_HZ of the previous frame.  Among those candidates pick the
        one with maximum energy — the hard gate rejects harmonic jumps
        (always ≥ one fundamental width away) while allowing even the fastest
        observed bat FM sweeps (~20 kHz/ms).

    MAX_JUMP is computed from the STFT hop time and the fastest known bat
    sweep rate so it adapts to whatever sample rate the recording uses.
    """
    n = seg.shape[1]
    if n == 0:
        return np.array([], dtype=float)

    hop_s        = (A_NPERSEG - A_NOVERLAP) / sr   # seconds per STFT frame
    max_jump_hz  = 20_000 * hop_s * 1000            # 20 kHz/ms × step_ms
    max_jump_hz  = min(max_jump_hz, 15_000)

    tracked = np.empty(n)

    init_pow = seg.mean(axis=1)
    in_bd2   = (seg_f >= low_hz * 0.85) & (seg_f <= high_hz * 1.15)
    if in_bd2.any():
        masked     = np.where(in_bd2, init_pow, 0.0)
        tracked[0] = seg_f[masked.argmax()]
    else:
        tracked[0] = seg_f[init_pow.argmax()]

    for i in range(1, n):
        prev_f    = tracked[i - 1]
        reachable = np.abs(seg_f - prev_f) <= max_jump_hz
        if reachable.any():
            e          = seg[:, i] * reachable
            tracked[i] = seg_f[e.argmax()]
        else:
            tracked[i] = tracked[i - 1]

    return tracked


def run_detection(entry):
    sr, nf = entry.finfo["sr"], entry.finfo["nframes"]
    raw    = []

    # ── Try BatDetect2 first ──────────────────────────────────────
    use_bd2 = False
    try:
        import torch
        import batdetect2.api as bd2
        device = (torch.device("mps")
                  if torch.backends.mps.is_available()
                  else torch.device("cpu"))
        entry.detection_progress["status"] = f"Loading BatDetect2 on {str(device).upper()}…"
        with _gpu_lock:
            bd2_model, bd2_params = bd2.load_model(device=device)
        use_bd2 = True
        detector_label = f"BatDetect2/{str(device).upper()}"
    except Exception as exc:
        print(f"BatDetect2 unavailable ({exc}), using energy-threshold detector")
        detector_label = "threshold"

    chunk_frames   = int(BD2_CHUNK_S   * sr) if use_bd2 else int(CHUNK_SECS * sr)
    overlap_frames = int(BD2_OVERLAP_S * sr) if use_bd2 else 0
    total_ch       = int(np.ceil(nf / chunk_frames))
    entry.detection_progress["total"] = total_ch
    offset = 0; chunk_num = 0

    dur_s = nf / sr
    print(f"\nDetection starting  [{detector_label}]  {entry.name}")
    print(f"  Recording: {dur_s:.1f} s  |  chunks: {total_ch}  ({BD2_CHUNK_S if use_bd2 else CHUNK_SECS:.0f} s each)")
    if use_bd2:
        print(f"  Rough ETA: ~{dur_s/60*3:.0f} min on MPS  /  ~{dur_s/60*6:.0f} min on CPU")
    t_detect_start = time.time()

    while offset < nf:
        # Abort early if a stop was requested
        if entry.stop_event.is_set():
            print(f"Detection aborted: stop requested ({entry.name}).", flush=True)
            return

        end = min(nf, offset + chunk_frames + overlap_frames)
        with entry.audio_lock:
            entry.audio_fh.seek(offset)
            audio = entry.audio_fh.read(end - offset, dtype="float32", always_2d=True)
        mono           = audio.mean(axis=1)
        chunk_offset_s = offset / sr

        f, t, Sxx = signal.spectrogram(
            mono, fs=sr, nperseg=A_NPERSEG, noverlap=A_NOVERLAP, window="hann")
        bm = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
        fb = f[bm]; Sb = Sxx[bm, :]

        if use_bd2:
            # ── BatDetect2 detection ──────────────────────────────
            # Serialise GPU inference: MPS is not thread-safe
            with _gpu_lock:
                preds, _, _ = bd2.process_audio(
                    mono, sr, model=bd2_model, config=bd2_params, device=device)

            for p in preds:
                if p["det_prob"] < BD2_THRESH:
                    continue
                t0_rel = p["start_time"]
                if t0_rel >= BD2_CHUNK_S and (offset + chunk_frames) < nf:
                    continue
                t1_rel = p["end_time"]
                dur_s  = t1_rel - t0_rel
                if not (MIN_CALL <= dur_s <= MAX_CALL):
                    continue

                t0_abs = chunk_offset_s + t0_rel
                t1_abs = chunk_offset_s + t1_rel

                i0 = max(0,            np.searchsorted(t, t0_rel - 0.001))
                i1 = min(Sb.shape[1],  np.searchsorted(t, t1_rel + 0.001))

                # Try Hilbert contour first (sample-level resolution)
                from contour import (hilbert_contour      as _hilbert_contour,
                                     cwt_contour          as _cwt_contour,
                                     chirplet_contour     as _chirplet_contour,
                                     stft_contour         as _stft_contour,
                                     reassigned_contour   as _reassigned_contour)
                _hc = _hilbert_contour(mono, sr, t0_rel, t1_rel,
                                       p["low_freq"], p["high_freq"],
                                       chunk_t0_s=chunk_offset_s)
                if _hc is not None:
                    contour, _fc_hz, Fmin_k, Fmax_k, swp = _hc
                    fpeak = float(_fc_hz[len(_fc_hz)//2]) / 1000
                else:
                    # Fallback: STFT-based contour
                    if i1 - i0 < 2:
                        Fmin_k  = p["low_freq"]  / 1000
                        Fmax_k  = p["high_freq"] / 1000
                        fpeak   = (Fmin_k + Fmax_k) / 2
                        swp     = 0.0
                        contour = [[t0_abs, fpeak], [t1_abs, fpeak]]
                    else:
                        flo_hz  = max(FREQ_LOW  * 1000, p["low_freq"]  * 0.75)
                        fhi_hz  = min(FREQ_HIGH * 1000, p["high_freq"] * 1.25)
                        bm_seg  = (fb >= flo_hz) & (fb <= fhi_hz)
                        if not bm_seg.any():
                            bm_seg = np.ones(len(fb), dtype=bool)
                        seg_f   = fb[bm_seg]
                        seg     = Sb[bm_seg, :][:, i0:i1]
                        fc_t    = t[i0:i1] + chunk_offset_s
                        fc_hz   = track_fundamental(seg, seg_f,
                                                    p["low_freq"], p["high_freq"], sr)
                        Fmax_k  = fc_hz.max() / 1000
                        Fmin_k  = fc_hz.min() / 1000
                        fpeak   = seg_f[seg.mean(axis=1).argmax()] / 1000
                        tms     = np.linspace(0, dur_s * 1000, len(fc_hz))
                        swp     = (abs(np.polyfit(tms, fc_hz / 1000, 1)[0])
                                   if len(fc_hz) > 2 else 0.0)
                        contour = [[float(ct), float(cf / 1000)]
                                   for ct, cf in zip(fc_t, fc_hz)]

                # Alternative contours (run in parallel; each falls back to Hilbert)
                _cw    = _cwt_contour(mono, sr, t0_rel, t1_rel,
                                      p["low_freq"], p["high_freq"],
                                      chunk_t0_s=chunk_offset_s)
                _ch    = _chirplet_contour(mono, sr, t0_rel, t1_rel,
                                           p["low_freq"], p["high_freq"],
                                           chunk_t0_s=chunk_offset_s)
                _stft  = _stft_contour(mono, sr, t0_rel, t1_rel,
                                       p["low_freq"], p["high_freq"],
                                       chunk_t0_s=chunk_offset_s)
                _sharp = _reassigned_contour(mono, sr, t0_rel, t1_rel,
                                             p["low_freq"], p["high_freq"],
                                             chunk_t0_s=chunk_offset_s)

                raw.append({
                    "t0":             t0_abs,    "t1":    t1_abs,
                    "dur":            dur_s * 1000,
                    "Fmax":           Fmax_k,    "Fmin":  Fmin_k,  "Fpeak": fpeak,
                    "sweep":          swp,
                    "contour":        contour,
                    "contour_cwt":    _cw[0]    if _cw    is not None else contour,
                    "contour_chirp":  _ch[0]    if _ch    is not None else contour,
                    "contour_stft":   _stft[0]  if _stft  is not None else contour,
                    "contour_sharp":  _sharp[0] if _sharp is not None else contour,
                    "det_prob":       round(float(p["det_prob"]), 3),
                })

        else:
            # ── Fallback: energy-threshold detector ───────────────
            noise_pf = np.median(Sb, axis=1, keepdims=True)
            energy   = np.maximum(0, Sb - noise_pf).max(axis=0)
            act      = energy > np.median(energy) + THRESH_SIGMA * energy.std()
            act      = binary_dilation(binary_erosion(act, iterations=1), iterations=2)
            lbl, n   = label(act)

            for i in range(1, n + 1):
                idx  = np.where(lbl == i)[0]
                i0, i1 = idx[0], idx[-1]
                dur_s  = t[i1] - t[i0]
                if not (MIN_CALL <= dur_s <= MAX_CALL):
                    continue
                t0_rel = t[i0]
                t1_rel = t[i1]
                seg    = Sb[:, i0:i1+1]
                ms     = seg.mean(axis=1)
                fpeak  = fb[ms.argmax()] / 1000

                from contour import (hilbert_contour      as _hilbert_contour,
                                     cwt_contour          as _cwt_contour,
                                     chirplet_contour     as _chirplet_contour,
                                     stft_contour         as _stft_contour,
                                     reassigned_contour   as _reassigned_contour)
                _hc = _hilbert_contour(mono, sr, t0_rel, t1_rel,
                                       FREQ_LOW * 1000, FREQ_HIGH * 1000,
                                       chunk_t0_s=chunk_offset_s)
                if _hc is not None:
                    contour, _fc_hz, Fmin_k, Fmax_k, swp = _hc
                    fpeak = float(_fc_hz[len(_fc_hz)//2]) / 1000
                else:
                    fc_t   = t[i0:i1+1] + chunk_offset_s
                    fc_hz  = track_fundamental(seg, fb, FREQ_LOW * 1000, FREQ_HIGH * 1000, sr)
                    tms    = np.linspace(0, dur_s * 1000, len(fc_hz))
                    swp    = (abs(np.polyfit(tms, fc_hz / 1000, 1)[0])
                              if len(fc_hz) > 2 else 0.0)
                    Fmin_k  = fc_hz.min() / 1000
                    Fmax_k  = fc_hz.max() / 1000
                    contour = [[float(ct), float(cf / 1000)]
                               for ct, cf in zip(fc_t, fc_hz)]

                _cw    = _cwt_contour(mono, sr, t0_rel, t1_rel,
                                      FREQ_LOW * 1000, FREQ_HIGH * 1000,
                                      chunk_t0_s=chunk_offset_s)
                _ch    = _chirplet_contour(mono, sr, t0_rel, t1_rel,
                                           FREQ_LOW * 1000, FREQ_HIGH * 1000,
                                           chunk_t0_s=chunk_offset_s)
                _stft  = _stft_contour(mono, sr, t0_rel, t1_rel,
                                       FREQ_LOW * 1000, FREQ_HIGH * 1000,
                                       chunk_t0_s=chunk_offset_s)
                _sharp = _reassigned_contour(mono, sr, t0_rel, t1_rel,
                                             FREQ_LOW * 1000, FREQ_HIGH * 1000,
                                             chunk_t0_s=chunk_offset_s)

                raw.append({
                    "t0":             chunk_offset_s + t[i0],
                    "t1":             chunk_offset_s + t[i1],
                    "dur":            dur_s * 1000,
                    "Fmax":           Fmax_k,
                    "Fmin":           Fmin_k,
                    "Fpeak":          fpeak,
                    "sweep":          swp,
                    "contour":        contour,
                    "contour_cwt":    _cw[0]    if _cw    is not None else contour,
                    "contour_chirp":  _ch[0]    if _ch    is not None else contour,
                    "contour_stft":   _stft[0]  if _stft  is not None else contour,
                    "contour_sharp":  _sharp[0] if _sharp is not None else contour,
                    "det_prob":       0.0,
                })

        offset    += chunk_frames
        chunk_num += 1
        entry.detection_progress["done"]   = chunk_num
        entry.detection_progress["status"] = (f"Detecting ({detector_label})…"
                                              f" {chunk_num}/{total_ch}")
        if chunk_num % 5 == 0 or chunk_num == total_ch:
            elapsed = time.time() - t_detect_start
            eta     = elapsed / chunk_num * (total_ch - chunk_num) if chunk_num > 0 else 0
            print(f"  chunk {chunk_num:3d}/{total_ch}  ({100*chunk_num//total_ch:3d}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  calls so far: {len(raw)}", flush=True)

    merged = merge(raw)
    from startup import trim_call_contour
    for c in merged:
        trim_call_contour(c)
    for idx, c in enumerate(merged):
        sp, conf     = classify(c)
        c["id"]      = idx
        c["species"] = sp
        c["conf"]    = round(conf, 2)
        c["color"]   = COLORS.get(sp, "#888888")
        c["short"]   = next((p["short"] for p in PROFILES if p["name"] == sp), "????")

    entry.all_calls.extend(merged)
    from startup import compact_calls
    compact_calls(entry.all_calls)   # list-of-lists → float32 numpy (15× RAM reduction)
    entry.detection_progress["status"] = f"Done — {len(entry.all_calls)} calls  [{detector_label}]"
    print(f"\nDetection done in {time.time() - t_detect_start:.0f} s  —  "
          f"{len(entry.all_calls)} calls", flush=True)

    # ── Persist results to disk (v6 split format) ─────────────────
    try:
        from startup import save_calls_split
        save_calls_split(entry, detector_label, detector="batdetect2")
        print(f"Results cached → {entry.calls_dir}")
    except Exception as exc:
        print(f"Warning: could not write cache ({exc})")

    # ── Free the BatDetect2 model now that detection is complete ─────
    # The model weights (~100–200 MB) are no longer needed.  Explicitly
    # deleting and collecting prevents them sitting in RAM for the rest
    # of the server's lifetime, which matters on a 2 GB host.
    if use_bd2:
        try:
            del bd2_model, bd2_params
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    entry.calls_ready.set()
    print(entry.detection_progress["status"])
    from tiles import _pregenerate_mask_tiles
    threading.Thread(target=_pregenerate_mask_tiles, args=(entry,), daemon=True).start()
