"""
Bat vocalization detector and frequency analyzer.
Processes a high-sample-rate FLAC recording in chunks to detect
ultrasonic bat calls and measure their peak/fundamental frequencies.
"""

import soundfile as sf
import numpy as np
from scipy import signal
import sys

AUDIO_FILE = "/Users/brandon/claude/2025-05-28 1942 bats on campbell 1.flac"

# --- tuneable parameters ---
FREQ_LOW_HZ   = 15_000   # ignore everything below this (wind, handling noise)
FREQ_HIGH_HZ  = 96_000   # Nyquist for 192 kHz
CHUNK_SECS    = 5.0       # process in 5-second windows
NPERSEG       = 1024      # FFT window  → freq res ≈ 187.5 Hz at 192 kHz
NOVERLAP      = 768       # 75% overlap → time res ≈ 1.3 ms
# Energy threshold: a frame is "active" if its band power exceeds this
# multiple of the median band power for the chunk.
THRESHOLD_MULT = 6.0
# Minimum / maximum call duration in seconds
MIN_CALL_S    = 0.001     # 1 ms
MAX_CALL_S    = 0.100     # 100 ms
# Merge calls separated by less than this (handles fragmented detections)
MERGE_GAP_S   = 0.010     # 10 ms

def spectrogram_chunk(audio_mono, sr):
    """Return (freqs, times, power_dB) for one chunk."""
    f, t, Sxx = signal.spectrogram(
        audio_mono, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP,
        window="hann", scaling="density"
    )
    return f, t, 10 * np.log10(Sxx + 1e-12)

def detect_calls_in_chunk(f, t, Sxx_dB, chunk_offset_s):
    """
    Returns a list of dicts with keys:
      t_start, t_end, peak_freq_hz, freq_min_hz, freq_max_hz
    All times are absolute (chunk_offset already added).
    """
    # Restrict to bat frequency band
    band_mask = (f >= FREQ_LOW_HZ) & (f <= FREQ_HIGH_HZ)
    f_band    = f[band_mask]
    S_band    = Sxx_dB[band_mask, :]    # shape: (n_freqs, n_times)

    # Per-frame band energy (mean across frequencies)
    frame_energy = S_band.mean(axis=0)

    # Adaptive threshold from the median of this chunk
    threshold = np.median(frame_energy) + \
                THRESHOLD_MULT * frame_energy.std()

    active = frame_energy > threshold

    # Find contiguous active regions
    calls = []
    in_call = False
    for i, a in enumerate(active):
        if a and not in_call:
            call_start = i
            in_call = True
        elif not a and in_call:
            call_end = i
            in_call = False
            calls.append((call_start, call_end))
    if in_call:
        calls.append((call_start, len(active) - 1))

    results = []
    for (cs, ce) in calls:
        dur = t[ce] - t[cs]
        if dur < MIN_CALL_S or dur > MAX_CALL_S:
            continue
        # Frequency of maximum energy within this call segment
        seg = S_band[:, cs:ce+1]
        mean_spectrum = seg.mean(axis=1)
        peak_idx  = np.argmax(mean_spectrum)
        peak_freq = f_band[peak_idx]
        # -10 dB bandwidth around peak
        cutoff_dB = mean_spectrum[peak_idx] - 10.0
        active_bins = mean_spectrum >= cutoff_dB
        freq_min = f_band[active_bins][0]
        freq_max = f_band[active_bins][-1]
        results.append({
            "t_start":     chunk_offset_s + t[cs],
            "t_end":       chunk_offset_s + t[ce],
            "peak_freq_hz": peak_freq,
            "freq_min_hz":  freq_min,
            "freq_max_hz":  freq_max,
        })
    return results

def merge_calls(calls, gap_s=MERGE_GAP_S):
    """Merge calls that are very close together (single call split across frames)."""
    if not calls:
        return calls
    calls.sort(key=lambda c: c["t_start"])
    merged = [calls[0].copy()]
    for c in calls[1:]:
        prev = merged[-1]
        if c["t_start"] - prev["t_end"] < gap_s:
            prev["t_end"]       = max(prev["t_end"], c["t_end"])
            # merge frequency stats
            prev["peak_freq_hz"] = (prev["peak_freq_hz"] + c["peak_freq_hz"]) / 2
            prev["freq_min_hz"]  = min(prev["freq_min_hz"], c["freq_min_hz"])
            prev["freq_max_hz"]  = max(prev["freq_max_hz"], c["freq_max_hz"])
        else:
            merged.append(c.copy())
    return merged

def main():
    print(f"Opening: {AUDIO_FILE}")
    with sf.SoundFile(AUDIO_FILE) as f:
        sr          = f.samplerate
        n_channels  = f.channels
        n_frames    = f.frames
        duration_s  = n_frames / sr
        print(f"  Sample rate : {sr:,} Hz")
        print(f"  Channels    : {n_channels}")
        print(f"  Duration    : {duration_s:.1f} s ({duration_s/60:.1f} min)")

    chunk_frames = int(CHUNK_SECS * sr)
    all_calls = []
    offset_frames = 0

    with sf.SoundFile(AUDIO_FILE) as f:
        chunk_num = 0
        total_chunks = int(np.ceil(n_frames / chunk_frames))
        while offset_frames < n_frames:
            frames_to_read = min(chunk_frames, n_frames - offset_frames)
            audio = f.read(frames_to_read, dtype="float32", always_2d=True)
            # Mix to mono
            mono = audio.mean(axis=1)
            fq, t, Sxx = spectrogram_chunk(mono, sr)
            chunk_offset_s = offset_frames / sr
            calls = detect_calls_in_chunk(fq, t, Sxx, chunk_offset_s)
            all_calls.extend(calls)
            offset_frames += frames_to_read
            chunk_num += 1
            if chunk_num % 20 == 0 or chunk_num == total_chunks:
                pct = 100 * offset_frames / n_frames
                print(f"  ... {pct:.0f}% ({chunk_num}/{total_chunks} chunks)", flush=True)

    all_calls = merge_calls(all_calls)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total vocalizations detected : {len(all_calls)}")

    if all_calls:
        peaks    = np.array([c["peak_freq_hz"] for c in all_calls]) / 1000  # kHz
        mins     = np.array([c["freq_min_hz"]  for c in all_calls]) / 1000
        maxs     = np.array([c["freq_max_hz"]  for c in all_calls]) / 1000
        durs_ms  = np.array([(c["t_end"]-c["t_start"])*1000 for c in all_calls])

        print(f"\nPeak frequency (kHz):")
        print(f"  Min    : {peaks.min():.1f}")
        print(f"  Max    : {peaks.max():.1f}")
        print(f"  Mean   : {peaks.mean():.1f}")
        print(f"  Median : {np.median(peaks):.1f}")
        print(f"  Std    : {peaks.std():.1f}")
        print(f"  25th % : {np.percentile(peaks, 25):.1f}")
        print(f"  75th % : {np.percentile(peaks, 75):.1f}")

        print(f"\nCall frequency span (kHz) — per-call -10 dB bandwidth:")
        print(f"  Lowest  freq seen : {mins.min():.1f}")
        print(f"  Highest freq seen : {maxs.max():.1f}")
        print(f"  Mean bandwidth    : {(maxs - mins).mean():.1f}")

        print(f"\nCall duration (ms):")
        print(f"  Min    : {durs_ms.min():.1f}")
        print(f"  Max    : {durs_ms.max():.1f}")
        print(f"  Mean   : {durs_ms.mean():.1f}")
        print(f"  Median : {np.median(durs_ms):.1f}")

        # Histogram of peak frequencies
        bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 96]
        counts, edges = np.histogram(peaks, bins=bins)
        print(f"\nPeak frequency histogram (kHz):")
        for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
            bar = "#" * (cnt * 40 // max(counts, default=1))
            print(f"  {lo:>4.0f}–{hi:<4.0f} : {cnt:>5}  {bar}")

if __name__ == "__main__":
    main()
