"""
Refined bat species-level analyzer.

For each detected call extracts:
  - Fmax (start of sweep), Fmin (end of sweep)
  - Characteristic frequency (Fchar) = freq of peak energy
  - Duration, sweep rate (kHz/ms)
  - Call shape: FM vs QCF (quasi-constant-frequency)
  - Inter-pulse interval within call sequences

Then attempts species grouping against North American profiles.
"""

import soundfile as sf
import numpy as np
from scipy import signal
from scipy.ndimage import label, binary_dilation, binary_erosion
import sys, warnings
warnings.filterwarnings("ignore")

AUDIO_FILE = "/Users/brandon/claude/2025-05-28 1942 bats on campbell 1.flac"

# Spectrogram parameters — finer resolution than pass 1
NPERSEG  = 2048   # → freq res ≈ 93.75 Hz at 192 kHz
NOVERLAP = 1792   # 87.5% overlap → time step ≈ 1.3 ms

FREQ_LOW_HZ  = 13_000
FREQ_HIGH_HZ = 96_000

# Detection threshold: frames whose band power exceeds median + N*std
THRESHOLD_SIGMA = 5.0
MIN_CALL_S  = 0.0015   # 1.5 ms
MAX_CALL_S  = 0.120    # 120 ms
MERGE_GAP_S = 0.008    # merge detections within 8 ms

CHUNK_SECS  = 10.0     # seconds per processing block

# -----------------------------------------------------------------
# North-American bat species reference profiles
# Keys: Fchar range (kHz), Fmin range, sweep_rate (kHz/ms), duration
# Source: Bat Conservation International / Bat Call Library references
# -----------------------------------------------------------------
SPECIES_PROFILES = [
    {
        "name":   "Lasiurus cinereus (Hoary Bat)",
        "Fchar":  (16, 22),   # kHz
        "Fmin":   (13, 20),
        "dur_ms": (8, 25),
        "sweep":  (1.0, 5.0), # kHz/ms
        "notes":  "Low-freq, powerful FM sweeps; migratory species",
    },
    {
        "name":   "Eptesicus fuscus (Big Brown Bat)",
        "Fchar":  (22, 32),
        "Fmin":   (18, 28),
        "dur_ms": (8, 20),
        "sweep":  (0.5, 3.5),
        "notes":  "Loud calls, moderate sweep, very common",
    },
    {
        "name":   "Tadarida brasiliensis (Mexican Free-tailed Bat)",
        "Fchar":  (20, 28),
        "Fmin":   (18, 25),
        "dur_ms": (8, 25),
        "sweep":  (0.2, 1.5),
        "notes":  "Nearly constant frequency; often quasi-CF",
    },
    {
        "name":   "Lasiurus borealis (Eastern/Western Red Bat)",
        "Fchar":  (35, 50),
        "Fmin":   (25, 40),
        "dur_ms": (8, 20),
        "sweep":  (1.5, 5.0),
        "notes":  "Higher Fchar than E. fuscus; moderate calls",
    },
    {
        "name":   "Antrozous pallidus (Pallid Bat)",
        "Fchar":  (35, 50),
        "Fmin":   (28, 40),
        "dur_ms": (3, 12),
        "sweep":  (2.0, 8.0),
        "notes":  "Short steep FM; western species, gleaning hunter",
    },
    {
        "name":   "Myotis californicus / M. ciliolabrum (Small Myotis)",
        "Fchar":  (45, 70),
        "Fmin":   (32, 55),
        "dur_ms": (1.5, 6),
        "sweep":  (5.0, 20.0),
        "notes":  "Short, steep FM sweeps; high-frequency",
    },
    {
        "name":   "Myotis lucifugus / M. yumanensis (Medium Myotis)",
        "Fchar":  (40, 60),
        "Fmin":   (30, 50),
        "dur_ms": (2, 8),
        "sweep":  (3.0, 15.0),
        "notes":  "Steep FM; frequents water",
    },
    {
        "name":   "Parastrellus hesperus (Canyon Bat)",
        "Fchar":  (50, 70),
        "Fmin":   (38, 55),
        "dur_ms": (1.5, 5),
        "sweep":  (4.0, 15.0),
        "notes":  "High freq; western arid regions",
    },
    {
        "name":   "Corynorhinus townsendii (Townsend's Big-eared Bat)",
        "Fchar":  (30, 45),
        "Fmin":   (22, 35),
        "dur_ms": (1.5, 5),
        "sweep":  (3.0, 12.0),
        "notes":  "Short low-amplitude calls; cave specialist",
    },
]


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def spectrogram_band(mono, sr):
    f, t, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP,
        window="hann", scaling="density"
    )
    mask = (f >= FREQ_LOW_HZ) & (f <= FREQ_HIGH_HZ)
    return f[mask], t, Sxx[mask, :]


def detect_active_frames(S_lin, sigma=THRESHOLD_SIGMA):
    """Return boolean mask of time frames with elevated band energy."""
    energy = S_lin.mean(axis=0)
    med    = np.median(energy)
    std    = energy.std()
    thresh = med + sigma * std
    active = energy > thresh
    # erode then dilate to remove single-frame spikes
    active = binary_dilation(binary_erosion(active, iterations=1), iterations=2)
    return active


def segment_calls(active, t, chunk_offset):
    """Convert boolean active mask → list of (t_start, t_end) absolute times."""
    labeled, n = label(active)
    segs = []
    for i in range(1, n + 1):
        idx = np.where(labeled == i)[0]
        dur = t[idx[-1]] - t[idx[0]]
        if MIN_CALL_S <= dur <= MAX_CALL_S:
            segs.append((chunk_offset + t[idx[0]],
                         chunk_offset + t[idx[-1]],
                         idx[0], idx[-1]))
    return segs


def contour(f, S_lin_band, i0, i1):
    """
    Track frequency of peak power frame by frame → frequency contour (kHz).
    Returns array of peak frequencies for each time frame in [i0..i1].
    """
    seg = S_lin_band[:, i0:i1+1]
    peak_rows = np.argmax(seg, axis=0)
    return f[peak_rows] / 1000.0   # → kHz


def call_params(f, t, S_lin_band, t_start, t_end, i0, i1, chunk_offset):
    fc = contour(f, S_lin_band, i0, i1)
    dur_ms = (t_end - t_start) * 1000.0

    Fchar = fc.max()           # highest freq in contour (start of FM sweep)
    Fmin  = fc.min()           # lowest  freq in contour (end of sweep)
    Fmean = fc.mean()

    # Mean power spectrum for this call → peak energy frequency
    mean_spec  = S_lin_band[:, i0:i1+1].mean(axis=1)
    peak_e_idx = np.argmax(mean_spec)
    Fpeak_energy = f[peak_e_idx] / 1000.0  # kHz

    # Sweep rate: fit linear slope to contour
    times_ms = np.linspace(0, dur_ms, len(fc))
    if len(fc) > 2:
        slope = np.polyfit(times_ms, fc, 1)[0]  # kHz / ms
    else:
        slope = (fc[-1] - fc[0]) / max(dur_ms, 0.001)

    # QCF index: fraction of frames within 5 kHz of Fchar
    qcf_frac = np.mean(np.abs(fc - Fchar) < 5.0)

    return {
        "t_start":      t_start,
        "t_end":        t_end,
        "dur_ms":       dur_ms,
        "Fchar_kHz":    Fchar,
        "Fmin_kHz":     Fmin,
        "Fmean_kHz":    Fmean,
        "Fpeak_e_kHz":  Fpeak_energy,
        "sweep_kHzms":  abs(slope),
        "qcf_frac":     qcf_frac,
    }


def merge_calls(calls, gap_s=MERGE_GAP_S):
    if not calls:
        return []
    calls.sort(key=lambda c: c["t_start"])
    merged = [dict(calls[0])]
    for c in calls[1:]:
        p = merged[-1]
        if c["t_start"] - p["t_end"] < gap_s:
            # absorb into previous — keep broadest frequency range
            p["t_end"]       = max(p["t_end"], c["t_end"])
            p["dur_ms"]      = (p["t_end"] - p["t_start"]) * 1000
            p["Fchar_kHz"]   = max(p["Fchar_kHz"],  c["Fchar_kHz"])
            p["Fmin_kHz"]    = min(p["Fmin_kHz"],   c["Fmin_kHz"])
            p["Fmean_kHz"]   = (p["Fmean_kHz"] + c["Fmean_kHz"]) / 2
            p["Fpeak_e_kHz"] = (p["Fpeak_e_kHz"] + c["Fpeak_e_kHz"]) / 2
            p["sweep_kHzms"] = (p["sweep_kHzms"] + c["sweep_kHzms"]) / 2
            p["qcf_frac"]    = (p["qcf_frac"] + c["qcf_frac"]) / 2
        else:
            merged.append(dict(c))
    return merged


def score_species(call, profile):
    """0–1 score: fraction of parameters within expected range."""
    checks = [
        profile["Fchar"][0]  <= call["Fpeak_e_kHz"] <= profile["Fchar"][1],
        profile["Fmin"][0]   <= call["Fmin_kHz"]    <= profile["Fmin"][1],
        profile["dur_ms"][0] <= call["dur_ms"]       <= profile["dur_ms"][1],
        profile["sweep"][0]  <= call["sweep_kHzms"]  <= profile["sweep"][1],
    ]
    return sum(checks) / len(checks)


def classify_call(call):
    scores = {p["name"]: score_species(call, p) for p in SPECIES_PROFILES}
    best   = max(scores, key=scores.get)
    return best, scores[best]


def inter_pulse_intervals(calls):
    """Compute intervals between consecutive call starts (ms)."""
    starts = sorted(c["t_start"] for c in calls)
    if len(starts) < 2:
        return np.array([])
    return np.diff(starts) * 1000.0


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    print(f"Opening: {AUDIO_FILE}\n")
    with sf.SoundFile(AUDIO_FILE) as fh:
        sr         = fh.samplerate
        n_channels = fh.channels
        n_frames   = fh.frames
        duration_s = n_frames / sr

    chunk_frames = int(CHUNK_SECS * sr)
    all_calls    = []
    offset       = 0

    with sf.SoundFile(AUDIO_FILE) as fh:
        chunk_num   = 0
        total_chunks = int(np.ceil(n_frames / chunk_frames))
        while offset < n_frames:
            to_read = min(chunk_frames, n_frames - offset)
            audio   = fh.read(to_read, dtype="float32", always_2d=True)
            mono    = audio.mean(axis=1)
            f, t, Sxx = spectrogram_band(mono, sr)
            active    = detect_active_frames(Sxx)
            segs      = segment_calls(active, t, offset / sr)
            for (ts, te, i0, i1) in segs:
                params = call_params(f, t, Sxx, ts, te, i0, i1, offset / sr)
                all_calls.append(params)
            offset    += to_read
            chunk_num += 1
            if chunk_num % 25 == 0 or chunk_num == total_chunks:
                pct = 100 * offset / n_frames
                print(f"  {pct:.0f}%  ({chunk_num}/{total_chunks} chunks)  "
                      f"{len(all_calls)} calls so far", flush=True)

    all_calls = merge_calls(all_calls)
    N = len(all_calls)

    print(f"\n{'='*60}")
    print(f"REFINED RESULTS  —  {N} vocalizations detected")
    print(f"{'='*60}")

    if N == 0:
        print("No calls detected.")
        return

    # ---- per-parameter arrays ----
    Fchar   = np.array([c["Fpeak_e_kHz"] for c in all_calls])
    Fmin    = np.array([c["Fmin_kHz"]    for c in all_calls])
    Fmax    = np.array([c["Fchar_kHz"]   for c in all_calls])
    dur     = np.array([c["dur_ms"]      for c in all_calls])
    sweep   = np.array([c["sweep_kHzms"] for c in all_calls])
    qcf     = np.array([c["qcf_frac"]    for c in all_calls])
    ipis    = inter_pulse_intervals(all_calls)

    def stat(label, arr, unit=""):
        print(f"  {label:<28} min={arr.min():.1f}  "
              f"med={np.median(arr):.1f}  "
              f"mean={arr.mean():.1f}  "
              f"max={arr.max():.1f}  "
              f"std={arr.std():.1f}  {unit}")

    print("\nCharacteristic freq (peak energy, kHz):")
    stat("Fchar", Fchar, "kHz")
    print("\nSweep range (kHz):")
    stat("Fmax (sweep start)", Fmax, "kHz")
    stat("Fmin (sweep end)  ", Fmin, "kHz")
    print("\nCall duration (ms):")
    stat("Duration", dur, "ms")
    print("\nSweep rate (kHz/ms):")
    stat("Sweep rate", sweep, "kHz/ms")
    print(f"\nQCF fraction (>0.6 = quasi-CF call):")
    print(f"  Mean QCF index: {qcf.mean():.2f}  "
          f"  Fraction with QCF>0.6: {(qcf>0.6).mean()*100:.0f}%")
    if len(ipis):
        print(f"\nInter-pulse interval (ms):")
        stat("IPI (consecutive)", ipis, "ms")

    # ---- Species classification ----
    print(f"\n{'='*60}")
    print("SPECIES GROUPING")
    print(f"{'='*60}")

    species_counts = {}
    species_calls  = {}
    for c in all_calls:
        sp, sc = classify_call(c)
        if sc >= 0.5:                   # at least 2/4 criteria match
            species_counts[sp] = species_counts.get(sp, 0) + 1
            species_calls.setdefault(sp, []).append(c)
        else:
            species_counts["Unclassified"] = species_counts.get("Unclassified", 0) + 1
            species_calls.setdefault("Unclassified", []).append(c)

    total_classified = sum(v for k, v in species_counts.items() if k != "Unclassified")
    print(f"\nClassified {total_classified}/{N} calls to species level "
          f"(≥2/4 criteria match)\n")

    for sp, cnt in sorted(species_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / N
        bar = "#" * (cnt * 35 // N)
        print(f"  {cnt:>4} ({pct:>4.0f}%)  {bar}")
        print(f"           {sp}")
        if sp != "Unclassified" and sp in species_calls:
            sc_calls = species_calls[sp]
            fc_arr  = np.array([c["Fpeak_e_kHz"] for c in sc_calls])
            dur_arr = np.array([c["dur_ms"]       for c in sc_calls])
            sw_arr  = np.array([c["sweep_kHzms"]  for c in sc_calls])
            profile = next(p for p in SPECIES_PROFILES if p["name"] == sp)
            print(f"           Fchar: {fc_arr.mean():.1f} kHz (expected {profile['Fchar'][0]}–{profile['Fchar'][1]})")
            print(f"           Dur:   {dur_arr.mean():.1f} ms  "
                  f"Sweep: {sw_arr.mean():.2f} kHz/ms")
            print(f"           {profile['notes']}")
        print()

    # ---- Frequency histogram ----
    print("Peak-energy frequency histogram:")
    bins   = [13,16,18,20,22,24,26,28,30,32,35,40,45,50,55,60,70,80,96]
    counts, edges = np.histogram(Fchar, bins=bins)
    scale  = max(counts) if max(counts) else 1
    for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
        if cnt == 0:
            continue
        bar = "#" * max(1, cnt * 40 // scale)
        print(f"  {lo:>4.0f}–{hi:<4.0f} kHz : {cnt:>4}  {bar}")

    # ---- Duration vs Fchar table ----
    print("\nDuration × Fchar summary (median per Fchar bin):")
    print(f"  {'Fchar bin':<14} {'N':>5}  {'Dur(ms)':>8}  {'Sweep(kHz/ms)':>14}  {'Fmin':>6}")
    bin_edges = [13,20,25,30,35,45,55,96]
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (Fchar >= lo) & (Fchar < hi)
        if mask.sum() == 0:
            continue
        print(f"  {lo}–{hi} kHz    "
              f"{mask.sum():>5}  "
              f"{np.median(dur[mask]):>8.1f}  "
              f"{np.median(sweep[mask]):>14.2f}  "
              f"{np.median(Fmin[mask]):>6.1f}")


if __name__ == "__main__":
    main()
