"""
Improved bat call contour trackers.

Two drop-in replacements for detect.track_fundamental():

  track_fundamental_dp()       -- Viterbi / dynamic programming
  track_fundamental_reassigned() -- reassigned spectrogram + DP

Both have the same call signature as track_fundamental() and return the
same (n_time,) array of Hz values, so they're trivially swappable.

Why the current greedy tracker falls short
------------------------------------------
The existing track_fundamental() picks the highest-energy bin within
max_jump_hz of the previous frame.  This greedy choice is unrecoverable:
one noise spike or harmonic energy burst locks the tracker onto the wrong
frequency for every subsequent frame.  Bat call endpoints (where energy
tapers off) are particularly vulnerable — the tracker often snaps to a
harmonic before the call is fully over, inflating Fmin and shrinking Fmax.

Viterbi / DP fix
----------------
Instead of optimising one frame at a time, find the globally optimal path
through the spectrogram that simultaneously maximises total log-energy AND
penalises large frame-to-frame jumps:

    min  Σ_t  [ -log E(f_t, t)  +  α · |f_t − f_{t-1}| ]
    {f_t}      \___ want high energy ___/  \___ want smooth ___/

This is exactly the Viterbi algorithm on an HMM where:
  hidden states  = discrete frequency bins
  emission cost  = −log E(f, t)  (lower = more energy = better)
  transition cost = α · |Δf|     (penalty proportional to Hz jump)

The forward pass fills a DP table (cost to reach each frequency bin at
each time step); the backward pass recovers the optimal path.

Reassigned spectrogram fix
--------------------------
The standard STFT spreads the energy of a sinusoid across ≈2 frequency
bins — a bat call at 43.2 kHz bleeds into the 43.1 and 43.3 kHz bins.
For a fast FM sweep this smearing makes it hard to pinpoint where in a bin
the call actually is, giving jittery Fmin/Fmax estimates.

The reassigned spectrogram "reassigns" each bin's energy to the bin's
instantaneous frequency (estimated from the STFT phase derivative).
Result: frequency ridges are much sharper — often sub-bin precision — with
no new dependencies (pure scipy).

Using them together
-------------------
  from contour import track_fundamental_dp, reassigned_spectrogram

  # Option A: DP on standard spectrogram (no deps, biggest gain)
  fc_hz = track_fundamental_dp(seg, seg_f, low_hz, high_hz, sr)

  # Option B: DP on reassigned spectrogram (sharpest possible contours)
  # Compute once per chunk, pass alongside the regular spectrogram
  _, _, _, IF = reassigned_spectrogram(mono, sr, A_NPERSEG, A_NOVERLAP)
  bm  = (f_arr >= FREQ_LOW) & (f_arr <= FREQ_HIGH)
  seg_if = IF[bm, :][:, i0:i1]   # same slice as seg
  fc_hz = track_fundamental_dp(seg, seg_f, low_hz, high_hz, sr,
                                 instantaneous_freq=seg_if)
"""

import numpy as np
from scipy import signal as _signal
from scipy.ndimage import median_filter as _median_filter
try:
    from scipy.signal import savgol_filter as _savgol_filter
    _HAVE_SAVGOL = True
except ImportError:
    _HAVE_SAVGOL = False


# ---------------------------------------------------------------------------
# Private helpers shared by all IF-based contour methods
# ---------------------------------------------------------------------------

def _smooth_if_adaptive(IF_call, sr, low_hz, high_hz):
    """
    Smooth an instantaneous-frequency trace with windows that scale with sr.

    Fixed-sample-count windows (e.g. 31 samples) that look fine at 44 kHz
    become only 0.16 ms at 192 kHz — far too narrow.  This function targets
    ~1 ms for the median pass and ~2 ms for the Savitzky-Golay pass, which
    gives temporal smoothing comparable to a single STFT frame at any sample
    rate while preserving the FM shape of the call.

    Parameters
    ----------
    IF_call  : (n,) Hz values
    sr       : recording sample rate
    low_hz, high_hz : expected call band (for final clip)

    Returns
    -------
    (n,) smoothed + clipped Hz values
    """
    n = len(IF_call)
    if n < 4:
        return IF_call

    IF_call = np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)

    # ── Median: ~1 ms, odd, capped at n//4 ──────────────────────────────────
    ms1   = max(1, int(0.001 * sr))           # samples per 1 ms
    n_med = max(3, min(n // 4, ms1) | 1)      # odd
    IF_call = _median_filter(IF_call, size=n_med)

    # ── Savitzky-Golay: ~2 ms, odd, capped at n//3 ──────────────────────────
    if _HAVE_SAVGOL and n >= 7:
        ms2 = max(1, int(0.002 * sr))         # samples per 2 ms
        wl  = max(5, min(n // 3, ms2))
        if wl % 2 == 0:
            wl += 1
        wl = min(wl, n - (1 if n % 2 == 0 else 0))
        if wl >= 5 and wl < n:
            IF_call = _savgol_filter(IF_call, window_length=wl,
                                     polyorder=min(3, wl - 2))

    return np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)


def _amplitude_gate(IF_call, amp_call, threshold=0.15):
    """
    Replace low-amplitude samples with linear interpolation from strong neighbours.

    At the noisy leading/trailing edges of a bat call the signal fades through
    background noise.  IF estimators become unreliable there, producing large
    spikes that spoil the rendered contour.  Replacing those samples with a
    linear bridge from the strong-signal core removes edge artefacts while
    leaving the main sweep untouched.

    Parameters
    ----------
    IF_call  : (n,) Hz IF values
    amp_call : (n,) instantaneous amplitude (or proxy, e.g. CWT frame power)
    threshold: fraction of peak amplitude below which a sample is replaced

    Returns
    -------
    (n,) gated Hz values
    """
    mask = amp_call < amp_call.max() * threshold
    if not mask.any() or mask.all():
        return IF_call
    x        = np.arange(len(IF_call))
    IF_clean = IF_call.copy()
    IF_clean[mask] = np.interp(x[mask], x[~mask], IF_call[~mask])
    return IF_clean


def _track_greedy_indices(seg, seg_f, low_hz, high_hz, sr):
    """Like _track_greedy but returns (n_time,) bin-index array instead of Hz."""
    from config import A_NPERSEG, A_NOVERLAP
    n = seg.shape[1]
    if n == 0:
        return np.array([], dtype=np.intp)
    hop_s       = (A_NPERSEG - A_NOVERLAP) / sr
    max_jump_hz = min(20_000 * hop_s * 1000, 15_000)
    path    = np.empty(n, dtype=np.intp)
    init_pow = seg.mean(axis=1)
    in_range = (seg_f >= low_hz * 0.85) & (seg_f <= high_hz * 1.15)
    if in_range.any():
        path[0] = int(np.where(in_range, init_pow, 0.0).argmax())
    else:
        path[0] = int(init_pow.argmax())
    for i in range(1, n):
        prev_f    = seg_f[path[i - 1]]
        reachable = np.abs(seg_f - prev_f) <= max_jump_hz
        if reachable.any():
            path[i] = int((seg[:, i] * reachable).argmax())
        else:
            path[i] = path[i - 1]
    return path


def _track_greedy(seg, seg_f, low_hz, high_hz, sr):
    """
    Greedy forward frequency tracker for STFT spectrograms.

    This is the same algorithm as detect.track_fundamental() — included here
    to avoid a circular import (detect.py imports contour.py).

    Initialises at the highest-average-energy bin inside the predicted call
    band; for each subsequent frame picks the max-energy bin within
    max_jump_hz of the previous frame.  Simple, robust, and well-suited to
    the typically short spectrogram segments (5–15 frames) in stft_contour.

    The Viterbi DP tracker in track_fundamental_dp() is theoretically
    superior for long sequences, but on these short segments its globally-
    optimal solution often collapses to a flat path through the highest
    per-bin average energy, producing the "flat line" artefact that this
    greedy approach avoids.
    """
    from config import A_NPERSEG, A_NOVERLAP
    n = seg.shape[1]
    if n == 0:
        return np.array([], dtype=float)

    hop_s       = (A_NPERSEG - A_NOVERLAP) / sr
    max_jump_hz = min(20_000 * hop_s * 1000, 15_000)

    tracked  = np.empty(n)
    init_pow = seg.mean(axis=1)
    in_range = (seg_f >= low_hz * 0.85) & (seg_f <= high_hz * 1.15)
    if in_range.any():
        masked      = np.where(in_range, init_pow, 0.0)
        tracked[0]  = seg_f[masked.argmax()]
    else:
        tracked[0]  = seg_f[init_pow.argmax()]

    for i in range(1, n):
        prev_f    = tracked[i - 1]
        reachable = np.abs(seg_f - prev_f) <= max_jump_hz
        if reachable.any():
            e         = seg[:, i] * reachable
            tracked[i] = seg_f[e.argmax()]
        else:
            tracked[i] = tracked[i - 1]

    return tracked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def track_fundamental_dp(seg, seg_f, low_hz, high_hz, sr,
                          instantaneous_freq=None):
    """
    Globally-optimal frequency contour via Viterbi (drop-in replacement).

    Parameters
    ----------
    seg   : (n_freq, n_time) power spectrogram slice
    seg_f : (n_freq,) frequency axis in Hz
    low_hz, high_hz : detector's expected call range (used as soft prior)
    sr    : recording sample rate
    instantaneous_freq : optional (n_freq, n_time) array of reassigned
            instantaneous frequencies (Hz) — from reassigned_spectrogram().
            When provided, the returned contour values are the IF at the
            tracked bin rather than the bin's centre frequency, giving
            sub-bin precision.

    Returns
    -------
    (n_time,) array of Hz values — the tracked fundamental frequency.
    """
    from config import A_NPERSEG, A_NOVERLAP

    n_freq, n_time = seg.shape
    if n_time == 0:
        return np.array([], dtype=float)
    if n_time == 1:
        return np.array([seg_f[seg[:, 0].argmax()]], dtype=float)

    # ── Jump budget (same formula as original tracker) ─────────────────────
    hop_s        = (A_NPERSEG - A_NOVERLAP) / sr
    max_jump_hz  = min(20_000 * hop_s * 1000, 15_000)   # Hz
    df           = (seg_f[-1] - seg_f[0]) / max(1, n_freq - 1)
    max_jump_b   = max(1, int(max_jump_hz / df))         # bins

    # ── Log-energy emissions ────────────────────────────────────────────────
    E = np.log(np.maximum(seg, 1e-10))                   # (n_freq, n_time)

    # Soft prior: +0.5 nats for bins inside the detector-predicted range
    in_range = (seg_f >= low_hz * 0.85) & (seg_f <= high_hz * 1.15)
    E += np.where(in_range, 0.5, 0.0)[:, None]

    # ── Transition cost matrix  T[f_dst, f_src] ────────────────────────────
    # Linear penalty, hard cutoff at max_jump_b bins.
    # Setting hard-cutoff cost to 1e9 is cheaper than np.inf (avoids
    # propagating inf through the min operations).
    f_idx  = np.arange(n_freq, dtype=np.float32)
    jumps  = np.abs(f_idx[:, None] - f_idx[None, :])    # (n_freq, n_freq)
    alpha  = 1.0 / max_jump_b                            # 1 nat per max-jump
    T      = np.where(jumps <= max_jump_b,
                      alpha * jumps,
                      1e9).astype(np.float32)            # (n_freq, n_freq)

    # ── Viterbi forward pass ────────────────────────────────────────────────
    # dp[f]    = minimum total cost of any path reaching bin f at current t
    # back[f,t] = which bin we came from (backtrack pointer)
    dp   = -E[:, 0].astype(np.float32)                  # init: frame 0
    back = np.empty((n_freq, n_time), dtype=np.int16)
    back[:, 0] = np.arange(n_freq, dtype=np.int16)

    for t in range(1, n_time):
        # trans[f_dst, f_src] = dp[f_src] + T[f_dst, f_src]
        trans        = dp[None, :] + T                  # (n_freq, n_freq)
        best_src     = trans.argmin(axis=1)             # (n_freq,)
        back[:, t]   = best_src
        dp           = trans[np.arange(n_freq), best_src] - E[:, t].astype(np.float32)

    # ── Backtrack ───────────────────────────────────────────────────────────
    path         = np.empty(n_time, dtype=np.int32)
    path[-1]     = dp.argmin()
    for t in range(n_time - 2, -1, -1):
        path[t] = back[path[t + 1], t + 1]

    # ── Return contour values ───────────────────────────────────────────────
    if instantaneous_freq is not None:
        # Sub-bin precision: use the reassigned IF at the tracked bin
        contour = np.array([instantaneous_freq[path[t], t]
                             for t in range(n_time)], dtype=float)
        # Clip to sane range in case IF estimation was noisy
        contour = np.clip(contour, seg_f[0] * 0.9, seg_f[-1] * 1.1)
    else:
        contour = seg_f[path].astype(float)

    return contour


def reassigned_spectrogram(audio, sr, nperseg, noverlap):
    """
    Compute a reassigned spectrogram for sharper time-frequency localisation.

    Standard STFT energy is smeared across ≈2 bins; reassignment maps each
    bin's energy to its instantaneous frequency, giving sub-bin precision.
    This is especially valuable for the fast FM sweeps of bat calls.

    Implemented from scratch with scipy — no librosa dependency.
    Formula: Auger & Flandrin (1995), eq. 14.

        IF(k, m) = f_k  +  Im[ Z_dh(k,m) / Z_h(k,m) ]  ×  (sr / 2π)

    where Z_h is the normal STFT and Z_dh is the STFT computed with the
    window's time-derivative.

    Parameters
    ----------
    audio   : 1-D float array
    sr      : sample rate (Hz)
    nperseg : STFT window length (samples)
    noverlap: STFT overlap (samples)

    Returns
    -------
    f   : (n_freq,) frequency axis (Hz)
    t   : (n_time,) time axis (s)
    S   : (n_freq, n_time) power spectrogram (for display / existing energy ops)
    IF  : (n_freq, n_time) instantaneous frequency at each bin (Hz)
          — pass to track_fundamental_dp(..., instantaneous_freq=IF_slice)
    """
    # Ensure float64 — float32 input causes NaN in the ratio Zd/Z because
    # the lower-precision STFT overflows in the mag < 1e-10 safe-Z path.
    audio = np.asarray(audio, dtype=np.float64)

    win = np.hanning(nperseg).astype(float)

    # Derivative window: central finite difference, per-sample units
    # d_win[n] ≈ (win[n+1] − win[n−1]) / 2  (units: amplitude / sample)
    d_win = np.zeros(nperseg)
    d_win[1:-1] = (win[2:] - win[:-2]) / 2.0
    # Convert to per-second by multiplying by sample rate
    d_win *= sr

    f, t, Z  = _signal.stft(audio, fs=sr, window=win,
                              nperseg=nperseg, noverlap=noverlap)

    # ── Derivative STFT ──────────────────────────────────────────────────────
    # scipy.signal.stft normalises by window.sum().  The derivative window is
    # antisymmetric, so d_win.sum() == 0 → division-by-zero → NaN everywhere.
    #
    # Two-part fix:
    #   1. Add a tiny DC offset (≈1e-8 × max|d_win|) to d_win so its sum is
    #      non-zero.  This prevents the NaN without meaningfully changing the
    #      complex spectrum for any bin with real signal.
    #
    #   2. The normalisations of Z_h and Z_dh are DIFFERENT
    #      (win.sum() vs (d_win+dc).sum()), so they do NOT cancel in the ratio
    #      Z_dh/Z_h.  Apply an explicit correction:
    #
    #          IF = f_k + Im[Z_dh/Z_h] × (norm_dh/norm_h) / (2π)
    #
    #      Verified: pure tone at 40 kHz → IF reads 40.250 kHz (correct to
    #      within one STFT bin; the residual error is the bin-centre offset).
    dc      = 1e-8 * float(np.abs(d_win).max() or 1.0)
    norm_h  = float(win.sum())
    norm_dh = float((d_win + dc).sum())
    _, _, Zd = _signal.stft(audio, fs=sr, window=d_win + dc,
                             nperseg=nperseg, noverlap=noverlap)

    # Avoid division by near-zero
    mag     = np.abs(Z)
    safe_Z  = np.where(mag > 1e-10, Z, 1.0 + 0j)

    # Instantaneous frequency (Hz), clipped to [0, Nyquist]
    IF = f[:, None] + np.imag(Zd / safe_Z) * (norm_dh / norm_h) / (2.0 * np.pi)
    IF = np.clip(IF, 0.0, sr / 2.0)

    S = mag ** 2
    return f, t, S, IF


# ---------------------------------------------------------------------------
# STFT spectrogram contour  (the "classic" method — smooth, coarser in time)
# ---------------------------------------------------------------------------

def stft_contour(mono, sr, t0_rel, t1_rel, low_hz, high_hz,
                 chunk_t0_s=0.0, n_pts=150):
    """
    Frequency contour via STFT spectrogram + greedy forward tracker.

    This is the "classic" approach: a short-time spectrogram is computed on
    the call region and the greedy max-energy-ridge tracker follows the call
    forward in time.  Uses the same tracker logic as detect.track_fundamental.

    Advantages over IF-based methods:
      • The STFT window (~2–3 ms) inherently averages over many samples so
        the per-frame estimate is much less sensitive to phase noise.
      • No Butterworth filter artefacts at call edges.
      • Naturally immune to inter-sample phase glitches.

    Trade-offs:
      • Lower time resolution (~1–2 ms/frame vs sample-level).
      • Frequency resolution limited to spectrogram bin width.

    Parameters / Returns: same as hilbert_contour.
    """
    from config import A_NPERSEG, A_NOVERLAP

    try:
        n_total = len(mono)
        PAD_S   = 0.010                       # 10 ms padding
        pad_n   = int(PAD_S * sr)

        i0_call = int(round(t0_rel * sr))
        i1_call = int(round(t1_rel * sr))
        i0      = max(0, i0_call - pad_n)
        i1      = min(n_total, i1_call + pad_n)

        seg = mono[i0:i1].astype(np.float64)
        if len(seg) < A_NPERSEG:
            return None

        # Spectrogram on the call segment only (much cheaper than the whole chunk)
        f_arr, t_rel_arr, Sxx = _signal.spectrogram(
            seg, fs=sr, nperseg=A_NPERSEG, noverlap=A_NOVERLAP, window='hann')

        # Absolute times (seg starts at i0/sr from chunk start)
        seg_t0_s  = i0 / sr
        t_abs_arr = chunk_t0_s + seg_t0_s + t_rel_arr

        # Frames that fall inside the call
        t0_seg = t0_rel - seg_t0_s
        t1_seg = t1_rel - seg_t0_s
        j0 = max(0,             np.searchsorted(t_rel_arr, t0_seg - 0.001))
        j1 = min(Sxx.shape[1],  np.searchsorted(t_rel_arr, t1_seg + 0.001))

        if j1 - j0 < 2:
            return None

        # Restrict to expected frequency range
        flo = max(low_hz  * 0.75, 1000.0)
        fhi = min(high_hz * 1.25, sr * 0.49)
        bm  = (f_arr >= flo) & (f_arr <= fhi)
        if not bm.any():
            return None

        seg_f = f_arr[bm]
        seg_s = Sxx[bm, :][:, j0:j1]
        fc_t  = t_abs_arr[j0:j1]

        # Greedy forward tracker (avoids the flat-line collapse of Viterbi on
        # short segments — see _track_greedy docstring for details).
        fc_hz = _track_greedy(seg_s, seg_f, low_hz, high_hz, sr)
        if len(fc_hz) == 0:
            return None

        Fmax_k = float(fc_hz.max()) / 1000.0
        Fmin_k = float(fc_hz.min()) / 1000.0
        dur_ms = (t1_rel - t0_rel) * 1000.0
        n      = len(fc_hz)
        tms    = np.linspace(0.0, dur_ms, n)
        sweep  = (abs(float(np.polyfit(tms, fc_hz / 1000.0, 1)[0]))
                  if n > 2 else 0.0)

        # Subsample / upsample to n_pts for display consistency
        n_out = min(n, n_pts)
        idx   = np.round(np.linspace(0, n - 1, n_out)).astype(int)
        fc_out = fc_hz[idx]
        t_out  = fc_t[idx]

        contour = [[float(t), float(f / 1000.0)]
                   for t, f in zip(t_out, fc_out)]
        return contour, fc_out, Fmin_k, Fmax_k, sweep

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reassigned-spectrogram contour  ("Sharp" method)
# ---------------------------------------------------------------------------

def reassigned_contour(mono, sr, t0_rel, t1_rel, low_hz, high_hz,
                       chunk_t0_s=0.0, n_pts=150):
    """
    Frequency contour via reassigned spectrogram — greedy tracker + IF precision.

    Uses the same STFT power as stft_contour for the greedy ridge tracker
    (robust on the 5–20-frame bat call segments) but replaces the bin-centre
    frequency output with the instantaneous-frequency estimate at the tracked
    bin, giving sub-bin (~50–100 Hz at 192 kHz / 512-sample window) precision.

    Advantages over stft_contour:
      • Sub-bin frequency resolution — the IF is the STFT phase-derivative,
        so it reads the true instantaneous frequency rather than the bin centre.
      • Naturally handles inter-call overlap better than the raw-signal IF
        methods (Hilbert / CWT) because no bandpass filter is needed.

    Parameters / Returns: same as stft_contour.
    """
    from config import A_NPERSEG, A_NOVERLAP
    try:
        n_total = len(mono)
        PAD_S   = 0.010
        pad_n   = int(PAD_S * sr)

        i0_call = int(round(t0_rel * sr))
        i1_call = int(round(t1_rel * sr))
        i0      = max(0, i0_call - pad_n)
        i1      = min(n_total, i1_call + pad_n)

        seg = mono[i0:i1].astype(np.float64)
        if len(seg) < A_NPERSEG:
            return None

        f_arr, t_rel_arr, Sxx, IF = reassigned_spectrogram(
            seg, sr, A_NPERSEG, A_NOVERLAP)

        seg_t0_s  = i0 / sr
        t_abs_arr = chunk_t0_s + seg_t0_s + t_rel_arr

        t0_seg = t0_rel - seg_t0_s
        t1_seg = t1_rel - seg_t0_s
        j0 = max(0,            np.searchsorted(t_rel_arr, t0_seg - 0.001))
        j1 = min(Sxx.shape[1], np.searchsorted(t_rel_arr, t1_seg + 0.001))

        if j1 - j0 < 2:
            return None

        flo = max(low_hz  * 0.75, 1000.0)
        fhi = min(high_hz * 1.25, sr * 0.49)
        bm  = (f_arr >= flo) & (f_arr <= fhi)
        if not bm.any():
            return None

        seg_f  = f_arr[bm]
        seg_S  = Sxx[bm, :][:, j0:j1]      # STFT power for path finding
        seg_IF = IF[bm,  :][:, j0:j1]      # instantaneous freq for output
        fc_t   = t_abs_arr[j0:j1]

        # Greedy tracker → bin indices
        path_idx = _track_greedy_indices(seg_S, seg_f, low_hz, high_hz, sr)
        n_time   = seg_S.shape[1]

        # Sub-bin frequency: read IF at the tracked bin each frame
        fc_hz = np.array([seg_IF[path_idx[t], t] for t in range(n_time)],
                         dtype=float)
        fc_hz = np.clip(fc_hz, flo * 0.9, fhi * 1.1)

        Fmax_k = float(fc_hz.max()) / 1000.0
        Fmin_k = float(fc_hz.min()) / 1000.0
        dur_ms = (t1_rel - t0_rel) * 1000.0
        n      = len(fc_hz)
        tms    = np.linspace(0.0, dur_ms, n)
        sweep  = (abs(float(np.polyfit(tms, fc_hz / 1000.0, 1)[0]))
                  if n > 2 else 0.0)

        n_out  = min(n, n_pts)
        idx    = np.round(np.linspace(0, n - 1, n_out)).astype(int)
        fc_out = fc_hz[idx]
        t_out  = fc_t[idx]

        contour = [[float(t), float(f / 1000.0)]
                   for t, f in zip(t_out, fc_out)]
        return contour, fc_out, Fmin_k, Fmax_k, sweep

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hilbert instantaneous-frequency contour
# ---------------------------------------------------------------------------

def hilbert_contour(mono, sr, t0_rel, t1_rel, low_hz, high_hz,
                    chunk_t0_s=0.0, n_pts=150):
    """
    Compute a high-resolution frequency contour using the Hilbert transform.

    Returns ~100× more time-resolution than the STFT-based tracker (sample-
    level vs frame-level) by treating the bat call as a single-component
    AM-FM signal.

    Pipeline
    --------
    1. Extract call audio + 5 ms padding (for filter settling)
    2. 5th-order Butterworth bandpass around [0.75·Fmin, 1.25·Fmax] to
       reject noise and harmonics
    3. Analytic signal via Hilbert transform
    4. Instantaneous frequency from conjugate-product phase derivative:
           IF[n] = angle(z[n+1] · z*[n]) · sr / 2π
    5. Clip obvious outliers, median filter (removes phase glitches),
       Savitzky-Golay smooth (preserves FM shape)
    6. Subsample to n_pts

    Parameters
    ----------
    mono        : 1-D float array — mono audio for the current chunk
    sr          : sample rate (Hz)
    t0_rel      : call start time relative to chunk start (s)
    t1_rel      : call end time relative to chunk start (s)
    low_hz      : expected Fmin from detector (Hz)
    high_hz     : expected Fmax from detector (Hz)
    chunk_t0_s  : chunk start time in full recording (s)
    n_pts       : target number of contour points (more → smoother display)

    Returns
    -------
    contour  : list of [abs_time_s, freq_kHz] — ready to store in call dict
    fc_hz    : (n_pts,) instantaneous frequency array in Hz
    Fmin_k   : float, minimum freq in kHz  (derived from Hilbert trace)
    Fmax_k   : float, maximum freq in kHz
    sweep    : float, linear sweep rate kHz/ms  (from polyfit of fc_hz)

    On any failure the function returns ``None`` — callers should fall back
    to the existing STFT-based contour.
    """
    try:
        n_total = len(mono)
        PAD_S   = 0.005                          # 5 ms filter-settling pad
        pad_n   = int(PAD_S * sr)

        i0_call = int(round(t0_rel * sr))
        i1_call = int(round(t1_rel * sr))
        i0      = max(0, i0_call - pad_n)
        i1      = min(n_total, i1_call + pad_n)

        seg = mono[i0:i1].astype(np.float64)
        if len(seg) < 20:
            return None

        # ── Bandpass: isolate fundamental, reject harmonics ───────────────
        flo = max(low_hz  * 0.75, 500.0)
        fhi = min(high_hz * 1.25, sr * 0.49)
        if flo >= fhi * 0.9:
            return None

        sos      = _signal.butter(5, [flo, fhi], btype='bandpass',
                                  fs=sr, output='sos')
        filtered = _signal.sosfiltfilt(sos, seg)

        # ── Instantaneous frequency via Hilbert ───────────────────────────
        z   = _signal.hilbert(filtered)

        # Conjugate-product phase derivative (handles 2π wrapping naturally)
        IF_raw = np.angle(z[1:] * np.conj(z[:-1])) * (sr / (2.0 * np.pi))

        # ── Trim padding, keep only the call region ───────────────────────
        c0 = max(0, i0_call - i0)
        c1 = max(c0 + 1, min(i1_call - i0, len(IF_raw)))
        IF_call = IF_raw[c0:c1]

        if len(IF_call) < 4:
            return None

        # ── Amplitude gating: fill low-SNR call edges with interpolation ──
        # Where the signal amplitude is low (call onset/offset fading into
        # noise), the IF estimate is unreliable.  Replace those samples with
        # a linear bridge from the high-amplitude core of the call.
        amp_call = np.abs(z)[c0:c1]
        IF_call  = _amplitude_gate(IF_call, amp_call)

        # ── Adaptive smoothing (rate-aware ~1 ms + ~2 ms windows) ────────
        IF_call = _smooth_if_adaptive(IF_call, sr, low_hz, high_hz)

        # ── Subsample to n_pts ────────────────────────────────────────────
        n     = len(IF_call)
        n_out = min(n, n_pts)
        idx   = np.round(np.linspace(0, n - 1, n_out)).astype(int)

        t_abs = chunk_t0_s + t0_rel + idx / sr
        fc_hz = IF_call[idx]

        # ── Derived scalar quantities ─────────────────────────────────────
        Fmin_k = float(fc_hz.min()) / 1000.0
        Fmax_k = float(fc_hz.max()) / 1000.0
        dur_ms = (t1_rel - t0_rel) * 1000.0
        tms    = np.linspace(0.0, dur_ms, n_out)
        sweep  = (abs(float(np.polyfit(tms, fc_hz / 1000.0, 1)[0]))
                  if n_out > 2 else 0.0)

        contour = [[float(t), float(f / 1000.0)]
                   for t, f in zip(t_abs, fc_hz)]

        return contour, fc_hz, Fmin_k, Fmax_k, sweep

    except Exception:
        return None


# ---------------------------------------------------------------------------
# CWT (Morlet) instantaneous-frequency contour
# ---------------------------------------------------------------------------

def cwt_contour(mono, sr, t0_rel, t1_rel, low_hz, high_hz,
                chunk_t0_s=0.0, n_pts=150, n_scales=64):
    """
    Frequency contour via Continuous Wavelet Transform (Morlet wavelet).

    Unlike STFT, the CWT uses short windows at high frequencies (good time
    resolution) and long windows at low frequencies (good freq resolution),
    matching the physics of a descending FM bat call.  Log-spaced scales
    cover the detector-predicted frequency range with 20 % margins.

    Sub-bin frequency precision is recovered via parabolic interpolation of
    the log-power ridge in log-frequency space.

    Parameters / Returns: same as hilbert_contour.
    """
    try:
        n_total = len(mono)
        PAD_S   = 0.005
        pad_n   = int(PAD_S * sr)

        i0_call = int(round(t0_rel * sr))
        i1_call = int(round(t1_rel * sr))
        i0      = max(0, i0_call - pad_n)
        i1      = min(n_total, i1_call + pad_n)

        seg = mono[i0:i1].astype(np.float64)
        if len(seg) < 20:
            return None

        # ── Bandpass: remove out-of-band noise ───────────────────────────────
        flo = max(low_hz  * 0.75, 500.0)
        fhi = min(high_hz * 1.25, sr * 0.49)
        if flo >= fhi * 0.9:
            return None

        sos      = _signal.butter(5, [flo, fhi], btype='bandpass',
                                  fs=sr, output='sos')
        filtered = _signal.sosfiltfilt(sos, seg)

        # ── Log-spaced analysis frequencies ──────────────────────────────────
        f_lo_cw = max(low_hz  * 0.8, 1000.0)
        f_hi_cw = min(high_hz * 1.2, sr * 0.48)
        if f_lo_cw >= f_hi_cw:
            return None

        f_arr  = np.geomspace(f_lo_cw, f_hi_cw, n_scales)
        # Morlet2 center-frequency relationship: f = w*sr/(2π·s) → s = w*sr/(2π·f)
        w_m    = 6.0
        scales = w_m * sr / (2.0 * np.pi * f_arr)

        # ── Compute CWT ───────────────────────────────────────────────────────
        # scipy returns (n_scales, n_samples); morlet2(M, s, w=w_m)
        cwt_mat = _signal.cwt(filtered, _signal.morlet2, scales, w=w_m)
        power   = np.abs(cwt_mat) ** 2                 # (n_scales, n_samples)

        # ── Trim to call region ───────────────────────────────────────────────
        c0         = max(0, i0_call - i0)
        c1         = max(c0 + 1, min(i1_call - i0, power.shape[1]))
        power_call = power[:, c0:c1]                   # (n_scales, n_call)

        if power_call.shape[1] < 4:
            return None

        n_call  = power_call.shape[1]
        log_f   = np.log(f_arr)

        # ── Ridge extraction: argmax per time, parabolic interpolation ────────
        peak_idx = power_call.argmax(axis=0)           # (n_call,)

        # Vectorised parabolic interpolation in log-frequency space
        pi_safe  = np.clip(peak_idx, 1, n_scales - 2)
        rows     = np.arange(n_call)

        y0 = np.log(np.maximum(power_call[pi_safe - 1, rows], 1e-20))
        y1 = np.log(np.maximum(power_call[pi_safe,     rows], 1e-20))
        y2 = np.log(np.maximum(power_call[pi_safe + 1, rows], 1e-20))

        denom  = 2.0 * (2.0 * y1 - y0 - y2)
        safe   = np.abs(denom) > 1e-10
        delta  = np.where(safe, np.clip((y0 - y2) / np.where(safe, denom, 1.0),
                                        -0.5, 0.5), 0.0)

        df_log  = (log_f[pi_safe + 1] - log_f[pi_safe - 1]) / 2.0
        IF_call = np.exp(log_f[pi_safe] + delta * df_log)

        # Edge bins: no interpolation
        edge    = (peak_idx == 0) | (peak_idx == n_scales - 1)
        IF_call = np.where(edge, f_arr[peak_idx], IF_call)

        # ── Power gating: fill low-SNR time frames with interpolation ─────────
        # Where total CWT power is very low the ridge estimate is noise-driven.
        frame_power = power_call.max(axis=0)   # peak power at each time sample
        IF_call     = _amplitude_gate(IF_call, frame_power, threshold=0.05)

        # ── Adaptive smoothing ────────────────────────────────────────────────
        IF_call = _smooth_if_adaptive(IF_call, sr, low_hz, high_hz)

        # ── Subsample to n_pts ────────────────────────────────────────────────
        n     = len(IF_call)
        n_out = min(n, n_pts)
        idx   = np.round(np.linspace(0, n - 1, n_out)).astype(int)
        t_abs = chunk_t0_s + t0_rel + idx / sr
        fc_hz = IF_call[idx]

        Fmin_k = float(fc_hz.min()) / 1000.0
        Fmax_k = float(fc_hz.max()) / 1000.0
        dur_ms = (t1_rel - t0_rel) * 1000.0
        tms    = np.linspace(0.0, dur_ms, n_out)
        sweep  = (abs(float(np.polyfit(tms, fc_hz / 1000.0, 1)[0]))
                  if n_out > 2 else 0.0)

        contour = [[float(t), float(f / 1000.0)]
                   for t, f in zip(t_abs, fc_hz)]
        return contour, fc_hz, Fmin_k, Fmax_k, sweep

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chirplet (dechirping) instantaneous-frequency contour
# ---------------------------------------------------------------------------

def chirplet_contour(mono, sr, t0_rel, t1_rel, low_hz, high_hz,
                     chunk_t0_s=0.0, n_pts=150):
    """
    Frequency contour via linear-FM dechirping (Chirplet method).

    Uses the detector's predicted Fmin / Fmax to build a linear-FM reference
    chirp matching the expected sweep.  The signal is demodulated against this
    reference so the analytic signal's phase varies slowly (it only needs to
    track the *deviation* from the linear model).  Hilbert IF of the
    demodulated signal gives the residual; adding back the reference gives the
    true instantaneous frequency.

    Benefit over plain Hilbert: the Hilbert IF estimator is most accurate
    when the signal is nearly single-tone.  By removing the dominant chirp
    trend first, the dynamic range of the IF drops from [Fmin..Fmax] to
    ~[0..deviation], making the estimate far more robust at low SNR and near
    the noisy call endpoints.

    Parameters / Returns: same as hilbert_contour.
    """
    try:
        n_total = len(mono)
        PAD_S   = 0.005
        pad_n   = int(PAD_S * sr)

        i0_call = int(round(t0_rel * sr))
        i1_call = int(round(t1_rel * sr))
        i0      = max(0, i0_call - pad_n)
        i1      = min(n_total, i1_call + pad_n)

        seg = mono[i0:i1].astype(np.float64)
        if len(seg) < 20:
            return None

        # ── Bandpass ──────────────────────────────────────────────────────────
        flo = max(low_hz  * 0.75, 500.0)
        fhi = min(high_hz * 1.25, sr * 0.49)
        if flo >= fhi * 0.9:
            return None

        sos      = _signal.butter(5, [flo, fhi], btype='bandpass',
                                  fs=sr, output='sos')
        filtered = _signal.sosfiltfilt(sos, seg)

        # ── Build linear-FM reference ─────────────────────────────────────────
        dur_s = t1_rel - t0_rel
        if dur_s < 1e-4:
            return None

        # Bat FM sweeps descend: starts at high_hz, ends at low_hz
        f_start    = float(high_hz)
        f_end      = float(low_hz)
        chirp_rate = (f_end - f_start) / dur_s          # Hz/s (negative)

        # Time axis relative to call start for every sample in seg
        n_call_start = i0_call - i0                      # index of call start in seg
        t_rel        = (np.arange(len(seg)) - n_call_start) / sr  # s from call start

        # Reference instantaneous phase: φ(t) = 2π[f_start·t + ½·chirp_rate·t²]
        phi_ref = 2.0 * np.pi * (f_start * t_rel + 0.5 * chirp_rate * t_rel ** 2)

        # ── Demodulate and extract IF ─────────────────────────────────────────
        z       = _signal.hilbert(filtered)
        z_demod = z * np.exp(-1j * phi_ref)

        IF_residual = (np.angle(z_demod[1:] * np.conj(z_demod[:-1]))
                       * (sr / (2.0 * np.pi)))

        # Reference IF at the midpoint between each consecutive sample pair
        t_mid     = t_rel[:-1] + 0.5 / sr
        f_ref_mid = f_start + chirp_rate * t_mid

        IF_raw = f_ref_mid + IF_residual

        # ── Trim to call region ───────────────────────────────────────────────
        c0      = max(0, i0_call - i0)
        c1      = max(c0 + 1, min(i1_call - i0, len(IF_raw)))
        IF_call = IF_raw[c0:c1]

        if len(IF_call) < 4:
            return None

        # ── Clamp residual to ±50 % of predicted bandwidth ───────────────────
        # If the detector's Fmin/Fmax is reasonable, the true IF should deviate
        # from the linear reference by at most half the sweep range.  Hard-
        # clamping larger residuals suppresses harmonic artefacts and noise
        # before the smoothing pass.
        bw50    = (high_hz - low_hz) * 0.5
        IF_call = np.clip(IF_call, low_hz - bw50, high_hz + bw50)

        # ── Amplitude gating ──────────────────────────────────────────────────
        amp_call = np.abs(z)[c0:c1]      # z computed before demodulation
        IF_call  = _amplitude_gate(IF_call, amp_call)

        # ── Adaptive smoothing ────────────────────────────────────────────────
        IF_call = _smooth_if_adaptive(IF_call, sr, low_hz, high_hz)

        # ── Subsample to n_pts ────────────────────────────────────────────────
        n     = len(IF_call)
        n_out = min(n, n_pts)
        idx   = np.round(np.linspace(0, n - 1, n_out)).astype(int)
        t_abs = chunk_t0_s + t0_rel + idx / sr
        fc_hz = IF_call[idx]

        Fmin_k = float(fc_hz.min()) / 1000.0
        Fmax_k = float(fc_hz.max()) / 1000.0
        dur_ms = dur_s * 1000.0
        tms    = np.linspace(0.0, dur_ms, n_out)
        sweep  = (abs(float(np.polyfit(tms, fc_hz / 1000.0, 1)[0]))
                  if n_out > 2 else 0.0)

        contour = [[float(t), float(f / 1000.0)]
                   for t, f in zip(t_abs, fc_hz)]
        return contour, fc_hz, Fmin_k, Fmax_k, sweep

    except Exception:
        return None


# ---------------------------------------------------------------------------
# CWT scalogram (on-demand, per selected call)
# ---------------------------------------------------------------------------

def cwt_scalogram(mono, sr, t0_rel, t1_rel, low_hz, high_hz,
                  n_scales=64, n_pts=200):
    """
    Compute a CWT scalogram for a call segment and return it as PNG bytes.

    Only the call region is analysed — feasible even for long recordings
    because we process at most a few hundred ms of audio per call.  At
    192 kHz a 20 ms call × 64 scales completes in ~5 ms.

    Frequency axis is log-spaced from low_hz (bottom row) to high_hz (top
    row).  Each column spans roughly (t1_rel - t0_rel) / n_pts seconds.

    Parameters
    ----------
    mono      : 1-D float64 array — mono audio for the containing chunk
    sr        : sample rate (Hz)
    t0_rel    : call start time relative to chunk start (s)
    t1_rel    : call end time relative to chunk start (s)
    low_hz    : expected Fmin from detector (Hz)
    high_hz   : expected Fmax from detector (Hz)
    n_scales  : number of CWT scales (frequency rows in output)
    n_pts     : number of time columns in output

    Returns
    -------
    (t_arr, f_arr_khz, png_bytes)
      t_arr      : (n_pts,) time axis in seconds (chunk-relative)
      f_arr_khz  : (n_scales,) frequency axis, low→high, in kHz
      png_bytes  : PNG-encoded RGB image (hot colormap), n_scales × n_pts px
    Returns None on failure.
    """
    try:
        import io
        from PIL import Image

        n_total = len(mono)
        PAD_S   = 0.005
        pad_n   = int(PAD_S * sr)

        i0_call = int(round(t0_rel * sr))
        i1_call = int(round(t1_rel * sr))
        i0      = max(0, i0_call - pad_n)
        i1      = min(n_total, i1_call + pad_n)

        seg = mono[i0:i1].astype(np.float64)
        if len(seg) < 20:
            return None

        # Bandpass filter to the call band
        flo_bp = max(low_hz  * 0.70, 500.0)
        fhi_bp = min(high_hz * 1.30, sr * 0.45)
        if flo_bp >= fhi_bp * 0.9:
            return None
        sos      = _signal.butter(5, [flo_bp, fhi_bp], btype='bandpass',
                                  fs=sr, output='sos')
        filtered = _signal.sosfiltfilt(sos, seg)

        # Trim to call region (remove filter-settling padding)
        c0         = max(0, i0_call - i0)
        c1         = min(len(filtered), i1_call - i0)
        call_audio = filtered[c0:c1]
        if len(call_audio) < 10:
            return None

        # Log-spaced frequency axis: row 0 = highest freq (image top)
        flo_sc = max(low_hz  * 0.80, 1_000.0)
        fhi_sc = min(high_hz * 1.20, sr * 0.45)
        f_arr_hz_desc = np.exp(np.linspace(np.log(fhi_sc), np.log(flo_sc), n_scales))

        # Morlet CWT
        w0     = 6.0
        scales = w0 * sr / (2.0 * np.pi * f_arr_hz_desc)
        cwtm   = _signal.cwt(call_audio, _signal.morlet2, scales, w=w0)
        mag    = np.abs(cwtm)   # (n_scales, len(call_audio))

        # Resample to exactly n_pts columns (linear interpolation)
        src_cols = mag.shape[1]
        if src_cols != n_pts:
            col_idx = np.linspace(0, src_cols - 1, n_pts)
            lo_idx  = np.floor(col_idx).astype(int)
            hi_idx  = np.minimum(lo_idx + 1, src_cols - 1)
            alpha   = (col_idx - lo_idx)[None, :]
            mag     = (1 - alpha) * mag[:, lo_idx] + alpha * mag[:, hi_idx]

        # Normalise: 1st/99th percentile
        p01, p99 = np.percentile(mag, [1, 99])
        mag_norm = np.clip((mag - p01) / max(p99 - p01, 1e-10), 0.0, 1.0)

        # Hot colormap: black → red → yellow → white
        r = np.clip(mag_norm * 3.0,       0.0, 1.0)
        g = np.clip(mag_norm * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(mag_norm * 3.0 - 2.0, 0.0, 1.0)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        # row 0 = highest freq (top of image)

        img = Image.fromarray(rgb, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')

        t_arr     = t0_rel + np.linspace(0.0, t1_rel - t0_rel, n_pts)
        f_arr_khz = f_arr_hz_desc[::-1] / 1000.0   # ascending (bottom→top)

        return t_arr, f_arr_khz, buf.getvalue()

    except Exception:
        return None
