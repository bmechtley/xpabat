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
    win = np.hanning(nperseg).astype(float)

    # Derivative window: central finite difference, per-sample units
    # d_win[n] ≈ (win[n+1] − win[n−1]) / 2  (units: amplitude / sample)
    d_win = np.zeros(nperseg)
    d_win[1:-1] = (win[2:] - win[:-2]) / 2.0
    # Convert to per-second by multiplying by sample rate
    d_win *= sr

    f, t, Z  = _signal.stft(audio, fs=sr, window=win,
                              nperseg=nperseg, noverlap=noverlap)
    _, _, Zd = _signal.stft(audio, fs=sr, window=d_win,
                              nperseg=nperseg, noverlap=noverlap)

    # Avoid division by near-zero
    mag     = np.abs(Z)
    safe_Z  = np.where(mag > 1e-10, Z, 1.0 + 0j)

    # Instantaneous frequency (Hz), clipped to [0, Nyquist]
    IF = f[:, None] + np.imag(Zd / safe_Z) / (2.0 * np.pi)
    IF = np.clip(IF, 0.0, sr / 2.0)

    S = mag ** 2
    return f, t, S, IF


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
    from scipy.ndimage import median_filter
    try:
        from scipy.signal import savgol_filter
        _have_savgol = True
    except ImportError:
        _have_savgol = False

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

        # ── Clip outliers to ±50% of expected range ───────────────────────
        IF_call = np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)

        # ── Smooth 1: median (removes phase-wrap spikes) ──────────────────
        n       = len(IF_call)
        n_med   = max(3, min(31, n // 20) | 1)      # odd, ≤5% of call length
        IF_call = median_filter(IF_call, size=n_med)

        # ── Smooth 2: Savitzky-Golay (preserves FM shape) ─────────────────
        if _have_savgol and n >= 7:
            wl = max(5, min(51, n // 10))
            if wl % 2 == 0:
                wl += 1
            wl = min(wl, n - (1 if n % 2 == 0 else 0))
            if wl >= 5 and wl < n:
                IF_call = savgol_filter(IF_call, window_length=wl,
                                        polyorder=min(3, wl - 2))

        # Final clip after smoothing
        IF_call = np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)

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
    from scipy.ndimage import median_filter
    try:
        from scipy.signal import savgol_filter
        _have_savgol = True
    except ImportError:
        _have_savgol = False

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

        # ── Clip outliers ─────────────────────────────────────────────────────
        IF_call = np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)

        # ── Smooth ────────────────────────────────────────────────────────────
        n     = len(IF_call)
        n_med = max(3, min(31, n // 20) | 1)
        IF_call = median_filter(IF_call, size=n_med)

        if _have_savgol and n >= 7:
            wl = max(5, min(51, n // 10))
            if wl % 2 == 0: wl += 1
            wl = min(wl, n - (1 if n % 2 == 0 else 0))
            if wl >= 5 and wl < n:
                IF_call = savgol_filter(IF_call, window_length=wl,
                                        polyorder=min(3, wl - 2))

        IF_call = np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)

        # ── Subsample to n_pts ────────────────────────────────────────────────
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
    from scipy.ndimage import median_filter
    try:
        from scipy.signal import savgol_filter
        _have_savgol = True
    except ImportError:
        _have_savgol = False

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

        # ── Clip, smooth ──────────────────────────────────────────────────────
        IF_call = np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)

        n     = len(IF_call)
        n_med = max(3, min(31, n // 20) | 1)
        IF_call = median_filter(IF_call, size=n_med)

        if _have_savgol and n >= 7:
            wl = max(5, min(51, n // 10))
            if wl % 2 == 0: wl += 1
            wl = min(wl, n - (1 if n % 2 == 0 else 0))
            if wl >= 5 and wl < n:
                IF_call = savgol_filter(IF_call, window_length=wl,
                                        polyorder=min(3, wl - 2))

        IF_call = np.clip(IF_call, low_hz * 0.5, high_hz * 2.0)

        # ── Subsample to n_pts ────────────────────────────────────────────────
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
