"""
Per-call acoustic features computed from a call's raw audio segment.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Cache one high-pass filter per (sr, cutoff) so we don't redesign per call.
_HP_SOS = {}

def _highpass_sos(sr, cutoff_hz, order=6):
    key = (int(sr), int(cutoff_hz), order)
    sos = _HP_SOS.get(key)
    if sos is None:
        sos = butter(order, cutoff_hz / (sr * 0.5), btype="highpass", output="sos")
        _HP_SOS[key] = sos
    return sos


def ar_features_band(x, sr, cutoff_hz):
    """ar_features() after high-passing the segment to cutoff_hz, so the fit
    characterises the call's band (13–96 kHz) rather than low-frequency
    background that otherwise dominates the raw waveform amplitude."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size >= 32 and 0 < cutoff_hz < sr * 0.49:
        try:
            x = sosfiltfilt(_highpass_sos(sr, cutoff_hz), x)
        except Exception:
            pass
    return ar_features(x)


def ar_features(x):
    """Autoregressive spectral-shape features of a 1-D signal.

    From a single set of autocorrelations (r0, r1, r2) returns three numbers:

      a1, a2 — the second-order (AR(2)) Yule-Walker coefficients
               (x[n] ≈ a1·x[n-1] + a2·x[n-2]).  For a lightly-damped
               resonance a1 ≈ 2r·cos(2π f / fs) tracks the centre frequency
               and a2 ≈ −r² its damping / bandwidth.
      b1     — the first-order (AR(1)) coefficient r1/r0 (the lag-1 normalised
               autocorrelation): for a damped tone ≈ r·cos(2π f / fs), bounded
               to [-1, 1].

    Returns (a1, a2, b1) as plain floats; zeros for degenerate input.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return 0.0, 0.0, 0.0
    x = x - x.mean()
    r0 = float(np.dot(x, x) / n)
    if r0 <= 1e-20:
        return 0.0, 0.0, 0.0
    r1 = float(np.dot(x[:-1], x[1:]) / n)
    r2 = float(np.dot(x[:-2], x[2:]) / n)
    b1 = r1 / r0                                  # AR(1) coefficient
    denom = r0 * r0 - r1 * r1
    if abs(denom) < 1e-20:
        return 0.0, 0.0, float(b1)
    a1 = (r1 * r0 - r2 * r1) / denom
    a2 = (r2 * r0 - r1 * r1) / denom
    return float(a1), float(a2), float(b1)


def ar2_coeffs(x):
    """Back-compat: just the AR(2) pair (a1, a2)."""
    a1, a2, _ = ar_features(x)
    return a1, a2
