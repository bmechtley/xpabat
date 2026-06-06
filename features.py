"""
Per-call acoustic features computed from a call's raw audio segment.
"""

import numpy as np


def ar2_coeffs(x):
    """Second-order autoregressive (AR(2)) coefficients of a 1-D signal.

    Fits  x[n] ≈ a1·x[n-1] + a2·x[n-2]  by the Yule-Walker equations
    (autocorrelation method).  For a lightly-damped resonance the pair
    encodes the dominant spectral peak: a1 ≈ 2·r·cos(2π f / fs) tracks the
    centre frequency and a2 ≈ −r² its damping / bandwidth.  For a bat call
    (a narrowband FM sweep that dominates its own time window) this is a
    compact 2-number summary of the call's spectral shape.

    Returns (a1, a2) as plain floats; (0.0, 0.0) for degenerate input.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return 0.0, 0.0
    x = x - x.mean()
    r0 = float(np.dot(x, x) / n)
    if r0 <= 1e-20:
        return 0.0, 0.0
    r1 = float(np.dot(x[:-1], x[1:]) / n)
    r2 = float(np.dot(x[:-2], x[2:]) / n)
    denom = r0 * r0 - r1 * r1
    if abs(denom) < 1e-20:
        return 0.0, 0.0
    a1 = (r1 * r0 - r2 * r1) / denom
    a2 = (r2 * r0 - r1 * r1) / denom
    return float(a1), float(a2)
