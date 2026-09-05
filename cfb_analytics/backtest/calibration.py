"""Margin-to-probability conversion and sigma calibration (plan section 6.3).

    P_home = Phi(M / sigma_M)

``Phi`` is the standard normal CDF, computed via ``math.erf`` -- pure
stdlib, no ``scipy.stats.norm`` (repo-wide constraint).

The plan's full form is ``sigma_M = sigma_0 + beta * (proj_total - mean_total)``:
a game projected to be higher-scoring than average gets a wider outcome
distribution, and margin is therefore a noisier signal. ``beta``'s input is a
projected game total, which needs the totals model -- not yet built. This
module implements ``sigma_M = sigma_0`` (a single fitted constant) as a
first cut. That is a real, tracked gap, not a silent simplification: once a
totals projection exists, the beta term should be added back and sigma_0
recalibrated against the residual it leaves behind.
"""

from __future__ import annotations

import math


def margin_to_prob(margin: float, sigma: float) -> float:
    """P(home wins) implied by a projected home-minus-away point margin.

    ``sigma`` is a point-spread standard deviation and must be positive; the
    caller (``calibrate_sigma``) is responsible for fitting it from historical
    residuals rather than assuming a number.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    return 0.5 * (1.0 + math.erf(margin / (sigma * math.sqrt(2.0))))


def calibrate_sigma(residuals: list[float]) -> float:
    """Population standard deviation of (actual_margin - predicted_margin).

    This is ``sigma_0``, fitted empirically from walk-forward residuals
    rather than assumed. Raises rather than fabricating a number when there
    is nothing real to calibrate from.
    """
    if len(residuals) < 2:
        raise ValueError("need at least 2 residuals to calibrate sigma")
    mean = sum(residuals) / len(residuals)
    variance = sum((r - mean) ** 2 for r in residuals) / (len(residuals) - 1)
    return math.sqrt(variance)
