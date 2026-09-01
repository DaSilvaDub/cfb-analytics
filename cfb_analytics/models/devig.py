"""Remove the bookmaker's margin from a set of quoted prices.

Three methods, all pure stdlib. They disagree most where this pipeline cares
most -- on heavy favourites, where the choice moves the fair probability by
1-3 percentage points, which is the difference between a CORE leg and an AVOID.
So all three are computed and stored, and the backtest picks the winner on
out-of-sample log loss rather than the author picking one on taste.

* **multiplicative** (proportional): p_i = q_i / sum(q). Simple, and assumes the
  margin is spread proportionally. Known to under-price favourites, because in
  practice books load more margin onto longshots.
* **shin**: models the margin as compensation for informed bettors. Usually
  lands between multiplicative and power, and is the most defensible default for
  two-outcome markets.
* **power**: p_i = q_i^k with k solved so the probabilities sum to 1. Applies the
  most correction to longshots.

Both Shin and power solve by bisection: robust, derivative-free, and the
monotonicity that makes bisection valid is argued at each call site.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cfb_analytics.errors import DevigError

METHODS = ("multiplicative", "shin", "power")
_TOLERANCE = 1e-12
_MAX_ITERATIONS = 200


def overround(quoted: Sequence[float]) -> float:
    """Sum of quoted (vig-inclusive) probabilities. 1.0 means no margin."""
    return math.fsum(quoted)


def hold(quoted: Sequence[float]) -> float:
    """The book's margin as a fraction of the total market. Negative = arb."""
    total = overround(quoted)
    if total <= 0:
        raise DevigError("Quoted probabilities must sum to a positive number")
    return (total - 1.0) / total


def _validate(quoted: Sequence[float]) -> list[float]:
    values = [float(q) for q in quoted]
    if len(values) < 2:
        raise DevigError("Devigging needs at least two mutually exclusive outcomes")
    if any(q <= 0.0 for q in values):
        raise DevigError("Quoted probabilities must all be positive")
    if any(q >= 1.0 for q in values):
        # A single quote at or above 1.0 implies a certainty; the market is
        # malformed or the price was misparsed. Refuse rather than emit a
        # confident-looking number.
        raise DevigError("A quoted probability was >= 1.0, which is not a real price")
    return values


def multiplicative(quoted: Sequence[float]) -> list[float]:
    values = _validate(quoted)
    total = math.fsum(values)
    return [q / total for q in values]


def power(quoted: Sequence[float]) -> list[float]:
    """Solve sum(q_i^k) = 1 for k > 0.

    sum(q_i^k) is strictly decreasing in k for 0 < q_i < 1, so the root is
    unique and bisection is safe.
    """
    values = _validate(quoted)

    def total_at(k: float) -> float:
        return math.fsum(q**k for q in values)

    low, high = 1e-6, 1.0
    # Overround > 1 means we need k > 1 to shrink the sum; expand the bracket.
    while total_at(high) > 1.0:
        high *= 2.0
        if high > 1e6:
            raise DevigError("Power devig failed to bracket a solution")
    while total_at(low) < 1.0:
        low /= 2.0
        if low < 1e-12:
            raise DevigError("Power devig failed to bracket a solution")

    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        if total_at(mid) > 1.0:
            low = mid
        else:
            high = mid
        if high - low < _TOLERANCE:
            break
    k = (low + high) / 2.0
    result = [q**k for q in values]
    return _renormalise(result)


def shin(quoted: Sequence[float]) -> list[float]:
    """Shin (1993): margin as compensation for a proportion z of informed money.

    p_i = [sqrt(z^2 + 4(1-z) q_i^2 / S) - z] / (2(1-z)),  S = sum(q)

    sum(p_i) is monotonic in z over [0, 1), so bisection finds the unique z that
    makes the fair probabilities sum to 1.
    """
    values = _validate(quoted)
    total = math.fsum(values)
    if abs(total - 1.0) < _TOLERANCE:
        return list(values)
    if total < 1.0:
        # Negative margin: no informed-money interpretation exists. Fall back
        # rather than invent a z, and let the caller see a normalised book.
        return multiplicative(values)

    def implied_sum(z: float) -> float:
        if z >= 1.0:
            return float("inf")
        return math.fsum(
            (math.sqrt(z * z + 4.0 * (1.0 - z) * q * q / total) - z) / (2.0 * (1.0 - z))
            for q in values
        )

    low, high = 0.0, 0.999999
    if implied_sum(low) < 1.0:
        return multiplicative(values)
    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        if implied_sum(mid) > 1.0:
            low = mid
        else:
            high = mid
        if high - low < _TOLERANCE:
            break
    z = (low + high) / 2.0
    result = [
        (math.sqrt(z * z + 4.0 * (1.0 - z) * q * q / total) - z) / (2.0 * (1.0 - z))
        for q in values
    ]
    return _renormalise(result)


def _renormalise(values: list[float]) -> list[float]:
    """Absorb bisection residue so the result sums to exactly 1."""
    total = math.fsum(values)
    if total <= 0:
        raise DevigError("Devig produced a non-positive total")
    return [v / total for v in values]


def devig(quoted: Sequence[float], method: str = "multiplicative") -> list[float]:
    if method not in METHODS:
        raise DevigError(f"Unknown devig method {method!r}; expected one of {METHODS}")
    return {"multiplicative": multiplicative, "shin": shin, "power": power}[method](quoted)


def devig_all(quoted: Sequence[float]) -> dict[str, list[float]]:
    """Every method at once, so the store keeps all three for the backtest."""
    return {method: devig(quoted, method) for method in METHODS}
