"""Backtest scoring: Brier score, log loss, and calibration tables (plan section 8).

Every function here takes ``(probability, outcome)`` pairs -- the model's own
claimed P(event), and whether the event actually happened -- and nothing
else. None of it knows about ridge, margins, or games; that separation is
deliberate, so the same functions score any future model (Elo, logistic,
totals) without duplication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Prediction = tuple[float, bool]

_LOGLOSS_EPS = 1e-12
_Z_95 = 1.959963985


def brier_score(predictions: list[Prediction]) -> float:
    """Mean squared error between claimed probability and the 0/1 outcome."""
    if not predictions:
        raise ValueError("no predictions to score")
    return sum((p - (1.0 if won else 0.0)) ** 2 for p, won in predictions) / len(predictions)


def log_loss(predictions: list[Prediction]) -> float:
    """Mean negative log-likelihood, clipped away from exact 0/1.

    A single confident-and-wrong prediction (p=1.0 on a loss) would otherwise
    make this infinite; clipping to ``_LOGLOSS_EPS`` keeps one bad game from
    making the whole backtest unreportable.
    """
    if not predictions:
        raise ValueError("no predictions to score")
    total = 0.0
    for p, won in predictions:
        clipped = min(max(p, _LOGLOSS_EPS), 1.0 - _LOGLOSS_EPS)
        total += -math.log(clipped) if won else -math.log(1.0 - clipped)
    return total / len(predictions)


def favorite_side(prob_home: float, home_won: bool) -> Prediction:
    """Re-express a game as (the favored side's win probability, did it win).

    A model's raw output is always "P(home wins)"; a parlay leg backs
    whichever side is actually favored, which is the away team exactly when
    ``prob_home < 0.5``. The confidence-bucket table (plan section 8) is
    about the reliability of the side actually being bet, not about home
    teams specifically.
    """
    if prob_home >= 0.5:
        return prob_home, home_won
    return 1.0 - prob_home, not home_won


@dataclass(frozen=True)
class BucketResult:
    label: str
    low: float
    high: float
    n: int
    win_rate: float | None
    wilson_low: float | None
    wilson_high: float | None


def wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> tuple[float, float]:
    """95%-by-default Wilson score interval for a binomial proportion.

    Preferred over a naive normal approximation at small n or extreme p --
    exactly the regime a single confidence bucket often lands in.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))
    return center - half_width, center + half_width


def _bucket(predictions: list[Prediction], low: float, high: float, label: str) -> BucketResult:
    in_bucket = [(p, won) for p, won in predictions if low <= p < high]
    if not in_bucket:
        return BucketResult(label, low, high, 0, None, None, None)
    wins = sum(1 for _, won in in_bucket if won)
    win_rate = wins / len(in_bucket)
    wilson_low, wilson_high = wilson_interval(wins, len(in_bucket))
    return BucketResult(label, low, high, len(in_bucket), win_rate, wilson_low, wilson_high)


# The plan's specified high-confidence bands (section 8): is a pick the model
# calls "80%+ likely" actually landing anywhere near that rate out of sample?
# The upper bound of the last bucket is nudged past 1.0 so a probability of
# exactly 1.0 is still captured by a half-open [low, high) range.
CONFIDENCE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.800, 0.849, "80-84.9%"),
    (0.850, 0.899, "85-89.9%"),
    (0.900, 0.924, "90-92.4%"),
    (0.925, 0.949, "92.5-94.9%"),
    (0.950, 1.001, "95%+"),
)


def bucketed_win_rate(
    predictions: list[Prediction],
    buckets: tuple[tuple[float, float, str], ...] = CONFIDENCE_BUCKETS,
) -> list[BucketResult]:
    return [_bucket(predictions, low, high, label) for low, high, label in buckets]


def reliability_curve(
    predictions: list[Prediction], *, bucket_width: float = 0.1
) -> list[BucketResult]:
    """A general 0-100% calibration curve, in ``bucket_width``-wide bands.

    Where ``bucketed_win_rate`` only asks "are our confident picks right,"
    this is the full picture -- including the near-toss-up games where most
    of a season's games actually land.
    """
    n_buckets = round(1.0 / bucket_width)
    buckets = []
    for i in range(n_buckets):
        low = i * bucket_width
        high = 1.001 if i == n_buckets - 1 else (i + 1) * bucket_width
        buckets.append((low, high, f"{low:.0%}-{min(high, 1.0):.0%}"))
    return bucketed_win_rate(predictions, tuple(buckets))
