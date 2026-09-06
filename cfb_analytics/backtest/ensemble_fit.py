"""Fits the ensemble pooling weights (plan section 6.4):

    v fitted by coordinate search minimising walk-forward out-of-sample log loss

A calibration concern, not a math one -- ``models/ensemble.py`` has the pure
blending formula; this module searches for the weights that make it work
best against real outcomes, the same relationship ``backtest/calibration.py``
has to ``models/ridge.py``'s margin-to-probability conversion.

"Coordinate search" here is an exhaustive grid over the weight simplex at a
fixed step size (0.05 by default: 231 combinations for 3 members), not a
continuous optimizer -- deliberately, since a 3-member simplex at that
resolution is cheap to search exhaustively and exhaustive search cannot get
stuck in a bad local optimum the way true coordinate ASCENT can.
"""

from __future__ import annotations

from collections.abc import Iterator

from cfb_analytics.backtest.metrics import Prediction, log_loss
from cfb_analytics.models.ensemble import pool_probabilities

DEFAULT_GRID_STEP = 0.05


def _simplex_grid(n_members: int, step: float) -> Iterator[tuple[float, ...]]:
    """Every ``n_members``-tuple of non-negative multiples of ``step`` that
    sums to exactly 1 (up to floating point)."""
    steps = round(1.0 / step)

    def _recurse(remaining_slots: int, remaining_steps: int) -> Iterator[tuple[float, ...]]:
        if remaining_slots == 1:
            yield (remaining_steps * step,)
            return
        for i in range(remaining_steps + 1):
            for rest in _recurse(remaining_slots - 1, remaining_steps - i):
                yield (i * step, *rest)

    yield from _recurse(n_members, steps)


def fit_ensemble_weights(
    per_game_member_probs: list[dict[str, float]],
    outcomes: list[bool],
    member_names: tuple[str, ...],
    *,
    grid_step: float = DEFAULT_GRID_STEP,
) -> dict[str, float]:
    """The ``member_names`` weighting (summing to 1, plan's simplex
    constraint) that minimizes log loss on ``(per_game_member_probs,
    outcomes)``.

    A game missing one member's prediction (e.g. internal Elo's own
    insufficient-history weeks) still contributes, scored on whichever
    members it does have -- ``pool_probabilities`` renormalizes among those,
    exactly as it does for a live prediction with a missing member.
    """
    if len(per_game_member_probs) != len(outcomes):
        raise ValueError("per_game_member_probs and outcomes must be the same length")
    if not per_game_member_probs:
        raise ValueError("no games to fit ensemble weights on")

    best_weights: dict[str, float] | None = None
    best_loss: float | None = None
    for combo in _simplex_grid(len(member_names), grid_step):
        weights = dict(zip(member_names, combo, strict=True))
        predictions: list[Prediction] = []
        for probs, outcome in zip(per_game_member_probs, outcomes, strict=True):
            available = {name: w for name, w in weights.items() if w > 0 and name in probs}
            if not available:
                continue
            predictions.append((pool_probabilities(probs, available), outcome))
        if not predictions:
            continue
        loss = log_loss(predictions)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_weights = weights

    if best_weights is None:
        raise ValueError("no games had any weighted member's prediction available")
    return best_weights
