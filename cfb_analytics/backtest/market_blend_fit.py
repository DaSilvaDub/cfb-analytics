"""Walk-forward market blend weight fit for promote evidence.

For each evaluation season S, choose ``w`` in ``[floor, 1]`` that minimises
log loss of ``blend_with_market(p_market, p_model, w)`` on **prior seasons
only**. Season with no prior labeled overlap uses ``w = floor`` (fail-closed
toward discipline, not toward pure market — pure market is reported as the
w=1 bracket separately).

This mirrors ``prob_calibrate.walk_forward_calibrate``'s season-blocked shape:
never fit on the evaluation slice.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfb_analytics.backtest.metrics import Prediction, log_loss
from cfb_analytics.models.market_blend import (
    DEFAULT_MARKET_WEIGHT_FLOOR,
    blend_with_market,
    clamp_market_weight,
)

DEFAULT_WEIGHT_GRID_STEP = 0.05


@dataclass(frozen=True)
class BlendRow:
    """One overlap game for blend scoring / weight fit."""

    season: int
    p_market: float
    p_model: float
    won: bool
    key: str = ""


@dataclass(frozen=True)
class BlendWalkForwardResult:
    """Per-game blended probs + the w applied to each season."""

    blended: list[Prediction]  # (p_blend, won) in same order as rows
    weight_by_season: dict[int, float]
    floor: float
    grid_step: float


def _weight_grid(floor: float, step: float) -> list[float]:
    """Inclusive grid from floor to 1.0 at ``step`` (always includes 1.0)."""
    floor_c = clamp_market_weight(floor, floor=floor)
    if step <= 0 or step > 1:
        raise ValueError(f"grid_step must be in (0, 1], got {step!r}")
    weights: list[float] = []
    # Start at floor; step up; always end with 1.0.
    k = 0
    while True:
        w = round(floor_c + k * step, 10)
        if w >= 1.0 - 1e-12:
            break
        if w >= floor_c - 1e-12:
            weights.append(float(w))
        k += 1
        if k > 10_000:
            raise ValueError("weight grid did not terminate")
    weights.append(1.0)
    # Dedup while preserving order
    out: list[float] = []
    for w in weights:
        if not out or abs(out[-1] - w) > 1e-12:
            out.append(w)
    return out


def fit_market_weight(
    rows: list[BlendRow],
    *,
    floor: float = DEFAULT_MARKET_WEIGHT_FLOOR,
    grid_step: float = DEFAULT_WEIGHT_GRID_STEP,
) -> float:
    """Grid-search ``w`` in ``[floor, 1]`` minimising log loss on ``rows``."""
    if not rows:
        raise ValueError("no rows to fit market blend weight on")
    best_w = floor
    best_loss: float | None = None
    for w in _weight_grid(floor, grid_step):
        preds: list[Prediction] = [
            (blend_with_market(r.p_market, r.p_model, w, floor=floor), r.won)
            for r in rows
        ]
        loss = log_loss(preds)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_w = w
    return float(best_w)


def walk_forward_market_blend(
    rows: list[BlendRow],
    *,
    floor: float = DEFAULT_MARKET_WEIGHT_FLOOR,
    grid_step: float = DEFAULT_WEIGHT_GRID_STEP,
) -> BlendWalkForwardResult:
    """Season-blocked expanding-window blend: fit w on prior seasons, apply to S.

    Rows may be in any order; output ``blended`` follows input order.
    """
    if not rows:
        return BlendWalkForwardResult(
            blended=[], weight_by_season={}, floor=floor, grid_step=grid_step
        )

    seasons = sorted({r.season for r in rows})
    by_season: dict[int, list[BlendRow]] = {s: [] for s in seasons}
    for r in rows:
        by_season[r.season].append(r)

    weight_by_season: dict[int, float] = {}
    prior: list[BlendRow] = []
    for season in seasons:
        if prior:
            weight_by_season[season] = fit_market_weight(
                prior, floor=floor, grid_step=grid_step
            )
        else:
            # No prior overlap: fail-closed to floor (max model share under discipline).
            weight_by_season[season] = float(floor)
        prior.extend(by_season[season])

    blended: list[Prediction] = [
        (
            blend_with_market(
                r.p_market,
                r.p_model,
                weight_by_season[r.season],
                floor=floor,
            ),
            r.won,
        )
        for r in rows
    ]
    return BlendWalkForwardResult(
        blended=blended,
        weight_by_season=weight_by_season,
        floor=float(floor),
        grid_step=float(grid_step),
    )


def score_fixed_weight(
    rows: list[BlendRow],
    w: float,
    *,
    floor: float = DEFAULT_MARKET_WEIGHT_FLOOR,
) -> list[Prediction]:
    """Apply a single clamped weight to every row (bracket helpers)."""
    return [
        (blend_with_market(r.p_market, r.p_model, w, floor=floor), r.won)
        for r in rows
    ]
