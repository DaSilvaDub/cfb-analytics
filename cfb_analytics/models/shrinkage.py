"""Early-season shrinkage prior for ridge ratings (plan section 6.1).

    O_prior = a * O_prev_season_final + b * z(talent) + c * z(returning_ppa)
    O_final = (n * O_fit + k * O_prior) / (n + k),    k ~ 4 team-games

Applied identically to offense and defense: the plan only writes the
offense version, and there is no stated reason to treat D differently, so
the same coefficients and formula are reused for it (see
``features/team_ratings.py``, where both actually get computed).

This module is pure math with no database access, matching ``models/ridge.py``
and ``models/linalg.py`` -- gathering the actual inputs (a team's previous
season rating, its talent composite, its returning-production percentage)
is ``features/team_ratings.py``'s job, exactly like the ridge/DB split
already established for the base model.

Missing inputs (no previous season on file, no talent or returning-
production data -- true for every FCS-to-FBS transition team, and for every
2014 team since this store's history starts there) contribute zero to
``O_prior`` rather than raising: a smaller-magnitude prior is the honest
response to missing context, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fitted 2026-09-06 by grid search (a in [0, 3], b in [0, 7], c in [0, 5] at
# 0.5-1.0 steps) against 2021 weeks 1-5 out-of-sample margin RMSE, one-off,
# not shipped as a runnable calibration tool -- not hand-picked. `k` is the
# plan's own stated value ("~4 team-games"), not fitted; grid-searching it
# too would need games-vs-margin data this store does not cleanly separate
# from the (a, b, c) search itself.
#
# The search RMSE improved monotonically from a=b=c=0 (23.49) out to roughly
# a=2.5 (21.32), then got WORSE beyond a=3 (22.24) -- a real elbow, not just
# a grid edge. Chosen values sit inside that plateau rather than at its
# fragile edge, and were checked against a held-out season (2019 weeks 1-5,
# never used in the search): RMSE improved there too, from 24.33 (no prior)
# to 20.92 at these settings -- generalizing, not overfitting the fit season.
DEFAULT_A = 2.0
DEFAULT_B = 4.0
DEFAULT_C = 1.5
DEFAULT_K = 4.0


@dataclass(frozen=True)
class ShrinkageCoefficients:
    a: float = DEFAULT_A
    b: float = DEFAULT_B
    c: float = DEFAULT_C
    k: float = DEFAULT_K


DEFAULT_COEFFICIENTS = ShrinkageCoefficients()


def prior_rating(
    prev_season_final: float | None,
    z_talent: float | None,
    z_returning_ppa: float | None,
    *,
    coeffs: ShrinkageCoefficients = DEFAULT_COEFFICIENTS,
) -> float:
    """O_prior (or D_prior): a blend of last season's rating and this
    season's preseason-knowable talent/returning-production signals."""
    return (
        coeffs.a * (prev_season_final or 0.0)
        + coeffs.b * (z_talent or 0.0)
        + coeffs.c * (z_returning_ppa or 0.0)
    )


def blend_toward_prior(fit_value: float, prior_value: float, n_games: int, *, k: float) -> float:
    """O_final (or D_final): fit_value weighted by games played so far this
    season, prior_value weighted by ``k`` -- a team with zero games this
    season is entirely the prior; one with n >> k is almost entirely its
    own in-season fit."""
    if n_games < 0:
        raise ValueError("n_games cannot be negative")
    if k < 0:
        raise ValueError("k cannot be negative")
    return (n_games * fit_value + k * prior_value) / (n_games + k)


def zscore(values: dict[str, float]) -> dict[str, float]:
    """Population z-score of a ``{team_id: value}`` map.

    A team simply absent from ``values`` (no talent/returning-production row
    at all) has no entry in the result either -- the caller treats a missing
    key as "no data" and contributes 0 to the prior, never a fabricated
    league-average value. Fewer than 2 distinct values, or zero variance
    (every team tied), returns 0.0 for everyone rather than dividing by zero.
    """
    if len(values) < 2:
        return dict.fromkeys(values, 0.0)
    mean = sum(values.values()) / len(values)
    variance = sum((v - mean) ** 2 for v in values.values()) / len(values)
    stdev = variance**0.5
    if stdev == 0.0:
        return dict.fromkeys(values, 0.0)
    return {team: (v - mean) / stdev for team, v in values.items()}
