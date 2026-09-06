"""Elo-only win probability -- plan section 8's baseline #3.

    E = 1 / (1 + 10^(-(R_home - R_away + HFA) / 400))

The classic Elo logistic formula, applied to CFBD's own weekly Elo ratings
(``backfill-elo``) rather than the plan's own not-yet-built internal Elo
model (section 6.2's ``K_eff``/preseason-blend formula). Unlike the ridge
model's margin, this is already a probability by construction -- no
separate sigma-fitting step is needed, unlike ``backtest/calibration.py``.

``HFA_ELO_POINTS = 60`` is not invented for this baseline: it is the exact
home-field constant plan section 6.2 documents for Elo (~2.6 points on the
scoring-margin scale), reused here so this baseline stays traceable to the
plan rather than introducing a second, unrelated home-field number.
"""

from __future__ import annotations

HFA_ELO_POINTS = 60.0


def elo_win_probability(
    rating_home: float,
    rating_away: float,
    *,
    neutral_site: bool = False,
    hfa: float = HFA_ELO_POINTS,
) -> float:
    bonus = 0.0 if neutral_site else hfa
    return 1.0 / (1.0 + 10.0 ** (-(rating_home - rating_away + bonus) / 400.0))
