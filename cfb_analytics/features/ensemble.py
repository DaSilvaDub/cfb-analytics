"""Leakage-safe ``P_logit`` (plan section 6.4's third ensemble member).

The plan specifies an IRLS L2 logistic regression on the full T1-T5 feature
vector (~40 features spanning rating diffs, EPA/success-rate/havoc diffs, QB
experience, and rest/travel/rivalry/weather -- see the plan's section 5
tier table). This module implements a deliberately SMALLER slice of that:

    talent_diff          (T4: z-scored recruiting talent, home minus away)
    returning_prod_diff  (T4: z-scored returning-production PPA, home minus away)
    home_field_indicator (T1's hfa_venue, simplified to a plain 1/0 -- IRLS
                           fits its own coefficient for it, which is exactly
                           a league-average home-field effect in log-odds)

Deliberately NOT yet implemented, and tracked as a real gap rather than
silently narrowed: T2 (EPA/success-rate/explosiveness diffs -- computable
from the already-ingested ``team_season_advanced`` table, but needs its own
leakage-safe as-of aggregation, not yet built), T3 (line yards/havoc/QB
efficiency), and the rest of T5 (rest days, travel, rivalry, weather --
weather itself is already ingested but not yet wired into any feature).
Deliberately EXCLUDING ridge/Elo rating diffs from this feature set, even
though they are readily available and are T1 features per the plan: this
model is one of the three ensemble MEMBERS being blended alongside ridge
and internal Elo, so feeding their own outputs back in as P_logit's inputs
would make the "three independent members" pooled in ``models/ensemble.py``
partly the same signal counted twice.

Unlike ridge and internal Elo (both season-scoped: a team's rating resets
conceptually each year), P_logit's relationship between talent/returning-
production and win probability is a stable, cross-season pattern, so its
training set is every completed CFBD game across ALL seasons strictly
before the cutoff -- not reset per season. This is still fully leakage-safe
(``AsOfReader`` enforces the same "strictly before" rule globally, not just
within one season), and is actually a more natural fit for what IRLS needs:
a few thousand games to estimate three coefficients reliably, which even
one full season alone would strain.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.features.preseason import returning_ppa_zscores, talent_zscores
from cfb_analytics.models.logistic import DEFAULT_L2_LAMBDA, LogisticFit, fit_logistic

SOURCE = "cfbd"
FEATURE_NAMES = ("talent_diff", "returning_prod_diff", "home_field_indicator")


def game_features(
    home_team_id: str,
    away_team_id: str,
    *,
    neutral_site: bool,
    talent_z: dict[str, float],
    returning_z: dict[str, float],
) -> list[float]:
    """Always defined -- a team absent from the z-score maps (no talent or
    returning-production row at all) contributes 0 to its side of the diff,
    the same "missing means no signal, not a guess" convention
    ``models/shrinkage.py`` uses.
    """
    talent_diff = talent_z.get(home_team_id, 0.0) - talent_z.get(away_team_id, 0.0)
    returning_diff = returning_z.get(home_team_id, 0.0) - returning_z.get(away_team_id, 0.0)
    hfa_indicator = 0.0 if neutral_site else 1.0
    return [talent_diff, returning_diff, hfa_indicator]


def fit_logistic_as_of(
    conn: sqlite3.Connection,
    as_of_utc: str,
    *,
    l2_lambda: float = DEFAULT_L2_LAMBDA,
    min_n: int = 30,
    reader_game_id: str = "logistic-fit-lookup",
) -> LogisticFit:
    """Fit P_logit on every completed CFBD game, any season, strictly
    before ``as_of_utc``.

    Cheap regardless of how many seasons of history that spans: IRLS on 3
    features converges in a handful of iterations even over 10,000+ games,
    nothing like ridge's O(n^3) per-fit cost at FBS scale.
    """
    reader = AsOfReader(game_id=reader_game_id, kickoff_utc=as_of_utc, season=None)
    rows = conn.execute(
        """SELECT season, home_team_id, away_team_id, home_points, away_points,
                  neutral_site, kickoff_utc
           FROM games
           WHERE source = ? AND completed = 1
             AND home_points IS NOT NULL AND away_points IS NOT NULL""",
        (SOURCE,),
    ).fetchall()
    games = reader.admissible(
        [dict(row) for row in rows], what="games", as_of_field="kickoff_utc"
    )

    by_season: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for game in games:
        by_season[int(game["season"])].append(game)

    design_rows: list[list[float]] = []
    outcomes: list[bool] = []
    for season, season_games in by_season.items():
        talent_z = talent_zscores(conn, season)
        returning_z = returning_ppa_zscores(conn, season)
        for game in season_games:
            home_points = float(game["home_points"])
            away_points = float(game["away_points"])
            if home_points == away_points:
                continue  # no ties in the modern rulebook; a malformed row, not a result
            design_rows.append(game_features(
                game["home_team_id"], game["away_team_id"],
                neutral_site=bool(game["neutral_site"]),
                talent_z=talent_z, returning_z=returning_z,
            ))
            outcomes.append(home_points > away_points)

    return fit_logistic(
        design_rows, outcomes, feature_names=FEATURE_NAMES, l2_lambda=l2_lambda, min_n=min_n
    )
