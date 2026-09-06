"""Leakage-safe ``P_logit`` (plan section 6.4's third ensemble member).

The plan specifies an IRLS L2 logistic regression on the full T1-T5 feature
vector (~40 features spanning rating diffs, EPA/success-rate/havoc diffs, QB
experience, and rest/travel/rivalry/weather -- see the plan's section 5
tier table). This module implements a deliberately SMALLER slice of that:

    talent_diff                     (T4: z-scored recruiting talent)
    returning_prod_diff             (T4: z-scored returning-production PPA)
    home_field_indicator            (T1's hfa_venue, simplified to a plain
                                      1/0 -- IRLS fits its own coefficient,
                                      which is exactly a league-average
                                      home-field effect in log-odds)
    rest_diff                       (T5: days since each team's last game,
                                      home minus away -- see features/rest.py)
    ppa_net_diff                    (T2: offense PPA minus defense PPA
    success_rate_net_diff            allowed, home minus away, for each --
    explosiveness_net_diff           see features/advanced_stats.py for the
    points_per_opportunity_net_diff  full sign-convention writeup, since
    line_yards_net_diff              CFBD's own off/def field semantics are
    stuff_rate_net_diff              NOT symmetric for all seven stats)
    havoc_net_diff

Deliberately NOT yet implemented, and tracked as a real gap rather than
silently narrowed: T3's QB-specific features (``qb_epa_pp_diff``,
``qb_experience_diff``, ``qb_status_penalty``, ``backup_quality_gap`` --
``player_game_passing`` only has usable coverage for 2024 and 2026 in this
store, far too sparse for a cross-season pooled fit) and the rest of T5
(travel/rivalry/weather -- ``weather`` is ingested but only for live daily
slates, 2026 only, with no historical backfill to train or backtest against).
Deliberately EXCLUDING ridge/Elo rating diffs from this feature set, even
though they are readily available and are T1 features per the plan: this
model is one of the three ensemble MEMBERS being blended alongside ridge
and internal Elo, so feeding their own outputs back in as P_logit's inputs
would make the "three independent members" pooled in ``models/ensemble.py``
partly the same signal counted twice.

Unlike ridge and internal Elo (both season-scoped: a team's rating resets
conceptually each year), P_logit's relationship between these features and
win probability is a stable, cross-season pattern, so its training set is
every completed CFBD game across ALL seasons strictly before the cutoff --
not reset per season. This is still fully leakage-safe (``AsOfReader``
enforces the same "strictly before" rule globally, not just within one
season), and is actually a more natural fit for what IRLS needs: a few
thousand games to estimate eleven coefficients reliably, which even one
full season alone would strain.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from cfb_analytics.features.advanced_stats import (
    ADVANCED_STAT_KEYS,
    load_advanced_stats,
    team_advanced_nets,
)
from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.features.preseason import returning_ppa_zscores, talent_zscores
from cfb_analytics.features.rest import RestLookup
from cfb_analytics.models.logistic import DEFAULT_L2_LAMBDA, LogisticFit, fit_logistic

SOURCE = "cfbd"
FEATURE_NAMES = (
    "talent_diff",
    "returning_prod_diff",
    "home_field_indicator",
    "rest_diff",
    *(f"{key}_net_diff" for key in ADVANCED_STAT_KEYS),
)


def game_features(
    home_team_id: str,
    away_team_id: str,
    *,
    neutral_site: bool,
    talent_z: dict[str, float],
    returning_z: dict[str, float],
    home_advanced: dict[str, float],
    away_advanced: dict[str, float],
    rest_diff: float,
) -> list[float]:
    """Always defined and always ``len(FEATURE_NAMES)`` long -- a team
    absent from the z-score maps, or with no admissible advanced-stat row,
    contributes 0 to its side of the corresponding diff, the same "missing
    means no signal, not a guess" convention ``models/shrinkage.py`` and
    ``features/advanced_stats.py`` use.
    """
    talent_diff = talent_z.get(home_team_id, 0.0) - talent_z.get(away_team_id, 0.0)
    returning_diff = returning_z.get(home_team_id, 0.0) - returning_z.get(away_team_id, 0.0)
    hfa_indicator = 0.0 if neutral_site else 1.0
    advanced_diffs = [
        home_advanced[key] - away_advanced[key] for key in ADVANCED_STAT_KEYS
    ]
    return [talent_diff, returning_diff, hfa_indicator, rest_diff, *advanced_diffs]


def fit_logistic_as_of(
    conn: sqlite3.Connection,
    as_of_utc: str,
    *,
    l2_lambda: float = DEFAULT_L2_LAMBDA,
    min_n: int = 30,
    reader_game_id: str = "logistic-fit-lookup",
    rest_lookup: RestLookup | None = None,
) -> LogisticFit:
    """Fit P_logit on every completed CFBD game, any season, strictly
    before ``as_of_utc``.

    ``rest_lookup`` defaults to building one internally (a single query over
    every completed game); a caller refitting every week of a backtest
    (``backtest/harness.py``) should build ONE and pass it through instead,
    since the schedule it indexes never changes within a run.

    The advanced-stats cache is still built once PER SEASON here regardless
    -- ``team_season_advanced`` rows are looked up per-game with an as-of
    cutoff (weekly availability class, unlike the season-static z-scores),
    so caching per season keeps this at one query per season rather than one
    per game even as the training set grows to 10,000+ pooled games.
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

    rest = rest_lookup if rest_lookup is not None else RestLookup(conn)

    design_rows: list[list[float]] = []
    outcomes: list[bool] = []
    for season, season_games in by_season.items():
        talent_z = talent_zscores(conn, season)
        returning_z = returning_ppa_zscores(conn, season)
        advanced_cache = load_advanced_stats(conn, season)
        for game in season_games:
            home_points = float(game["home_points"])
            away_points = float(game["away_points"])
            if home_points == away_points:
                continue  # no ties in the modern rulebook; a malformed row, not a result
            kickoff_utc = str(game["kickoff_utc"])
            home_id, away_id = game["home_team_id"], game["away_team_id"]
            design_rows.append(game_features(
                home_id, away_id,
                neutral_site=bool(game["neutral_site"]),
                talent_z=talent_z, returning_z=returning_z,
                home_advanced=team_advanced_nets(
                    advanced_cache, home_id, kickoff_utc, season=season
                ),
                away_advanced=team_advanced_nets(
                    advanced_cache, away_id, kickoff_utc, season=season
                ),
                rest_diff=(
                    rest.rest_days(home_id, kickoff_utc) - rest.rest_days(away_id, kickoff_utc)
                ),
            ))
            outcomes.append(home_points > away_points)

    return fit_logistic(
        design_rows, outcomes, feature_names=FEATURE_NAMES, l2_lambda=l2_lambda, min_n=min_n
    )
