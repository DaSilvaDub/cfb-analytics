"""Walk-forward moneyline predictions for the internal ridge model (plan section 8).

Train strictly on ``as_of_utc < kickoff_utc``; predict week W; never refit
forward. Concretely: ratings are refit ONCE per (season, week) using only
that season's games with a kickoff strictly before the week's earliest
kickoff, then every game in that week is predicted from that single fit.
This is deliberately more conservative than a per-game as-of cutoff -- an
early-week Thursday game never leaks into a same-week Saturday prediction --
and matches the plan's own "predict week W" framing.

Deliberately regular-season only for this first pass: postseason/bowl week
numbering does not follow the same 1..N sequence as the regular season, and
mixing the two needs its own handling. Tracked as a real gap, not silently
dropped -- see ``run_walk_forward``'s docstring.

2020 is included in the walk-forward (it is real, played games), but the
plan calls for it to be *excluded from fitting* and *retained as a stress
slice*: see ``moneyline.py``, which is where that split into "calibration
season" vs "stress season" predictions actually happens. This module has no
opinion about it -- it just labels each prediction with its season.

Each prediction also carries CFBD's own weekly Elo rating for both teams
(``elo_home_rating``/``elo_away_rating``, possibly None), looked up with the
exact same per-week leakage cutoff used for the ridge fit -- see
``features/elo_ratings.py``. This makes the Elo-only baseline (plan section
8, baseline #3; see ``backtest/elo_baseline.py``) score on precisely the
same games as the ridge model, which is what makes the comparison fair: a
game the ridge model could not predict (e.g. insufficient season history)
never produces a ``GamePrediction`` at all, so it never enters either
model's metrics.

Each week's ridge fit gets both of ``models/ridge.py``'s recency weighting
(``as_of`` = that week's own cutoff) and ``features/team_ratings.py``'s
early-season shrinkage-prior blend, exactly like ``fit_ratings_as_of`` --
this module does not call that function directly (it already has its own
history query, built for a slightly different leakage shape: same-week
games grouped together, see the ``as_of_utc`` computation below), but it
applies the identical treatment by calling the same underlying pieces. The
previous season's final ratings are fit ONCE per season and reused across
every week of it, not refit on every single week -- a full-season fit costs
real time (~9s at FBS scale), and the previous season does not change from
week to week within a season.

Each week's fit also runs ``features/elo_internal.py``'s internal Elo model
in parallel (``internal_elo_home_prob``/``internal_elo_away_prob`` on each
prediction, already a probability -- Elo's own output, no margin-to-prob
conversion needed). Unlike ridge, this is cheap enough (Elo is a single
O(games) sequential pass, not an O(n^3) matrix solve) that refitting it
every week costs milliseconds, not seconds -- it does not meaningfully
change this backtest's runtime.

Each week also refits ``features/ensemble.py``'s ``P_logit`` (plan section
6.4's third ensemble member -- ``logit_win_prob`` on each prediction,
already a probability). Unlike ridge/Elo, its training set is not reset per
season (see that module's docstring for why), but IRLS on 3 features is
cheap enough regardless that refitting it fresh every week is still no
meaningful cost.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cfb_analytics.features.elo_internal import (
    fit_internal_elo_as_of,
    previous_season_final_elo,
)
from cfb_analytics.features.elo_ratings import elo_rating_as_of
from cfb_analytics.features.ensemble import fit_logistic_as_of, game_features
from cfb_analytics.features.preseason import returning_ppa_zscores, talent_zscores
from cfb_analytics.features.team_ratings import (
    apply_shrinkage_prior,
    previous_season_final_ratings,
)
from cfb_analytics.models.elo import DEFAULT_K as DEFAULT_ELO_K
from cfb_analytics.models.elo import HFA_ELO_POINTS
from cfb_analytics.models.logistic import DEFAULT_L2_LAMBDA as DEFAULT_LOGIT_L2_LAMBDA
from cfb_analytics.models.ridge import DEFAULT_MIN_GAMES, DEFAULT_RIDGE_LAMBDA, fit_ratings
from cfb_analytics.models.shrinkage import DEFAULT_COEFFICIENTS, ShrinkageCoefficients


@dataclass(frozen=True)
class GamePrediction:
    game_id: str
    season: int
    week: int
    home_team_id: str
    away_team_id: str
    neutral_site: bool
    predicted_margin: float
    actual_margin: float
    elo_home_rating: float | None
    elo_away_rating: float | None
    internal_elo_win_prob: float | None
    logit_win_prob: float | None

    @property
    def home_won(self) -> bool:
        return self.actual_margin > 0


@dataclass
class WalkForwardRun:
    predictions: list[GamePrediction] = field(default_factory=list)
    # Games in a week whose season had fewer than ``min_games`` played so far
    # -- normal and expected in the first few weeks of every season, not a bug.
    skipped_insufficient_history: int = 0
    # A team on one side of the game never appeared in the fit (e.g. an FCS
    # opponent) -- `RidgeRatings.margin` correctly refuses to guess, and this
    # counts how often that happened rather than silently dropping the game.
    skipped_unrated_team: int = 0


def _regular_season_weeks(conn: sqlite3.Connection, season: int) -> list[int]:
    rows = conn.execute(
        """SELECT DISTINCT week FROM games
           WHERE source = 'cfbd' AND season = ? AND season_type = 'regular'
             AND completed = 1 AND week IS NOT NULL
           ORDER BY week""",
        (season,),
    ).fetchall()
    return [int(row["week"]) for row in rows]


def _row_to_history_game(row: Any) -> dict[str, Any]:
    return {
        "home_team_id": row["home_team_id"],
        "away_team_id": row["away_team_id"],
        "home_points": row["home_points"],
        "away_points": row["away_points"],
        "neutral_site": row["neutral_site"],
        "kickoff_utc": row["kickoff_utc"],
    }


def run_walk_forward(
    conn: sqlite3.Connection,
    seasons: list[int],
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
    apply_shrinkage: bool = True,
    coeffs: ShrinkageCoefficients = DEFAULT_COEFFICIENTS,
    elo_k: float = DEFAULT_ELO_K,
    elo_hfa: float = HFA_ELO_POINTS,
    logit_l2_lambda: float = DEFAULT_LOGIT_L2_LAMBDA,
    logit_min_n: int = 30,
) -> WalkForwardRun:
    run = WalkForwardRun()
    for season in seasons:
        previous_season_ratings = (
            previous_season_final_ratings(conn, season, ridge_lambda=ridge_lambda,
                                           min_games=min_games)
            if apply_shrinkage
            else None
        )
        previous_season_elo = previous_season_final_elo(
            conn, season, k=elo_k, hfa=elo_hfa, min_games=min_games
        )
        talent_z = talent_zscores(conn, season)
        returning_z = returning_ppa_zscores(conn, season)
        for week in _regular_season_weeks(conn, season):
            week_games = conn.execute(
                """SELECT game_id, home_team_id, away_team_id, home_points,
                          away_points, neutral_site, kickoff_utc
                   FROM games
                   WHERE source = 'cfbd' AND season = ? AND season_type = 'regular'
                     AND week = ? AND completed = 1
                     AND home_points IS NOT NULL AND away_points IS NOT NULL""",
                (season, week),
            ).fetchall()
            if not week_games:
                continue
            as_of_utc = min(row["kickoff_utc"] for row in week_games)

            history_rows = conn.execute(
                """SELECT home_team_id, away_team_id, home_points, away_points,
                          neutral_site, kickoff_utc
                   FROM games
                   WHERE source = 'cfbd' AND season = ? AND completed = 1
                     AND kickoff_utc < ?""",
                (season, as_of_utc),
            ).fetchall()
            ratings = fit_ratings(
                [_row_to_history_game(row) for row in history_rows],
                ridge_lambda=ridge_lambda,
                min_games=min_games,
                as_of=datetime.fromisoformat(as_of_utc),
            )
            if apply_shrinkage:
                ratings = apply_shrinkage_prior(
                    conn, ratings, season,
                    previous_season_ratings=previous_season_ratings, coeffs=coeffs,
                )
            if ratings.status != "active":
                run.skipped_insufficient_history += len(week_games)
                continue

            elo_ratings_for_week = fit_internal_elo_as_of(
                conn, season, as_of_utc, k=elo_k, hfa=elo_hfa, min_games=min_games,
                previous_season_ratings=previous_season_elo,
            )
            logistic_fit_for_week = fit_logistic_as_of(
                conn, as_of_utc, l2_lambda=logit_l2_lambda, min_n=logit_min_n
            )

            for row in week_games:
                neutral_site = bool(row["neutral_site"])
                margin = ratings.margin(
                    row["home_team_id"], row["away_team_id"], neutral_site=neutral_site
                )
                if margin is None:
                    run.skipped_unrated_team += 1
                    continue
                logit_win_prob = logistic_fit_for_week.probability(
                    game_features(
                        row["home_team_id"], row["away_team_id"], neutral_site=neutral_site,
                        talent_z=talent_z, returning_z=returning_z,
                    )
                )
                run.predictions.append(GamePrediction(
                    game_id=row["game_id"], season=season, week=week,
                    home_team_id=row["home_team_id"], away_team_id=row["away_team_id"],
                    neutral_site=neutral_site,
                    predicted_margin=margin,
                    actual_margin=float(row["home_points"]) - float(row["away_points"]),
                    elo_home_rating=elo_rating_as_of(
                        conn, row["home_team_id"], season, as_of_utc
                    ),
                    elo_away_rating=elo_rating_as_of(
                        conn, row["away_team_id"], season, as_of_utc
                    ),
                    internal_elo_win_prob=elo_ratings_for_week.probability(
                        row["home_team_id"], row["away_team_id"], neutral_site=neutral_site
                    ),
                    logit_win_prob=logit_win_prob,
                ))
    return run
