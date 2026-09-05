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
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from cfb_analytics.models.ridge import DEFAULT_MIN_GAMES, DEFAULT_RIDGE_LAMBDA, fit_ratings


@dataclass(frozen=True)
class GamePrediction:
    game_id: str
    season: int
    week: int
    home_team_id: str
    away_team_id: str
    predicted_margin: float
    actual_margin: float

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
    }


def run_walk_forward(
    conn: sqlite3.Connection,
    seasons: list[int],
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
) -> WalkForwardRun:
    run = WalkForwardRun()
    for season in seasons:
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
                          neutral_site
                   FROM games
                   WHERE source = 'cfbd' AND season = ? AND completed = 1
                     AND kickoff_utc < ?""",
                (season, as_of_utc),
            ).fetchall()
            ratings = fit_ratings(
                [_row_to_history_game(row) for row in history_rows],
                ridge_lambda=ridge_lambda,
                min_games=min_games,
            )
            if ratings.status != "active":
                run.skipped_insufficient_history += len(week_games)
                continue

            for row in week_games:
                margin = ratings.margin(
                    row["home_team_id"], row["away_team_id"],
                    neutral_site=bool(row["neutral_site"]),
                )
                if margin is None:
                    run.skipped_unrated_team += 1
                    continue
                run.predictions.append(GamePrediction(
                    game_id=row["game_id"], season=season, week=week,
                    home_team_id=row["home_team_id"], away_team_id=row["away_team_id"],
                    predicted_margin=margin,
                    actual_margin=float(row["home_points"]) - float(row["away_points"]),
                ))
    return run
