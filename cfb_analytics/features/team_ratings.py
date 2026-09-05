"""Leakage-safe internal team-strength ratings (ridge regression).

``fit_ratings_as_of`` is the integration layer between the pure-math model in
``models/ridge.py`` and the store: it pulls this season's CFBD games with a
kickoff strictly before a given cutoff, routed through ``AsOfReader`` exactly
like ``features/qb.py``'s ``presumptive_starter_as_of``, and hands them to
``fit_ratings``.

Deliberately season-scoped, not multi-season: CFB rosters turn over enough
each year that a prior season's fit is not treated as a starting point here.
This is also why week 1 of every season legitimately comes back
``insufficient_history`` -- there is nothing yet to fit. Closing that gap is
exactly what ``ridge.py``'s documented-but-not-yet-built early-season
shrinkage prior (blending in returning-production/recruiting talent) is for.
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.models.ridge import (
    DEFAULT_MIN_GAMES,
    DEFAULT_RIDGE_LAMBDA,
    RidgeRatings,
    fit_ratings,
)

SOURCE = "cfbd"


def fit_ratings_as_of(
    conn: sqlite3.Connection,
    season: int,
    as_of_utc: str,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
    reader_game_id: str = "ridge-fit-lookup",
) -> RidgeRatings:
    """Fit ridge ratings from this season's CFBD games completed before ``as_of_utc``.

    Only ``completed`` CFBD-sourced games are eligible input -- Outlier-sourced
    rows in ``games`` carry no final score and would just be dropped by
    ``models.ridge``'s own score-completeness check anyway, but filtering by
    source here keeps the query itself honest about what it reads.
    """
    reader = AsOfReader(game_id=reader_game_id, kickoff_utc=as_of_utc, season=season)
    rows = conn.execute(
        """SELECT game_id, home_team_id, away_team_id, home_points, away_points,
                  neutral_site, kickoff_utc
           FROM games
           WHERE source = ? AND season = ? AND completed = 1""",
        (SOURCE, season),
    ).fetchall()
    games = reader.admissible(
        [dict(row) for row in rows], what="games", as_of_field="kickoff_utc"
    )
    return fit_ratings(games, ridge_lambda=ridge_lambda, min_games=min_games)
