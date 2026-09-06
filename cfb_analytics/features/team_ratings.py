"""Leakage-safe internal team-strength ratings (ridge regression).

``fit_ratings_as_of`` is the integration layer between the pure-math model in
``models/ridge.py`` and the store: it pulls this season's CFBD games with a
kickoff strictly before a given cutoff, routed through ``AsOfReader`` exactly
like ``features/qb.py``'s ``presumptive_starter_as_of``, hands them to
``fit_ratings`` with recency weighting enabled (plan section 6.1), and then
blends the result toward an early-season prior (``models/shrinkage.py``).

Deliberately season-scoped, not multi-season: CFB rosters turn over enough
each year that a prior season's fit is not treated as a starting point for
THIS season's in-season fit. The shrinkage prior is the one place a prior
season's rating still matters -- as one input to an explicitly-blended
early-season prior, not as extra training data.

Preseason talent/returning-production numbers are exactly what
``availability_class = 'preseason'`` (see the ``team_talent`` and
``returning_production`` migrations) is for: both are knowable before week 1
of THIS season, so using them as inputs to this season's prior is not a
leak, unlike using a season-final aggregate would be. The only check that
actually matters here is that a row's own ``season`` column matches the
season being fit -- the same thing ``AsOfReader.check_preseason_season``
checks for other preseason features.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime

from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.models.ridge import (
    DEFAULT_MIN_GAMES,
    DEFAULT_RIDGE_LAMBDA,
    RidgeRatings,
    TeamRating,
    fit_ratings,
)
from cfb_analytics.models.shrinkage import (
    DEFAULT_COEFFICIENTS,
    ShrinkageCoefficients,
    blend_toward_prior,
    prior_rating,
    zscore,
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
    apply_shrinkage: bool = True,
    previous_season_ratings: RidgeRatings | None = None,
    coeffs: ShrinkageCoefficients = DEFAULT_COEFFICIENTS,
) -> RidgeRatings:
    """Fit ridge ratings from this season's CFBD games completed before ``as_of_utc``.

    Only ``completed`` CFBD-sourced games are eligible input -- Outlier-sourced
    rows in ``games`` carry no final score and would just be dropped by
    ``models.ridge``'s own score-completeness check anyway, but filtering by
    source here keeps the query itself honest about what it reads.

    Recency weighting (``as_of``) is always passed to ``fit_ratings`` here --
    unlike that function's own default, this integration layer has no reason
    not to use it. The shrinkage prior is applied by default too
    (``apply_shrinkage=True``); pass ``previous_season_ratings`` when calling
    this repeatedly across many weeks of the same season (the walk-forward
    backtest does) to avoid refitting the entire previous season on every call.
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
    ratings = fit_ratings(
        games, ridge_lambda=ridge_lambda, min_games=min_games,
        as_of=datetime.fromisoformat(as_of_utc),
    )
    if not apply_shrinkage:
        return ratings
    return apply_shrinkage_prior(
        conn, ratings, season,
        previous_season_ratings=previous_season_ratings, coeffs=coeffs,
    )


def previous_season_final_ratings(
    conn: sqlite3.Connection,
    season: int,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
) -> RidgeRatings | None:
    """The prior season's own final ridge fit (no recency weighting -- every
    game in a completed season is equally "final"), or None if that season
    has no completed CFBD games in the store at all. That is the honest
    state for whatever season this store's history starts at (2014 here),
    not an error: ``prior_rating`` treats a missing previous-season input the
    same way it treats missing talent/returning-production data.
    """
    previous = season - 1
    exists = conn.execute(
        "SELECT 1 FROM games WHERE source = ? AND season = ? AND completed = 1 LIMIT 1",
        (SOURCE, previous),
    ).fetchone()
    if exists is None:
        return None
    rows = conn.execute(
        """SELECT home_team_id, away_team_id, home_points, away_points, neutral_site
           FROM games WHERE source = ? AND season = ? AND completed = 1""",
        (SOURCE, previous),
    ).fetchall()
    ratings = fit_ratings([dict(row) for row in rows], ridge_lambda=ridge_lambda,
                           min_games=min_games)
    return ratings if ratings.status == "active" else None


def _talent_zscores(conn: sqlite3.Connection, season: int) -> dict[str, float]:
    rows = conn.execute(
        """SELECT team_id, talent_composite FROM team_talent
           WHERE season = ? AND talent_composite IS NOT NULL""",
        (season,),
    ).fetchall()
    return zscore({row["team_id"]: float(row["talent_composite"]) for row in rows})


def _returning_ppa_zscores(conn: sqlite3.Connection, season: int) -> dict[str, float]:
    rows = conn.execute(
        """SELECT team_id, percent_ppa FROM returning_production
           WHERE season = ? AND percent_ppa IS NOT NULL""",
        (season,),
    ).fetchall()
    return zscore({row["team_id"]: float(row["percent_ppa"]) for row in rows})


def apply_shrinkage_prior(
    conn: sqlite3.Connection,
    ratings: RidgeRatings,
    season: int,
    *,
    previous_season_ratings: RidgeRatings | None = None,
    coeffs: ShrinkageCoefficients = DEFAULT_COEFFICIENTS,
) -> RidgeRatings:
    """Blend an already-fit ``RidgeRatings`` toward the early-season prior
    (plan section 6.1), team by team, offense and defense identically.

    A no-op unless ``ratings.status == 'active'`` -- there is nothing to
    blend on an insufficient-history or fit-failed result. Pass
    ``previous_season_ratings`` when the caller already has it (the
    walk-forward backtest fits it once per season and reuses it across every
    week); otherwise it is computed here, once, for this call.
    """
    if ratings.status != "active":
        return ratings
    previous = (
        previous_season_ratings
        if previous_season_ratings is not None
        else previous_season_final_ratings(conn, season)
    )
    talent_z = _talent_zscores(conn, season)
    returning_z = _returning_ppa_zscores(conn, season)

    blended: dict[str, TeamRating] = {}
    for team_id, rating in ratings.teams.items():
        prev = previous.teams.get(team_id) if previous is not None else None
        z_talent = talent_z.get(team_id)
        z_returning = returning_z.get(team_id)
        offense_prior = prior_rating(
            prev.offense if prev is not None else None, z_talent, z_returning, coeffs=coeffs
        )
        defense_prior = prior_rating(
            prev.defense if prev is not None else None, z_talent, z_returning, coeffs=coeffs
        )
        blended[team_id] = TeamRating(
            offense=blend_toward_prior(
                rating.offense, offense_prior, rating.games, k=coeffs.k
            ),
            defense=blend_toward_prior(
                rating.defense, defense_prior, rating.games, k=coeffs.k
            ),
            games=rating.games,
        )
    return replace(ratings, teams=blended)
