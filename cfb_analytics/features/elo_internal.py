"""Leakage-safe internal Elo ratings (plan section 6.2).

``fit_internal_elo_as_of`` is the integration layer between the pure-math
model in ``models/elo.py`` and the store, mirroring ``features/team_ratings.py``'s
role for ridge: it pulls this season's CFBD games with a kickoff strictly
before a given cutoff (via ``AsOfReader``), pools non-FBS opponents into a
single synthetic team, seeds every FBS team's rating via the plan's
preseason blend, and hands the result to ``fit_elo``.

Elo's own sequential nature is structurally different from ridge's batch fit
in one important way: ridge's shrinkage prior treats "the previous season's
rating" as an independently-fit reference point (fit fresh from that
season's own games, no recursion). Elo's ``R_0`` formula instead means a
season's preseason rating depends on the season before it, which depends on
the one before THAT, all the way back to whatever season this store's
history starts at (2014). ``_elo_chain_up_to`` walks that whole chain every
time it is asked for a previous season's final ratings -- cheap enough to
redo from scratch on every call, unlike ridge's O(n^3) matrix solve: a
season's Elo fit is a single O(games) pass over ~900 games, a few
milliseconds, so even a full 12-season chain costs well under a second.

The talent/returning-production coefficients (``g``, ``d`` in the plan's
formula) were grid-searched 2026-09-06 against 2021 weeks 1-5 out-of-sample
log loss, checked against held-out 2019 -- see ``models/elo.py``'s
``DEFAULT_PRESEASON_TALENT_COEFF``/``DEFAULT_PRESEASON_RETURNING_COEFF`` for
the exact values and search notes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.features.preseason import returning_ppa_zscores, talent_zscores
from cfb_analytics.models.elo import (
    DEFAULT_K,
    DEFAULT_MIN_GAMES,
    DEFAULT_PRESEASON_RETURNING_COEFF,
    DEFAULT_PRESEASON_TALENT_COEFF,
    HFA_ELO_POINTS,
    POOL_INITIAL_RATING,
    POOL_TEAM_ID,
    EloRatings,
    fit_elo,
    preseason_rating,
)

SOURCE = "cfbd"


def _fbs_team_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT team_id FROM teams WHERE classification = 'fbs'").fetchall()
    return {row["team_id"] for row in rows}


def _pool_non_fbs_teams(
    games: Iterable[Mapping[str, Any]], fbs_team_ids: set[str]
) -> list[dict[str, Any]]:
    """Remap any team not in ``fbs_team_ids`` to the shared FCS pool id.

    A game between two non-FBS teams (should not exist in a CFBD FBS-games
    feed, but never assumed) would collapse to the pool playing itself --
    dropped rather than fed to the model, since ``fit_elo`` cannot process a
    "team plays itself" row meaningfully.
    """
    pooled = []
    for game in games:
        home = game["home_team_id"] if game["home_team_id"] in fbs_team_ids else POOL_TEAM_ID
        away = game["away_team_id"] if game["away_team_id"] in fbs_team_ids else POOL_TEAM_ID
        if home == away:
            continue
        pooled.append({**game, "home_team_id": home, "away_team_id": away})
    return pooled


def _season_games(conn: sqlite3.Connection, season: int) -> list[dict]:
    rows = conn.execute(
        """SELECT game_id, home_team_id, away_team_id, home_points, away_points,
                  neutral_site, kickoff_utc
           FROM games
           WHERE source = ? AND season = ? AND completed = 1""",
        (SOURCE, season),
    ).fetchall()
    return [dict(row) for row in rows]


def _preseason_seeds(
    conn: sqlite3.Connection,
    season: int,
    previous: EloRatings | None,
    fbs_team_ids: set[str],
    *,
    talent_coeff: float = DEFAULT_PRESEASON_TALENT_COEFF,
    returning_coeff: float = DEFAULT_PRESEASON_RETURNING_COEFF,
) -> dict[str, float]:
    talent_z = talent_zscores(conn, season)
    returning_z = returning_ppa_zscores(conn, season)
    seeds = {
        team_id: preseason_rating(
            previous.teams[team_id].rating if previous and team_id in previous.teams else None,
            talent_z.get(team_id),
            returning_z.get(team_id),
            talent_coeff=talent_coeff,
            returning_coeff=returning_coeff,
        )
        for team_id in fbs_team_ids
    }
    seeds[POOL_TEAM_ID] = POOL_INITIAL_RATING
    return seeds


def _earliest_cfbd_season(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT MIN(season) AS s FROM games WHERE source = ? AND completed = 1", (SOURCE,)
    ).fetchone()
    return int(row["s"]) if row["s"] is not None else None


def _elo_chain_up_to(
    conn: sqlite3.Connection,
    season: int,
    *,
    k: float,
    hfa: float,
    min_games: int,
    talent_coeff: float = DEFAULT_PRESEASON_TALENT_COEFF,
    returning_coeff: float = DEFAULT_PRESEASON_RETURNING_COEFF,
) -> dict[int, EloRatings]:
    earliest = _earliest_cfbd_season(conn)
    if earliest is None:
        return {}
    fbs_team_ids = _fbs_team_ids(conn)
    chain: dict[int, EloRatings] = {}
    previous: EloRatings | None = None
    for year in range(earliest, season):
        games = _season_games(conn, year)
        if not games:
            previous = None
            continue
        seeds = _preseason_seeds(
            conn, year, previous, fbs_team_ids,
            talent_coeff=talent_coeff, returning_coeff=returning_coeff,
        )
        fitted = fit_elo(
            _pool_non_fbs_teams(games, fbs_team_ids),
            initial_ratings=seeds, k=k, hfa=hfa, min_games=min_games,
        )
        chain[year] = fitted
        previous = fitted if fitted.status == "active" else None
    return chain


def previous_season_final_elo(
    conn: sqlite3.Connection,
    season: int,
    *,
    k: float = DEFAULT_K,
    hfa: float = HFA_ELO_POINTS,
    min_games: int = DEFAULT_MIN_GAMES,
    talent_coeff: float = DEFAULT_PRESEASON_TALENT_COEFF,
    returning_coeff: float = DEFAULT_PRESEASON_RETURNING_COEFF,
) -> EloRatings | None:
    """The prior season's own final internal Elo fit, walking the full
    preseason-blend chain back to this store's earliest CFBD season.

    None when the store has no completed CFBD games for ANY season before
    ``season`` -- the honest state for this store's own earliest season
    (2014), not an error.
    """
    chain = _elo_chain_up_to(
        conn, season, k=k, hfa=hfa, min_games=min_games,
        talent_coeff=talent_coeff, returning_coeff=returning_coeff,
    )
    return chain.get(season - 1)


def fit_internal_elo_as_of(
    conn: sqlite3.Connection,
    season: int,
    as_of_utc: str,
    *,
    k: float = DEFAULT_K,
    hfa: float = HFA_ELO_POINTS,
    min_games: int = DEFAULT_MIN_GAMES,
    reader_game_id: str = "elo-fit-lookup",
    previous_season_ratings: EloRatings | None = None,
    talent_coeff: float = DEFAULT_PRESEASON_TALENT_COEFF,
    returning_coeff: float = DEFAULT_PRESEASON_RETURNING_COEFF,
) -> EloRatings:
    """Fit internal Elo ratings from this season's CFBD games completed
    before ``as_of_utc``.

    Pass ``previous_season_ratings`` when calling this repeatedly across
    many weeks of the same season (the walk-forward backtest does) to avoid
    re-walking the whole historical chain on every call -- cheap in
    isolation, but wasteful to repeat dozens of times for an unchanging
    answer.
    """
    reader = AsOfReader(game_id=reader_game_id, kickoff_utc=as_of_utc, season=season)
    games = reader.admissible(
        _season_games(conn, season), what="games", as_of_field="kickoff_utc"
    )
    fbs_team_ids = _fbs_team_ids(conn)
    previous = (
        previous_season_ratings
        if previous_season_ratings is not None
        else previous_season_final_elo(
            conn, season, k=k, hfa=hfa, min_games=min_games,
            talent_coeff=talent_coeff, returning_coeff=returning_coeff,
        )
    )
    seeds = _preseason_seeds(
        conn, season, previous, fbs_team_ids,
        talent_coeff=talent_coeff, returning_coeff=returning_coeff,
    )
    return fit_elo(
        _pool_non_fbs_teams(games, fbs_team_ids),
        initial_ratings=seeds, k=k, hfa=hfa, min_games=min_games,
    )
