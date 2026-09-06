"""Preseason talent/returning-production z-scores, shared by every model
that blends them into an early-season prior (ridge's shrinkage prior,
the internal Elo model's ``R_0`` formula).

Both source tables carry ``availability_class = 'preseason'`` by CHECK
constraint (see the ``team_talent`` and ``returning_production``
migrations): both are knowable before week 1 of the season they are
labelled with, so using them as an input to that season's early-season
prior is not a leak, unlike using a season-final aggregate would be. The
only thing that actually needs verifying is that a row's own ``season``
column matches the season being fit -- the same check
``AsOfReader.check_preseason_season`` does for other preseason features.
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.models.shrinkage import zscore


def talent_zscores(conn: sqlite3.Connection, season: int) -> dict[str, float]:
    rows = conn.execute(
        """SELECT team_id, talent_composite FROM team_talent
           WHERE season = ? AND talent_composite IS NOT NULL""",
        (season,),
    ).fetchall()
    return zscore({row["team_id"]: float(row["talent_composite"]) for row in rows})


def returning_ppa_zscores(conn: sqlite3.Connection, season: int) -> dict[str, float]:
    rows = conn.execute(
        """SELECT team_id, percent_ppa FROM returning_production
           WHERE season = ? AND percent_ppa IS NOT NULL""",
        (season,),
    ).fetchall()
    return zscore({row["team_id"]: float(row["percent_ppa"]) for row in rows})
