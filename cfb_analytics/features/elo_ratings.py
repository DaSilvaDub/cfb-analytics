"""Leakage-safe lookup of CFBD's own weekly Elo ratings (``backfill-elo``).

One important subtlety about what a stored row actually means: CFBD's
``/ratings/elo?week=W`` returns the rating AFTER week W's games have been
played, not a preseason-of-week-W number. ``ingest/cfbd_fundamentals.py``
stamps that row's ``as_of_utc`` as the day after week W's last kickoff,
which is what makes it correct to route through ``AsOfReader.latest`` here
exactly like every other feature: predicting week W's games must use the
most recent row strictly before week W's EARLIEST kickoff, which is week
W-1's post-game rating -- never week W's own row, which would leak that
same week's results into predicting it.

Using ``AsOfReader.latest`` rather than hand-computing "week - 1" also
survives bye weeks for free: CFBD emits a row for every FBS team every
week (carrying a bye-week team's rating forward unchanged), so the most
recent admissible row is always the right one regardless of whether a
specific week number exists for that team.
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.features.asof import AsOfReader


def elo_rating_as_of(
    conn: sqlite3.Connection,
    team_id: str,
    season: int,
    as_of_utc: str,
    *,
    reader_game_id: str = "elo-rating-lookup",
) -> float | None:
    """CFBD's weekly Elo rating for a team, as of strictly before ``as_of_utc``.

    Returns None -- never a guess -- when this team has no admissible Elo
    row this season: an FCS opponent (CFBD's Elo backfill only resolves FBS
    teams; see ``cfbd_fundamentals.backfill_elo``), or week 1 before any
    week has been played yet.
    """
    reader = AsOfReader(game_id=reader_game_id, kickoff_utc=as_of_utc, season=season)
    rows = conn.execute(
        """SELECT rating, as_of_utc FROM team_ratings
           WHERE team_id = ? AND season = ? AND source = 'elo_cfbd'
             AND snapshot_scope = 'weekly' AND rating IS NOT NULL""",
        (team_id, season),
    ).fetchall()
    latest = reader.latest(
        [dict(row) for row in rows], what="team_ratings.elo_cfbd", as_of_field="as_of_utc"
    )
    return float(latest["rating"]) if latest else None
