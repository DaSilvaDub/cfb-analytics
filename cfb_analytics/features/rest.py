"""Rest-days diff (plan section 5's T5 ``rest_diff``).

Unlike ``team_season_advanced`` or Elo, rest is pure SCHEDULE information --
which teams played when is public and fixed well before any game is played,
not something that accrues in-season the way performance stats do. So
unlike those, using a later cutoff here never smuggles in a RESULT; a game's
own kickoff (rather than the walk-forward harness's shared per-week cutoff)
would be equally safe, though this module accepts whatever ``as_of_utc`` a
caller passes and never claims a schedule fact known only after it.

Built once per backtest run (``RestLookup(conn)``) rather than queried per
game: the schedule does not change during a run, so one global sort beats
one SQL query per team per game once P_logit's cross-season training set
runs into the thousands of games (see ``features/ensemble.py``).
"""

from __future__ import annotations

import bisect
import sqlite3
from datetime import datetime

# Typical FBS bye-to-bye cadence. Used ONLY when a team has no earlier CFBD
# game at all (its first game in the dataset) -- a neutral assumption for
# the diff calculation, not a guessed rest value for a specific matchup.
DEFAULT_REST_DAYS = 7.0


def _team_game_dates(conn: sqlite3.Connection) -> dict[str, list[datetime]]:
    rows = conn.execute(
        """SELECT home_team_id AS team_id, kickoff_utc FROM games
           WHERE source = 'cfbd' AND completed = 1
           UNION ALL
           SELECT away_team_id AS team_id, kickoff_utc FROM games
           WHERE source = 'cfbd' AND completed = 1"""
    ).fetchall()
    by_team: dict[str, list[datetime]] = {}
    for row in rows:
        by_team.setdefault(row["team_id"], []).append(
            datetime.fromisoformat(row["kickoff_utc"])
        )
    for dates in by_team.values():
        dates.sort()
    return by_team


class RestLookup:
    """Days since a team's most recent completed game, as of any kickoff."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._by_team = _team_game_dates(conn)

    def rest_days(self, team_id: str, as_of_utc: str) -> float:
        dates = self._by_team.get(team_id, [])
        as_of = datetime.fromisoformat(as_of_utc)
        idx = bisect.bisect_left(dates, as_of)
        if idx == 0:
            return DEFAULT_REST_DAYS
        return (as_of - dates[idx - 1]).total_seconds() / 86400.0
