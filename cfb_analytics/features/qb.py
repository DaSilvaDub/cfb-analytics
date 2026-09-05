"""Presumptive starting quarterback, by recent pass-attempt share.

This is NOT a confirmed starter. No free source confirms who plays
quarterback in an upcoming game (see ``sources/__init__.py``): ESPN publishes
no CFB depth chart, and CFBD's own roster carries no ``starter`` field. What
CFBD does give, through ``/games/players``, is who has actually been throwing
the ball in this team's own recent games -- a genuine signal for QB experience
and efficiency differentials, even though it says nothing definitive about
next week's lineup.

The hard CORE-tier blocker on ``qb_status == 'unknown'`` is untouched by this
module. A team whose presumptive starter was just injured, or a team starting
a true freshman for the first time, looks identical here to a team with no QB
uncertainty at all -- this module has no way to know that, and does not
pretend to.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from cfb_analytics.features.asof import AsOfReader


@dataclass
class _Accumulator:
    name: str | None
    attempts: int = 0
    games: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PresumptiveStarter:
    """QB1-by-recent-usage for one team, as of one kickoff.

    ``attempt_share`` is this player's share of the TEAM's own pass attempts
    across the lookback window, not a league-wide percentile. Near 1.0 means
    one player has taken nearly every drop-back; a share well under 1.0 across
    several games is itself a signal of an unsettled QB room, and is exactly
    the kind of case the CORE-tier QB-status blocker exists for.
    """

    player_id: str
    name: str | None
    attempts: int
    games: int
    attempt_share: float


def presumptive_starter_as_of(
    conn: sqlite3.Connection,
    team_id: str,
    season: int,
    kickoff_utc: str,
    *,
    lookback_games: int = 4,
    reader_game_id: str = "presumptive-starter-lookup",
) -> PresumptiveStarter | None:
    """QB1-by-usage over the team's last N games strictly before ``kickoff_utc``.

    Returns None when no prior game exists yet this season -- the correct,
    honest answer for a week-1 matchup, and exactly the case the plan's hard
    CORE blocker on unknown QB status already exists to cover.

    Routed through ``AsOfReader.admissible`` rather than a hand-rolled date
    comparison, so this obeys the same leakage rule as every other feature
    and a bug in the rule only needs fixing in one place.
    """
    reader = AsOfReader(game_id=reader_game_id, kickoff_utc=kickoff_utc, season=season)

    rows = conn.execute(
        """SELECT p.player_id, pl.name, p.attempts, g.kickoff_utc, g.game_id
           FROM player_game_passing p
           JOIN games g ON g.game_id = p.game_id
           LEFT JOIN players pl ON pl.player_id = p.player_id
           WHERE p.team_id = ? AND p.season = ? AND p.attempts IS NOT NULL
           ORDER BY g.kickoff_utc DESC""",
        (team_id, season),
    ).fetchall()

    admissible = reader.admissible(
        [dict(row) for row in rows], what="player_game_passing", as_of_field="kickoff_utc"
    )
    if not admissible:
        return None

    recent_game_ids = list(dict.fromkeys(row["game_id"] for row in admissible))[:lookback_games]
    window = [row for row in admissible if row["game_id"] in recent_game_ids]

    accumulators: dict[str, _Accumulator] = {}
    team_attempts = 0
    for row in window:
        attempts = int(row["attempts"])
        team_attempts += attempts
        entry = accumulators.setdefault(row["player_id"], _Accumulator(name=row["name"]))
        entry.attempts += attempts
        entry.games.add(row["game_id"])

    if team_attempts == 0 or not accumulators:
        return None

    leader_id, leader = max(accumulators.items(), key=lambda item: item[1].attempts)
    return PresumptiveStarter(
        player_id=leader_id,
        name=leader.name,
        attempts=leader.attempts,
        games=len(leader.games),
        attempt_share=leader.attempts / team_attempts,
    )
