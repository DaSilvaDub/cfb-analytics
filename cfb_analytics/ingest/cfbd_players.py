"""Ingest CFBD roster and per-game passing stats.

Two cheap bulk endpoints (one call each, verified 2026-09-04): a full-league
roster per season (30k+ rows, 315 teams) and a full week of per-game player
box scores (137 games, 260 teams). Neither needs a per-team loop.

Team resolution differs by endpoint, and each is chosen to avoid guessing:

* **Roster** gives a school NAME with no id. It is resolved against the
  already-ingested ``teams`` table by exact name match (CFBD's own roster
  feed and CFBD's own team feed use the identical school-name vocabulary --
  verified 2026-09-04: all 246 stored CFBD teams matched exactly). A name
  with no match is a non-FBS school never ingested via ``backfill_years``
  (69 of 315 in the same check) and is skipped, not guessed at.
* **Per-game passing** gives no team id or name at the team level at all --
  only ``homeAway``. It is resolved via the game's own ``home_team_id`` /
  ``away_team_id`` in the already-ingested ``games`` table, which is exact by
  construction and needs no name matching whatsoever.

Both therefore require the relevant CFBD games/teams to already be in the
store; a roster or box score for a team/game not yet ingested is counted and
skipped rather than fabricating an identity for it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import (
    CFBDClient,
    parse_game_player_passing,
    parse_roster_row,
)

SOURCE = "cfbd"


def _name_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


@dataclass
class RosterIngestSummary:
    season: int
    rows_seen: int = 0
    written: int = 0
    unmatched_teams: set[str] = field(default_factory=set)

    def as_text(self) -> str:
        lines = [
            f"cfbd roster {self.season}",
            f"  roster rows seen : {self.rows_seen}",
            f"  written          : {self.written}",
        ]
        if self.unmatched_teams:
            lines.append(
                f"  unmatched teams  : {len(self.unmatched_teams)} "
                "(not in the FBS store; not guessed at)"
            )
        return "\n".join(lines)


def _team_id_by_school(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        "SELECT team_id, school FROM teams WHERE team_id LIKE 'cfbd:%' AND school IS NOT NULL"
    ).fetchall()
    return {_name_key(row["school"]): str(row["team_id"]) for row in rows}


def ingest_roster(
    conn: sqlite3.Connection, client: CFBDClient, season: int
) -> RosterIngestSummary:
    summary = RosterIngestSummary(season=season)
    resolver = _team_id_by_school(conn)

    raw_rows = client.fetch_roster(season)
    summary.rows_seen = len(raw_rows)

    for raw in raw_rows:
        parsed = parse_roster_row(raw)
        if parsed is None:
            continue
        team_id = resolver.get(_name_key(parsed["team_name"]))
        if team_id is None:
            summary.unmatched_teams.add(parsed["team_name"])
            continue

        store.upsert_player(conn, parsed["player_id"], parsed["name"])
        store.upsert_player_season(conn, {
            "player_id": parsed["player_id"],
            "season": season,
            "team_id": team_id,
            "position": parsed["position"],
            "class_year": parsed["class_year"],
            "height_in": parsed["height_in"],
            "weight_lb": parsed["weight_lb"],
            "home_state": parsed["home_state"],
            "source": SOURCE,
        })
        summary.written += 1

    return summary


@dataclass
class GamePassingIngestSummary:
    season: int
    week: int
    games_seen: int = 0
    games_unmatched: int = 0
    rows_written: int = 0

    def as_text(self) -> str:
        lines = [
            f"cfbd game passing {self.season} week {self.week}",
            f"  games in feed    : {self.games_seen}",
            f"  rows written     : {self.rows_written}",
        ]
        if self.games_unmatched:
            lines.append(
                f"  games not in store: {self.games_unmatched} "
                "(backfill games for this season/week first)"
            )
        return "\n".join(lines)


def ingest_game_passing(
    conn: sqlite3.Connection, client: CFBDClient, season: int, week: int,
    *, season_type: str = "regular",
) -> GamePassingIngestSummary:
    summary = GamePassingIngestSummary(season=season, week=week)

    for raw_game in client.fetch_game_players(season, week, season_type=season_type):
        summary.games_seen += 1
        rows = parse_game_player_passing(raw_game)
        if not rows:
            continue
        game_id = rows[0]["game_id"]

        game_row = conn.execute(
            "SELECT home_team_id, away_team_id FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if game_row is None:
            summary.games_unmatched += 1
            continue

        for row in rows:
            team_id = (
                game_row["home_team_id"] if row["home_away"] == "home"
                else game_row["away_team_id"]
            )
            store.upsert_player(conn, row["player_id"], row["name"])
            store.upsert_player_game_passing(conn, {
                "game_id": game_id,
                "team_id": team_id,
                "player_id": row["player_id"],
                "season": season,
                "week": week,
                "completions": row["completions"],
                "attempts": row["attempts"],
                "yards": row["yards"],
                "avg_yards": row["avg_yards"],
                "touchdowns": row["touchdowns"],
                "interceptions": row["interceptions"],
                "qbr": row["qbr"],
                "source": SOURCE,
            })
            summary.rows_written += 1

    return summary


def _require_season_stored(conn: sqlite3.Connection, season: int) -> None:
    if not conn.execute(
        "SELECT 1 FROM games WHERE source = 'cfbd' AND season = ? LIMIT 1", (season,)
    ).fetchone():
        raise SchemaError(f"No CFBD games stored for {season}; backfill games first")


def completed_weeks(
    conn: sqlite3.Connection, season: int, *, season_type: str = "regular"
) -> list[int]:
    """Weeks this season with at least one finished game, per the store.

    Drives a full backfill: iterate weeks that actually happened rather than
    guessing a season length (postseason especially varies) or blindly
    sweeping 1..N and eating empty responses for weeks not yet played.
    """
    _require_season_stored(conn, season)
    rows = conn.execute(
        """SELECT DISTINCT week FROM games
           WHERE source = 'cfbd' AND season = ? AND season_type = ?
             AND completed = 1 AND week IS NOT NULL
           ORDER BY week""",
        (season, season_type),
    ).fetchall()
    return [int(row["week"]) for row in rows]


def weeks_missing_passing(
    conn: sqlite3.Connection, season: int, *, season_type: str = "regular"
) -> list[int]:
    """Completed weeks this season with no passing rows ingested yet.

    Drives the incremental daily-job leg: only fetch weeks that actually have
    finished games and are not already captured, rather than re-fetching the
    whole season or guessing at "the current week".
    """
    _require_season_stored(conn, season)
    rows = conn.execute(
        """SELECT DISTINCT week FROM games
           WHERE source = 'cfbd' AND season = ? AND season_type = ?
             AND completed = 1 AND week IS NOT NULL
             AND game_id NOT IN (
                 SELECT DISTINCT game_id FROM player_game_passing WHERE season = ?
             )
           ORDER BY week""",
        (season, season_type, season),
    ).fetchall()
    return [int(row["week"]) for row in rows]
