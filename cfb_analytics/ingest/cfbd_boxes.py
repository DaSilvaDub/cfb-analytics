"""Ingest CFBD team box scores (rushing / net passing) for completed weeks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_fundamentals import _name_key
from cfb_analytics.sources.cfbd import CFBDClient


def parse_team_boxes(raw: dict[str, Any]) -> list[dict[str, Any]]:
    game_raw_id = raw.get("id")
    if game_raw_id is None:
        raise SchemaError("CFBD games/teams row missing id")
    game_id = f"cfbd:{game_raw_id}"
    teams = raw.get("teams")
    if not isinstance(teams, list):
        raise SchemaError(f"CFBD box {game_id} had no teams list")
    parsed: list[dict[str, Any]] = []
    for team in teams:
        if not isinstance(team, dict):
            raise SchemaError(f"CFBD box {game_id} contained a non-object team")
        school = str(team.get("team") or team.get("school") or "").strip()
        if not school:
            raise SchemaError(f"CFBD box {game_id} team missing name")
        cats = {
            str(item.get("category")): item.get("stat")
            for item in team.get("stats") or []
            if isinstance(item, dict)
        }
        parsed.append(
            {
                "game_id": game_id,
                "team_name": school,
                "rushing_yards": _stat_float(cats.get("rushingYards")),
                "net_passing_yards": _stat_float(
                    cats["netPassingYards"]
                    if "netPassingYards" in cats
                    else cats.get("passingYards")
                ),
                "total_yards": _stat_float(cats.get("totalYards")),
                "points": team.get("points"),
            }
        )
    return parsed


def _stat_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _team_name_index(conn: sqlite3.Connection) -> dict[str, str]:
    from cfb_analytics.ingest.cfbd_rankings import _team_name_index as rankings_index

    return rankings_index(conn)


@dataclass(frozen=True)
class BoxSummary:
    games: int
    rows: int
    unresolved: int

    def as_text(self) -> str:
        return (
            f"CFBD team boxes wrote {self.rows} team rows across {self.games} games; "
            f"{self.unresolved} names did not resolve."
        )


def ingest_team_boxes(
    conn: sqlite3.Connection,
    client: CFBDClient,
    year: int,
    week: int,
    *,
    season_type: str = "regular",
    completed_only: bool = False,
) -> BoxSummary:
    payload = client.fetch_game_teams(year, week, season_type=season_type)
    resolver = _team_name_index(conn)
    if completed_only:
        known_games = {
            str(row["game_id"])
            for row in conn.execute(
                "SELECT game_id FROM games WHERE season = ? AND completed = 1",
                (year,),
            )
        }
    else:
        known_games = {
            str(row["game_id"])
            for row in conn.execute("SELECT game_id FROM games WHERE season = ?", (year,))
        }
    written = 0
    unresolved = 0
    games = 0
    with store.RunRecorder(conn, "ingest-team-boxes") as run:
        run.record_health("cfbd", "/games/teams", ok=True, rows=len(payload))
        batch: list[dict[str, Any]] = []
        for raw in payload:
            parsed = parse_team_boxes(raw)
            if parsed:
                games += 1
            for row in parsed:
                if row["game_id"] not in known_games:
                    continue
                team_id = resolver.get(_name_key(row.pop("team_name")))
                if team_id is None:
                    unresolved += 1
                    continue
                batch.append({**row, "team_id": team_id})
        written = store.insert_team_game_boxes(conn, batch)
        run.add_rows(written)
    conn.commit()
    return BoxSummary(games=games, rows=written, unresolved=unresolved)


def ingest_completed_boxes(conn: sqlite3.Connection, client: CFBDClient, season: int) -> BoxSummary:
    weeks = conn.execute(
        """SELECT DISTINCT week FROM games
           WHERE season = ? AND completed = 1 AND week IS NOT NULL
           ORDER BY week""",
        (season,),
    ).fetchall()
    total_games = total_rows = total_unresolved = 0
    for row in weeks:
        summary = ingest_team_boxes(conn, client, season, int(row["week"]), completed_only=True)
        total_games += summary.games
        total_rows += summary.rows
        total_unresolved += summary.unresolved
    return BoxSummary(games=total_games, rows=total_rows, unresolved=total_unresolved)
