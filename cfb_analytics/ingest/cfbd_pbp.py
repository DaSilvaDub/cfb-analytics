"""Play-by-play backfill from CFBD: drives and plays for historical game replay."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import CFBDClient, parse_drive, parse_play


@dataclass(frozen=True)
class PBPBackfillSummary:
    seasons: int
    drives_written: int
    plays_written: int
    drives_skipped: int
    plays_skipped: int

    def as_text(self) -> str:
        return (
            f"PBP backfill wrote {self.drives_written} drives and "
            f"{self.plays_written} plays across {self.seasons} season(s). "
            f"Skipped {self.drives_skipped} unresolved drives, "
            f"{self.plays_skipped} unresolved plays."
        )


def _build_team_lookup(conn: sqlite3.Connection) -> dict[str, str]:
    """Map school name -> team_id for all teams in the database."""
    rows = conn.execute("SELECT team_id, school FROM teams WHERE school IS NOT NULL").fetchall()
    lookup: dict[str, str] = {}
    for row in rows:
        lookup[str(row["school"])] = str(row["team_id"])
    return lookup


def _regular_season_weeks(conn: sqlite3.Connection, season: int) -> list[int]:
    """Distinct completed regular-season weeks for a year."""
    rows = conn.execute(
        """SELECT DISTINCT week FROM games
           WHERE source = 'cfbd' AND season = ? AND season_type = 'regular'
             AND completed = 1 AND week IS NOT NULL
           ORDER BY week""",
        (season,),
    ).fetchall()
    return [int(row["week"]) for row in rows]


def _postseason_weeks(conn: sqlite3.Connection, season: int) -> list[int]:
    """Distinct completed postseason weeks for a year."""
    rows = conn.execute(
        """SELECT DISTINCT week FROM games
           WHERE source = 'cfbd' AND season = ? AND season_type = 'postseason'
             AND completed = 1 AND week IS NOT NULL
           ORDER BY week""",
        (season,),
    ).fetchall()
    return [int(row["week"]) for row in rows]


def backfill_pbp(
    conn: sqlite3.Connection,
    client: CFBDClient,
    *,
    start_year: int,
    end_year: int,
) -> PBPBackfillSummary:
    """Fetch and store drives + plays from CFBD for a range of seasons."""
    team_lookup = _build_team_lookup(conn)
    total_drives = 0
    total_plays = 0
    drives_skipped = 0
    plays_skipped = 0

    for year in range(start_year, end_year + 1):
        # Fetch all drives for the season
        raw_drives = client.fetch_drives(year)
        drive_rows: list[dict[str, Any]] = []
        for raw in raw_drives:
            parsed = parse_drive(raw)
            if parsed is None:
                drives_skipped += 1
                continue
            off_name = parsed.pop("offense_name", None)
            def_name = parsed.pop("defense_name", None)
            off_id = team_lookup.get(off_name or "") if off_name else None
            def_id = team_lookup.get(def_name or "") if def_name else None
            if off_id is None or def_id is None:
                drives_skipped += 1
                continue
            parsed["offense_team_id"] = off_id
            parsed["defense_team_id"] = def_id
            drive_rows.append(parsed)
        total_drives += store.insert_drives(conn, drive_rows)

        # Fetch plays per week (regular + postseason)
        weeks: list[tuple[int, str]] = [(w, "regular") for w in _regular_season_weeks(conn, year)]
        weeks.extend((w, "postseason") for w in _postseason_weeks(conn, year))
        for week, season_type in weeks:
            raw_plays = client.fetch_plays(year, week, season_type=season_type)
            play_rows: list[dict[str, Any]] = []
            for raw in raw_plays:
                parsed = parse_play(raw)
                if parsed is None:
                    plays_skipped += 1
                    continue
                off_name = parsed.pop("offense_name", None)
                def_name = parsed.pop("defense_name", None)
                off_id = team_lookup.get(off_name or "") if off_name else None
                def_id = team_lookup.get(def_name or "") if def_name else None
                if off_id is None or def_id is None or parsed.get("game_id") is None:
                    plays_skipped += 1
                    continue
                parsed["offense_team_id"] = off_id
                parsed["defense_team_id"] = def_id
                play_rows.append(parsed)
            total_plays += store.insert_plays(conn, play_rows)

        conn.commit()

    return PBPBackfillSummary(
        seasons=end_year - start_year + 1,
        drives_written=total_drives,
        plays_written=total_plays,
        drives_skipped=drives_skipped,
        plays_skipped=plays_skipped,
    )
