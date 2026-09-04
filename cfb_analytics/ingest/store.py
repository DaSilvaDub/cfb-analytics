"""Write helpers for the SQLite store.

Insert semantics are chosen per table for a reason:

* ``teams`` upserts, widening ``last_seen_utc`` — a team is a slowly-changing
  dimension.
* ``games`` upserts on identity but never overwrites a kickoff time with NULL.
* ``odds_snapshots`` and ``availability`` are append-only with a deterministic
  primary key, so re-running an ingest is idempotent rather than duplicating.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from cfb_analytics.sources.outlier import OddsRow
from cfb_analytics.utils import stable_id, utc_now_iso


@dataclass
class RunRecorder:
    """Records one command execution in the ``runs`` table."""

    conn: sqlite3.Connection
    command: str
    run_id: str = ""

    def __enter__(self) -> RunRecorder:
        self.run_id = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO runs (run_id, command, started_utc, status) VALUES (?, ?, ?, 'running')",
            (self.run_id, self.command, utc_now_iso()),
        )
        self.conn.commit()
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        status = "ok" if exc_type is None else "failed"
        detail = None if exc is None else f"{exc_type.__name__}: {exc}"[:500]
        self.conn.execute(
            "UPDATE runs SET finished_utc = ?, status = ?, error = ? WHERE run_id = ?",
            (utc_now_iso(), status, detail, self.run_id),
        )
        self.conn.commit()
        return False  # never swallow

    def add_rows(self, count: int) -> None:
        self.conn.execute(
            "UPDATE runs SET rows_written = rows_written + ? WHERE run_id = ?",
            (count, self.run_id),
        )

    def record_health(
        self, source: str, endpoint: str, ok: bool, rows: int = 0, detail: str = ""
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO source_health
               (run_id, source, endpoint, observed_utc, ok, rows, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (self.run_id, source, endpoint, utc_now_iso(), 1 if ok else 0, rows, detail[:300]),
        )


def upsert_team(conn: sqlite3.Connection, team: dict[str, Any]) -> None:
    now = utc_now_iso()
    payload = {
        "cfbd_id": None,
        "conference": None,
        "classification": None,
        "venue_id": None,
        **team,
        "now": now,
    }
    conn.execute(
        """INSERT INTO teams
           (team_id, cfbd_id, school, alias, market, conference, classification,
            venue_id, first_seen_utc, last_seen_utc)
           VALUES (:team_id, :cfbd_id, :school, :alias, :market, :conference, :classification,
                   :venue_id, :now, :now)
           ON CONFLICT(team_id) DO UPDATE SET
             cfbd_id = COALESCE(excluded.cfbd_id, teams.cfbd_id),
             school = COALESCE(excluded.school, teams.school),
             alias  = COALESCE(excluded.alias,  teams.alias),
             market = COALESCE(excluded.market, teams.market),
             conference = COALESCE(excluded.conference, teams.conference),
             classification = COALESCE(excluded.classification, teams.classification),
             venue_id = COALESCE(excluded.venue_id, teams.venue_id),
             last_seen_utc = excluded.last_seen_utc""",
        payload,
    )


def upsert_game(conn: sqlite3.Connection, game: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO games (game_id, season, kickoff_utc, football_date, day_of_week,
                              home_team_id, away_team_id, venue_name, network, status,
                              source, ingested_utc)
           VALUES (:game_id, :season, :kickoff_utc, :football_date, :day_of_week,
                   :home_team_id, :away_team_id, :venue_name, :network, :status,
                   'outlier', :ingested_utc)
           ON CONFLICT(game_id) DO UPDATE SET
             kickoff_utc = COALESCE(excluded.kickoff_utc, games.kickoff_utc),
             football_date = COALESCE(excluded.football_date, games.football_date),
             status      = COALESCE(excluded.status, games.status),
             network     = COALESCE(excluded.network, games.network),
             venue_name  = COALESCE(excluded.venue_name, games.venue_name),
             ingested_utc = excluded.ingested_utc""",
        {**game, "ingested_utc": utc_now_iso()},
    )


def upsert_cfbd_game(conn: sqlite3.Connection, game: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO games (game_id, season, week, season_type, kickoff_utc, football_date,
                              neutral_site, conference_game, home_team_id, away_team_id,
                              venue_name, status, home_points, away_points, completed,
                              source, ingested_utc)
           VALUES (:game_id, :season, :week, :season_type, :kickoff_utc, :football_date,
                   :neutral_site, :conference_game, :home_team_id, :away_team_id,
                   :venue_name, :status, :home_points, :away_points, :completed,
                   :source, :ingested_utc)
           ON CONFLICT(game_id) DO UPDATE SET
             season = excluded.season,
             week = excluded.week,
             season_type = excluded.season_type,
             kickoff_utc = excluded.kickoff_utc,
             football_date = excluded.football_date,
             neutral_site = excluded.neutral_site,
             conference_game = excluded.conference_game,
             home_team_id = excluded.home_team_id,
             away_team_id = excluded.away_team_id,
             venue_name = COALESCE(excluded.venue_name, games.venue_name),
             status = excluded.status,
             home_points = COALESCE(excluded.home_points, games.home_points),
             away_points = COALESCE(excluded.away_points, games.away_points),
             completed = excluded.completed,
             source = excluded.source,
             ingested_utc = excluded.ingested_utc""",
        {**game, "ingested_utc": utc_now_iso()},
    )


def upsert_venue(conn: sqlite3.Connection, venue: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO venues
           (venue_id, name, city, state, latitude, longitude, elevation_m, surface,
            dome, capacity, timezone)
           VALUES (:venue_id, :name, :city, :state, :latitude, :longitude, :elevation_m,
                   :surface, :dome, :capacity, :timezone)
           ON CONFLICT(venue_id) DO UPDATE SET
             name = COALESCE(excluded.name, venues.name),
             city = COALESCE(excluded.city, venues.city),
             state = COALESCE(excluded.state, venues.state),
             latitude = COALESCE(excluded.latitude, venues.latitude),
             longitude = COALESCE(excluded.longitude, venues.longitude),
             elevation_m = COALESCE(excluded.elevation_m, venues.elevation_m),
             surface = COALESCE(excluded.surface, venues.surface),
             dome = COALESCE(excluded.dome, venues.dome),
             capacity = COALESCE(excluded.capacity, venues.capacity),
             timezone = COALESCE(excluded.timezone, venues.timezone)""",
        venue,
    )


def upsert_team_season(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO team_seasons
           (team_id, season, source, conference, division, classification, venue_id)
           VALUES (:team_id, :season, :source, :conference, :division, :classification, :venue_id)
           ON CONFLICT(team_id, season, source) DO UPDATE SET
             conference = COALESCE(excluded.conference, team_seasons.conference),
             division = COALESCE(excluded.division, team_seasons.division),
             classification = COALESCE(excluded.classification, team_seasons.classification),
             venue_id = COALESCE(excluded.venue_id, team_seasons.venue_id)""",
        row,
    )


def insert_team_aliases(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    cursor = conn.executemany(
        """INSERT OR IGNORE INTO team_aliases (team_id, source, alias, alias_type)
           VALUES (:team_id, :source, :alias, :alias_type)""",
        payload,
    )
    return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


def _snapshot_id(kind: str, row: dict[str, Any]) -> str:
    stable = {key: value for key, value in row.items() if key != "ingested_utc"}
    return stable_id(kind, json.dumps(stable, sort_keys=True, separators=(",", ":")))


def _insert_snapshots(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    rows: Iterable[dict[str, Any]],
) -> int:
    now = utc_now_iso()
    payload = []
    for raw in rows:
        row = {**raw, "ingested_utc": now}
        row["snapshot_id"] = _snapshot_id(table, row)
        payload.append(tuple(row.get(column) for column in columns))
    if not payload:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    cursor = conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        payload,
    )
    return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


def insert_team_ratings(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    columns = (
        "snapshot_id", "season", "period", "week", "team_id", "source",
        "snapshot_scope", "provenance_mode", "as_of_utc", "ingested_utc",
        "rating", "ranking", "off_rating", "def_rating", "st_rating", "sos",
        "second_order_wins",
    )
    return _insert_snapshots(conn, "team_ratings", columns, rows)


def insert_team_advanced(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    columns = (
        "snapshot_id", "season", "week", "team_id", "side", "as_of_utc",
        "ingested_utc", "provenance_mode", "garbage_excluded", "plays", "drives",
        "ppa", "total_ppa", "success_rate", "explosiveness",
        "points_per_opportunity", "havoc", "line_yards", "stuff_rate",
        "passing_ppa", "rushing_ppa", "passing_success_rate",
        "rushing_success_rate",
    )
    return _insert_snapshots(conn, "team_season_advanced", columns, rows)


def insert_returning_production(
    conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]
) -> int:
    columns = (
        "snapshot_id", "season", "team_id", "availability_class", "ingested_utc",
        "total_ppa", "passing_ppa", "receiving_ppa", "rushing_ppa", "percent_ppa",
        "percent_passing_ppa", "percent_receiving_ppa", "percent_rushing_ppa",
        "usage", "passing_usage", "receiving_usage", "rushing_usage",
    )
    return _insert_snapshots(conn, "returning_production", columns, rows)


def insert_team_talent(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    columns = (
        "snapshot_id", "season", "team_id", "availability_class", "ingested_utc",
        "talent_composite",
    )
    return _insert_snapshots(conn, "team_talent", columns, rows)


def insert_odds(conn: sqlite3.Connection, rows: Iterable[OddsRow]) -> int:
    payload = [
        (
            row.snapshot_id, row.game_id, row.market_id, row.book, row.captured_utc,
            row.market, row.side, row.line, row.price_american, row.price_decimal,
            1 if row.is_primary else 0, row.source,
        )
        for row in rows
    ]
    if not payload:
        return 0
    cursor = conn.executemany(
        """INSERT OR IGNORE INTO odds_snapshots
           (snapshot_id, game_id, market_id, book, captured_utc, market, side,
            line, price_american, price_decimal, is_primary, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        payload,
    )
    return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


def insert_availability(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    cursor = conn.executemany(
        """INSERT OR IGNORE INTO availability
           (game_id, team_id, player_id, as_of_utc, position, position_group,
            designation, injury_type, return_date, last_updated_utc, has_news, source)
           VALUES (:game_id, :team_id, :player_id, :as_of_utc, :position, :position_group,
                   :designation, :injury_type, :return_date, :last_updated_utc,
                   :has_news, :source)""",
        payload,
    )
    return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
