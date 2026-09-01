"""Write helpers for the SQLite store.

Insert semantics are chosen per table for a reason:

* ``teams`` upserts, widening ``last_seen_utc`` — a team is a slowly-changing
  dimension.
* ``games`` upserts on identity but never overwrites a kickoff time with NULL.
* ``odds_snapshots`` and ``availability`` are append-only with a deterministic
  primary key, so re-running an ingest is idempotent rather than duplicating.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from cfb_analytics.sources.outlier import OddsRow
from cfb_analytics.utils import utc_now_iso


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
    conn.execute(
        """INSERT INTO teams (team_id, school, alias, market, first_seen_utc, last_seen_utc)
           VALUES (:team_id, :school, :alias, :market, :now, :now)
           ON CONFLICT(team_id) DO UPDATE SET
             school = COALESCE(excluded.school, teams.school),
             alias  = COALESCE(excluded.alias,  teams.alias),
             market = COALESCE(excluded.market, teams.market),
             last_seen_utc = excluded.last_seen_utc""",
        {**team, "now": now},
    )


def upsert_game(conn: sqlite3.Connection, game: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO games (game_id, season, kickoff_utc, day_of_week, home_team_id,
                              away_team_id, venue_name, network, status, source, ingested_utc)
           VALUES (:game_id, :season, :kickoff_utc, :day_of_week, :home_team_id,
                   :away_team_id, :venue_name, :network, :status, 'outlier', :ingested_utc)
           ON CONFLICT(game_id) DO UPDATE SET
             kickoff_utc = COALESCE(excluded.kickoff_utc, games.kickoff_utc),
             status      = COALESCE(excluded.status, games.status),
             network     = COALESCE(excluded.network, games.network),
             venue_name  = COALESCE(excluded.venue_name, games.venue_name),
             ingested_utc = excluded.ingested_utc""",
        {**game, "ingested_utc": utc_now_iso()},
    )


def insert_odds(conn: sqlite3.Connection, rows: Iterable[OddsRow]) -> int:
    payload = [
        (
            row.snapshot_id, row.game_id, row.market_id, row.book, row.captured_utc,
            row.market, row.side, row.line, row.price_american, row.price_decimal,
            1 if row.is_primary else 0, "outlier",
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
