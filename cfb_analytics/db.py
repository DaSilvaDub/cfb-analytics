"""SQLite store: connection handling and forward-only numbered migrations.

Design rules that the rest of the package depends on:

* **Append-only where values change.** Anything that can move during a week
  (odds, injury designations, weather) is keyed by ``as_of_utc`` and inserted,
  never updated. The backtest reconstructs any point in time by filtering on
  that column, which is what makes the leakage guard in ``features.asof``
  possible at all.
* **Migrations are additive and numbered.** A migration is never edited once
  committed; correcting one means adding the next.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cfb_analytics import paths

SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_utc TEXT NOT NULL
);
"""

MIGRATION_001 = """
-- Provenance for every command that writes to this database.
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    command      TEXT NOT NULL,
    started_utc  TEXT NOT NULL,
    finished_utc TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    rows_written INTEGER NOT NULL DEFAULT 0,
    source_versions_json TEXT,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS venues (
    venue_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    city       TEXT,
    state      TEXT,
    latitude   REAL,
    longitude  REAL,
    elevation_m REAL,
    surface    TEXT,
    dome       INTEGER,
    capacity   INTEGER,
    timezone   TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id     TEXT PRIMARY KEY,      -- Outlier teamId; CFBD id joined in a later migration
    cfbd_id     INTEGER,
    -- Nullable on purpose. SQLite checks NOT NULL against the proposed row
    -- BEFORE ON CONFLICT resolution, so a NOT NULL here would make the
    -- COALESCE upsert in ingest.store raise on a feed row with a missing
    -- name -- crashing a whole slate over a cosmetic field.
    school      TEXT,
    alias       TEXT,
    market      TEXT,
    conference  TEXT,
    classification TEXT,
    venue_id    TEXT REFERENCES venues(venue_id),
    first_seen_utc TEXT NOT NULL,
    last_seen_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_teams_alias ON teams(alias);

CREATE TABLE IF NOT EXISTS games (
    game_id       TEXT PRIMARY KEY,    -- Outlier eventId
    season        INTEGER,
    week          INTEGER,
    season_type   TEXT,
    kickoff_utc   TEXT NOT NULL,
    day_of_week   TEXT,
    neutral_site  INTEGER,
    conference_game INTEGER,
    home_team_id  TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id  TEXT NOT NULL REFERENCES teams(team_id),
    venue_name    TEXT,
    network       TEXT,
    status        TEXT,
    home_points   INTEGER,
    away_points   INTEGER,
    completed     INTEGER NOT NULL DEFAULT 0,
    source        TEXT NOT NULL,
    ingested_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_kickoff ON games(kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_games_slate ON games(substr(kickoff_utc, 1, 10));

-- Append-only. One row per (game, book, market, side, line) observation.
CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id     TEXT PRIMARY KEY,  -- deterministic hash: re-ingesting is idempotent
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    market_id       TEXT,
    book            TEXT NOT NULL,
    captured_utc    TEXT NOT NULL,
    market          TEXT NOT NULL CHECK (market IN ('ML','SPREAD','TOTAL')),
    side            TEXT,
    line            REAL,
    price_american  INTEGER,
    price_decimal   REAL,
    is_primary      INTEGER,
    source          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_odds_game_market ON odds_snapshots(game_id, market, captured_utc);
CREATE INDEX IF NOT EXISTS idx_odds_book ON odds_snapshots(book);

-- Append-only injury/availability designations.
CREATE TABLE IF NOT EXISTS availability (
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    team_id        TEXT NOT NULL REFERENCES teams(team_id),
    player_id      TEXT NOT NULL,
    as_of_utc      TEXT NOT NULL,
    position       TEXT,
    position_group TEXT,
    designation    TEXT CHECK (designation IN
                     ('Out','Out for Season','Doubtful','Questionable','Probable')),
    injury_type    TEXT,
    return_date    TEXT,
    last_updated_utc TEXT,
    has_news       INTEGER,
    source         TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id, player_id, as_of_utc)
);
CREATE INDEX IF NOT EXISTS idx_availability_team ON availability(team_id, as_of_utc);

-- Per-source fetch outcomes, so a degraded run is visible rather than silent.
CREATE TABLE IF NOT EXISTS source_health (
    run_id      TEXT NOT NULL,
    source      TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    observed_utc TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    rows        INTEGER,
    detail      TEXT,
    PRIMARY KEY (run_id, source, endpoint)
);
"""

MIGRATIONS: tuple[tuple[int, str, str], ...] = (
    (1, "outlier_ingestion_core", MIGRATION_001),
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    conn.executescript(SCHEMA_MIGRATIONS_DDL)
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {int(row["version"]) for row in rows}


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply every pending migration in order. Returns the versions applied."""
    from cfb_analytics.utils import utc_now_iso

    done = applied_versions(conn)
    applied: list[int] = []
    for version, name, sql in MIGRATIONS:
        if version in done:
            continue
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_utc) VALUES (?, ?, ?)",
            (version, name, utc_now_iso()),
        )
        conn.commit()
        applied.append(version)
    return applied


@contextmanager
def open_db(
    path: Path | None = None, *, migrate_on_open: bool = True
) -> Iterator[sqlite3.Connection]:
    """Open the store, applying pending migrations by default."""
    target = path or paths.database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(target)
    try:
        if migrate_on_open:
            migrate(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
