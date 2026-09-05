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
from collections.abc import Callable, Iterator
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

MIGRATION_002 = """
-- The slate a game belongs to: its US Eastern calendar date, NOT kickoff_utc[:10].
-- Stored rather than derived because SQLite has no time-zone database, so every
-- slate query would otherwise have to round-trip through Python.
ALTER TABLE games ADD COLUMN football_date TEXT;
CREATE INDEX IF NOT EXISTS idx_games_football_date ON games(football_date);
"""


def _backfill_football_date(conn: sqlite3.Connection) -> None:
    """Populate games.football_date for rows written before migration 002."""
    from cfb_analytics.utils import football_date

    rows = conn.execute(
        "SELECT game_id, kickoff_utc FROM games WHERE football_date IS NULL"
    ).fetchall()
    conn.executemany(
        "UPDATE games SET football_date = ? WHERE game_id = ?",
        [(football_date(row["kickoff_utc"]), row["game_id"]) for row in rows],
    )


MIGRATION_003 = """
-- Vig-free market view, one row per (game, market, line, side, capture).
-- Append-only like odds_snapshots: a later capture is a new row, so the
-- backtest can reconstruct what the market said at any point before kickoff.
CREATE TABLE IF NOT EXISTS market_consensus (
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    market         TEXT NOT NULL,
    line           REAL,
    side           TEXT NOT NULL,
    as_of_utc      TEXT NOT NULL,
    n_books        INTEGER NOT NULL,
    consensus_price INTEGER,
    best_price     INTEGER,
    best_book      TEXT,
    hold           REAL,
    anchor         TEXT NOT NULL,          -- sharp | all_books | none
    prob_multiplicative REAL,
    prob_shin      REAL,
    prob_power     REAL,
    prob_spread    REAL,                   -- max-min across methods
    flags          TEXT,
    PRIMARY KEY (game_id, market, line, side, as_of_utc)
);
CREATE INDEX IF NOT EXISTS idx_consensus_game ON market_consensus(game_id, market);

CREATE TABLE IF NOT EXISTS line_movement (
    game_id     TEXT NOT NULL REFERENCES games(game_id),
    market      TEXT NOT NULL,
    side        TEXT NOT NULL,
    as_of_utc   TEXT NOT NULL,
    open_line   REAL,
    open_price  INTEGER,
    current_line REAL,
    current_price INTEGER,
    move_magnitude REAL,
    move_direction TEXT,
    rlm_flag    INTEGER NOT NULL DEFAULT 0,
    -- Always 'line_only'. True RLM needs ticket/money percentages, which no
    -- free source provides; the weaker inference is labelled, never dressed up.
    rlm_basis   TEXT NOT NULL DEFAULT 'line_only',
    PRIMARY KEY (game_id, market, side, as_of_utc)
);
"""

MIGRATION_004 = """
-- Historical affiliation must be append-only by season; overwriting `teams`
-- would erase conference changes across the backfill window.
CREATE TABLE IF NOT EXISTS team_seasons (
    team_id        TEXT NOT NULL REFERENCES teams(team_id),
    season         INTEGER NOT NULL,
    source         TEXT NOT NULL,
    conference     TEXT,
    division       TEXT,
    classification TEXT,
    venue_id       TEXT REFERENCES venues(venue_id),
    PRIMARY KEY (team_id, season, source)
);
CREATE INDEX IF NOT EXISTS idx_team_seasons_season ON team_seasons(season);

CREATE TABLE IF NOT EXISTS team_aliases (
    team_id    TEXT NOT NULL REFERENCES teams(team_id),
    source     TEXT NOT NULL,
    alias      TEXT NOT NULL,
    alias_type TEXT NOT NULL,
    PRIMARY KEY (team_id, source, alias)
);
"""

MIGRATION_005 = """
-- CFBD fundamentals. `snapshot_scope` is a blocking leakage contract:
-- SP+ and SRS have no week parameter and are season-final only, while Elo is
-- captured at an explicit post-week cutoff. Consumers must never treat a
-- season-final row as knowable before a game in that same season.
CREATE TABLE IF NOT EXISTS team_ratings (
    snapshot_id      TEXT PRIMARY KEY,
    season          INTEGER NOT NULL,
    period          TEXT NOT NULL,
    week            INTEGER,
    season_type     TEXT,
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    source          TEXT NOT NULL CHECK (source IN ('sp', 'srs', 'elo_cfbd')),
    snapshot_scope  TEXT NOT NULL CHECK (snapshot_scope IN ('weekly', 'season_final')),
    provenance_mode TEXT NOT NULL DEFAULT 'reconstructed'
        CHECK (provenance_mode IN ('reconstructed', 'observed')),
    as_of_utc       TEXT NOT NULL,
    ingested_utc    TEXT NOT NULL,
    rating          REAL,
    ranking         INTEGER,
    off_rating      REAL,
    def_rating      REAL,
    st_rating       REAL,
    sos             REAL,
    second_order_wins REAL,
    CHECK ((snapshot_scope = 'weekly' AND week IS NOT NULL AND season_type IS NOT NULL
            AND period = printf('%s:week:%02d', season_type, week))
        OR (snapshot_scope = 'season_final' AND week IS NULL AND season_type IS NULL
            AND period = 'season_final'))
);
CREATE INDEX IF NOT EXISTS idx_team_ratings_asof
    ON team_ratings(team_id, source, as_of_utc);

-- One cumulative through-week snapshot per side. CFBD calls PPA "predicted
-- points added"; it is stored as PPA rather than relabelled as EPA.
CREATE TABLE IF NOT EXISTS team_season_advanced (
    snapshot_id      TEXT PRIMARY KEY,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    side            TEXT NOT NULL CHECK (side IN ('off', 'def')),
    as_of_utc       TEXT NOT NULL,
    ingested_utc    TEXT NOT NULL,
    provenance_mode TEXT NOT NULL DEFAULT 'reconstructed'
        CHECK (provenance_mode IN ('reconstructed', 'observed')),
    garbage_excluded INTEGER NOT NULL CHECK (garbage_excluded IN (0, 1)),
    plays           INTEGER,
    drives          INTEGER,
    ppa             REAL,
    total_ppa       REAL,
    success_rate    REAL,
    explosiveness   REAL,
    points_per_opportunity REAL,
    havoc           REAL,
    line_yards      REAL,
    stuff_rate      REAL,
    passing_ppa     REAL,
    rushing_ppa     REAL,
    passing_success_rate REAL,
    rushing_success_rate REAL
);
CREATE INDEX IF NOT EXISTS idx_team_advanced_asof
    ON team_season_advanced(team_id, as_of_utc);

CREATE TABLE IF NOT EXISTS returning_production (
    snapshot_id      TEXT PRIMARY KEY,
    season          INTEGER NOT NULL,
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    availability_class TEXT NOT NULL DEFAULT 'preseason'
        CHECK (availability_class = 'preseason'),
    ingested_utc    TEXT NOT NULL,
    total_ppa       REAL,
    passing_ppa     REAL,
    receiving_ppa   REAL,
    rushing_ppa     REAL,
    percent_ppa     REAL,
    percent_passing_ppa REAL,
    percent_receiving_ppa REAL,
    percent_rushing_ppa REAL,
    usage           REAL,
    passing_usage   REAL,
    receiving_usage REAL,
    rushing_usage   REAL
);
CREATE INDEX IF NOT EXISTS idx_returning_production_season
    ON returning_production(season, team_id, ingested_utc);

CREATE TABLE IF NOT EXISTS team_talent (
    snapshot_id      TEXT PRIMARY KEY,
    season          INTEGER NOT NULL,
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    availability_class TEXT NOT NULL DEFAULT 'preseason'
        CHECK (availability_class = 'preseason'),
    ingested_utc    TEXT NOT NULL,
    talent_composite REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_team_talent_season
    ON team_talent(season, team_id, ingested_utc);
"""

MIGRATION_006 = """
CREATE TABLE IF NOT EXISTS feature_rows (
    game_id            TEXT NOT NULL REFERENCES games(game_id),
    team_id            TEXT NOT NULL REFERENCES teams(team_id),
    feature_key        TEXT NOT NULL,
    feature_value      REAL NOT NULL,
    source             TEXT NOT NULL,
    availability_class TEXT NOT NULL
        CHECK (availability_class IN ('preseason', 'weekly', 'pregame')),
    source_season      INTEGER,
    source_week        INTEGER,
    source_detail      TEXT,
    generated_utc      TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id, feature_key, source)
);
CREATE INDEX IF NOT EXISTS idx_feature_rows_game ON feature_rows(game_id, team_id);
"""


MIGRATION_007 = """
-- Games link to venues by ID, not by name. Seven venue names are shared by
-- more than one venue ("Memorial Stadium" x3, "Husky Stadium" x3), so a
-- name join returned 10,973 rows for 10,465 games and would have attached
-- the wrong city's weather to those games.
ALTER TABLE games ADD COLUMN venue_id TEXT REFERENCES venues(venue_id);
CREATE INDEX IF NOT EXISTS idx_games_venue ON games(venue_id);

-- Append-only weather observations, one row per (game, capture).
-- Forecasts move, so a later capture is a new row and the backtest can ask
-- what the forecast said at any point before kickoff.
CREATE TABLE IF NOT EXISTS weather (
    game_id        TEXT NOT NULL REFERENCES games(game_id),
    as_of_utc      TEXT NOT NULL,
    hours_to_kick  REAL,
    temp_c         REAL,
    wind_kph       REAL,
    wind_gust_kph  REAL,
    wind_dir_deg   REAL,
    precip_mm      REAL,
    precip_prob    REAL,
    humidity       REAL,
    -- 1 = forecast (may still change), 0 = reanalysis of what actually happened.
    is_forecast    INTEGER NOT NULL,
    -- Indoor games have no weather. Recorded explicitly rather than left NULL,
    -- so "roof" is distinguishable from "we failed to fetch".
    is_indoor      INTEGER NOT NULL DEFAULT 0,
    source         TEXT NOT NULL,
    PRIMARY KEY (game_id, as_of_utc)
);
CREATE INDEX IF NOT EXISTS idx_weather_game ON weather(game_id);
"""


def _backfill_game_venue_id(conn: sqlite3.Connection) -> None:
    """Resolve venue_id from venue_name -- but only where the name is unique.

    Ambiguous names are left NULL rather than resolved to an arbitrary match.
    A wrong venue means a wrong latitude, which means a confident forecast for
    the wrong city; a NULL is honest and the weather ingest skips it. Rows
    ingested after this migration carry CFBD's venueId directly and need no
    guessing at all.
    """
    unambiguous = conn.execute(
        """SELECT name, MIN(venue_id) AS venue_id FROM venues
           WHERE name IS NOT NULL
           GROUP BY name HAVING COUNT(*) = 1"""
    ).fetchall()
    lookup = {str(row["name"]): str(row["venue_id"]) for row in unambiguous}
    if not lookup:
        return
    rows = conn.execute(
        "SELECT game_id, venue_name FROM games WHERE venue_id IS NULL AND venue_name IS NOT NULL"
    ).fetchall()
    updates = [
        (lookup[str(row["venue_name"])], row["game_id"])
        for row in rows
        if str(row["venue_name"]) in lookup
    ]
    conn.executemany("UPDATE games SET venue_id = ? WHERE game_id = ?", updates)


MIGRATION_008 = """
-- Player dimension. Unlike availability (injuries), names ARE stored here:
-- this is public roster/bio data feeding human-readable output ("QB1 is
-- Jalen Milroe"), not injury-adjacent data, and the plan's report sections
-- name specific players by design.
CREATE TABLE IF NOT EXISTS players (
    player_id      TEXT PRIMARY KEY,   -- cfbd:<id>
    name           TEXT NOT NULL,
    first_seen_utc TEXT NOT NULL,
    last_seen_utc  TEXT NOT NULL
);

-- Roster attributes are season-scoped, not a single mutable dimension row:
-- a player who is a Junior in 2025 is a Senior in 2026, and CFBD's own
-- roster feed is season-parameterised for exactly this reason.
CREATE TABLE IF NOT EXISTS player_seasons (
    player_id    TEXT NOT NULL REFERENCES players(player_id),
    season       INTEGER NOT NULL,
    team_id      TEXT NOT NULL REFERENCES teams(team_id),
    position     TEXT,
    -- CFBD's 'year' field: an integer years-in-program count (1=freshman,
    -- 4/5=senior/grad), not an FR/SO/JR/SR enum. Treated as a numeric proxy
    -- for experience, not eligibility class.
    class_year   INTEGER,
    height_in    INTEGER,
    weight_lb    INTEGER,
    home_state   TEXT,
    source       TEXT NOT NULL,
    ingested_utc TEXT NOT NULL,
    PRIMARY KEY (player_id, season)
);
CREATE INDEX IF NOT EXISTS idx_player_seasons_team
    ON player_seasons(team_id, season, position);

-- Per-game passing box score. One row per (game, player); CFBD reports a
-- completed final line, not an evolving one, so INSERT OR REPLACE is safe --
-- unlike odds or weather, there is no "a later capture changed the number".
CREATE TABLE IF NOT EXISTS player_game_passing (
    game_id       TEXT NOT NULL REFERENCES games(game_id),
    team_id       TEXT NOT NULL REFERENCES teams(team_id),
    player_id     TEXT NOT NULL REFERENCES players(player_id),
    season        INTEGER NOT NULL,
    week          INTEGER,
    completions   INTEGER,
    attempts      INTEGER,
    yards         INTEGER,
    avg_yards     REAL,
    touchdowns    INTEGER,
    interceptions INTEGER,
    qbr           REAL,
    source        TEXT NOT NULL,
    ingested_utc  TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_player_game_passing_team
    ON player_game_passing(team_id, season, week);
"""

# (version, name, SQL, optional Python step run after the SQL in the same transaction).
# The Python hook exists because some backfills need the IANA time-zone database,
# which SQLite does not have.
MIGRATION_009 = """
-- Internally-fit team ratings (ridge, and eventually Elo), kept separate from
-- the CFBD-sourced team_ratings table rather than widened into it: that
-- table's schema is a specific CFBD snapshot contract (period must be
-- 'week:NN' or 'season_final', source CHECK restricted to CFBD's own three
-- providers), which does not naturally fit a rating produced as of an
-- arbitrary kickoff cutoff for a walk-forward backtest.
CREATE TABLE IF NOT EXISTS internal_team_ratings (
    season          INTEGER NOT NULL,
    -- The exact leakage cutoff used for this fit: only games with
    -- kickoff_utc strictly before this were used. NOT a CFBD week label,
    -- since a walk-forward backtest fits at arbitrary points in time.
    as_of_utc       TEXT NOT NULL,
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    model           TEXT NOT NULL CHECK (model IN ('internal_ridge')),
    offense         REAL NOT NULL,
    defense         REAL NOT NULL,
    team_games      INTEGER NOT NULL,
    ridge_lambda    REAL,
    home_field_advantage REAL,
    league_avg_points REAL,
    -- Total games used across the WHOLE league fit (not just this team's own
    -- games), so a low-data early-season fit is visible on every row.
    n_games_in_fit  INTEGER NOT NULL,
    generated_utc   TEXT NOT NULL,
    PRIMARY KEY (season, as_of_utc, team_id, model)
);
CREATE INDEX IF NOT EXISTS idx_internal_ratings_team
    ON internal_team_ratings(team_id, season, as_of_utc);
"""

MIGRATIONS: tuple[tuple[int, str, str, Callable[[sqlite3.Connection], None] | None], ...] = (
    (1, "outlier_ingestion_core", MIGRATION_001, None),
    (2, "games_football_date", MIGRATION_002, _backfill_football_date),
    (3, "market_consensus_and_movement", MIGRATION_003, None),
    (4, "cfbd_team_history", MIGRATION_004, None),
    (5, "cfbd_fundamentals", MIGRATION_005, None),
    (6, "feature_rows", MIGRATION_006, None),
    (7, "game_venue_id_and_weather", MIGRATION_007, _backfill_game_venue_id),
    (8, "players_and_passing", MIGRATION_008, None),
    (9, "internal_team_ratings", MIGRATION_009, None),
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
    for version, name, sql, hook in MIGRATIONS:
        if version in done:
            continue
        conn.executescript(sql)
        if hook is not None:
            hook(conn)
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
