from __future__ import annotations

import pytest

from cfb_analytics import db
from cfb_analytics.ingest import store
from cfb_analytics.sources.outlier import parse_odds_rows

CAPTURED = "2026-09-01T00:00:00+00:00"


def _seed_game(conn):
    store.upsert_team(
        conn, {"team_id": "t-home", "school": "Tulsa", "alias": "TLSA", "market": "Tulsa"})
    store.upsert_team(
        conn, {"team_id": "t-away", "school": "Okla St", "alias": "OKST", "market": "OKC"})
    store.upsert_game(conn, {
        "game_id": "evt-1", "season": 2026, "kickoff_utc": "2026-09-06T02:30:00+00:00",
        "football_date": "2026-09-05", "day_of_week": 5,
        "home_team_id": "t-home", "away_team_id": "t-away",
        "venue_name": "Stadium", "network": "ESPN", "status": "pregame",
    })


class TestMigrations:
    def test_creates_expected_tables(self, conn):
        names = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"runs", "teams", "games", "odds_snapshots", "availability",
                "source_health", "schema_migrations", "team_seasons",
                "team_aliases"} <= names

    def test_is_idempotent(self, conn):
        assert db.migrate(conn) == []

    def test_records_applied_version(self, conn):
        versions = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
        assert versions == {1, 2, 3, 4}

    def test_foreign_keys_are_enforced(self, conn):
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO games (game_id, kickoff_utc, home_team_id, away_team_id, source,"
                " ingested_utc) VALUES ('g','2026-01-01T00:00:00+00:00','nope','nope2',"
                "'outlier','now')")

    def test_designation_check_constraint_rejects_unknown_value(self, conn):
        import sqlite3
        _seed_game(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO availability (game_id, team_id, player_id, as_of_utc,
                   designation, source) VALUES ('evt-1','t-home','p1',?, 'Injured','outlier')""",
                (CAPTURED,))


class TestTeamUpsert:
    def test_does_not_clobber_known_values_with_null(self, conn):
        store.upsert_team(
            conn, {"team_id": "t", "school": "Tulsa", "alias": "TLSA", "market": "T"})
        store.upsert_team(conn, {"team_id": "t", "school": None, "alias": None, "market": None})
        row = conn.execute("SELECT school, alias FROM teams WHERE team_id='t'").fetchone()
        assert row["school"] == "Tulsa" and row["alias"] == "TLSA"

    def test_advances_last_seen(self, conn):
        store.upsert_team(conn, {"team_id": "t", "school": "A", "alias": "A", "market": "A"})
        first = conn.execute("SELECT first_seen_utc, last_seen_utc FROM teams").fetchone()
        assert first["first_seen_utc"] == first["last_seen_utc"]


class TestOddsInsert:
    def test_is_idempotent_across_reruns(self, conn, moneyline_market):
        _seed_game(conn)
        rows = parse_odds_rows("evt-1", [moneyline_market], CAPTURED)
        first = store.insert_odds(conn, rows)
        second = store.insert_odds(conn, rows)
        total = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
        assert first == len(rows)
        assert second == 0
        assert total == len(rows)

    def test_a_later_capture_is_a_new_row_not_an_overwrite(self, conn, moneyline_market):
        _seed_game(conn)
        store.insert_odds(conn, parse_odds_rows("evt-1", [moneyline_market], CAPTURED))
        store.insert_odds(conn, parse_odds_rows(
            "evt-1", [moneyline_market], "2026-09-02T00:00:00+00:00"))
        captures = {r["captured_utc"] for r in conn.execute(
            "SELECT DISTINCT captured_utc FROM odds_snapshots")}
        assert len(captures) == 2

    def test_market_check_constraint_rejects_unknown_code(self, conn):
        import sqlite3
        _seed_game(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO odds_snapshots (snapshot_id, game_id, book, captured_utc,
                   market, source) VALUES ('s','evt-1','X',?, 'PROP','outlier')""",
                (CAPTURED,))


class TestRunRecorder:
    def test_marks_success(self, conn):
        with store.RunRecorder(conn, "ingest") as run:
            run.add_rows(5)
        row = conn.execute("SELECT status, rows_written FROM runs").fetchone()
        assert row["status"] == "ok" and row["rows_written"] == 5

    def test_records_failure_and_reraises(self, conn):
        with pytest.raises(ValueError), store.RunRecorder(conn, "ingest"):
            raise ValueError("boom")
        row = conn.execute("SELECT status, error FROM runs").fetchone()
        assert row["status"] == "failed" and "boom" in row["error"]

    def test_records_source_health(self, conn):
        with store.RunRecorder(conn, "ingest") as run:
            run.record_health("outlier", "schedule", ok=True, rows=137)
        row = conn.execute("SELECT ok, rows FROM source_health").fetchone()
        assert row["ok"] == 1 and row["rows"] == 137


class TestMigration002Backfill:
    """Migration 002 adds games.football_date and backfills existing rows.

    The backfill needs the IANA time-zone database, which SQLite lacks, so it
    runs as a Python hook rather than SQL.
    """

    def test_backfills_rows_written_before_the_migration(self, tmp_path):
        from cfb_analytics.db import MIGRATIONS, _connect, applied_versions, migrate

        path = tmp_path / "old.sqlite3"
        conn = _connect(path)
        # Apply only migration 001, simulating a store created before 002.
        version, name, sql, _hook = MIGRATIONS[0]
        applied_versions(conn)
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_utc) VALUES (?, ?, 'x')",
            (version, name),
        )
        store.upsert_team(conn, {"team_id": "h", "school": "H", "alias": "H", "market": "H"})
        store.upsert_team(conn, {"team_id": "a", "school": "A", "alias": "A", "market": "A"})
        conn.execute(
            """INSERT INTO games (game_id, kickoff_utc, home_team_id, away_team_id,
               source, ingested_utc)
               VALUES ('late', '2026-09-06T02:30:00+00:00', 'h', 'a', 'outlier', 'x')"""
        )
        conn.commit()

        assert migrate(conn) == [2, 3, 4]

        row = conn.execute(
            "SELECT football_date FROM games WHERE game_id = 'late'").fetchone()
        # 02:30Z Sunday is 10:30pm ET Saturday: it belongs to the Saturday slate.
        assert row["football_date"] == "2026-09-05"
        conn.close()
