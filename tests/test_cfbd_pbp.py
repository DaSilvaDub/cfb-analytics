"""Tests for play-by-play ingestion: migration, parsing, and storage."""

from __future__ import annotations

from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import parse_drive, parse_play
from tests.conftest import seed_canonical_game, seed_canonical_team


def test_migration_013_creates_tables(conn):
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "drives" in tables
    assert "plays" in tables


def test_parse_drive_golden():
    raw = {
        "id": 42,
        "gameId": 100,
        "driveNumber": 3,
        "offense": "Alabama",
        "defense": "Auburn",
        "scoring": True,
        "startTime": {"period": 1, "minutes": 12, "seconds": 30},
        "endTime": {"period": 1, "minutes": 8, "seconds": 15},
        "startYardline": 75,
        "endYardline": 0,
        "plays": 8,
        "yards": 75,
        "driveResult": "TOUCHDOWN",
    }
    result = parse_drive(raw)
    assert result is not None
    assert result["drive_id"] == "cfbd:42"
    assert result["game_id"] == "cfbd:100"
    assert result["drive_number"] == 3
    assert result["offense_name"] == "Alabama"
    assert result["defense_name"] == "Auburn"
    assert result["scoring"] == 1
    assert result["start_period"] == 1
    assert result["start_yardline"] == 75
    assert result["start_time_minutes"] == 12
    assert result["start_time_seconds"] == 30
    assert result["end_period"] == 1
    assert result["end_yardline"] == 0
    assert result["end_time_minutes"] == 8
    assert result["end_time_seconds"] == 15
    assert result["plays"] == 8
    assert result["yards"] == 75
    assert result["result"] == "TOUCHDOWN"


def test_parse_drive_missing_id():
    assert parse_drive({"gameId": 1, "driveNumber": 1}) is None
    assert parse_drive({"id": "", "gameId": 1, "driveNumber": 1}) is None


def test_parse_play_golden():
    raw = {
        "id": 999,
        "driveId": 42,
        "gameId": 100,
        "offense": "Alabama",
        "defense": "Auburn",
        "playNumber": 5,
        "period": 2,
        "clock": {"minutes": 7, "seconds": 45},
        "yardLine": 35,
        "down": 2,
        "distance": 7,
        "yardsGained": 12,
        "playType": "Rush",
        "scoring": False,
        "ppa": 1.25,
    }
    result = parse_play(raw)
    assert result is not None
    assert result["play_id"] == "cfbd:999"
    assert result["drive_id"] == "cfbd:42"
    assert result["game_id"] == "cfbd:100"
    assert result["play_number"] == 5
    assert result["period"] == 2
    assert result["clock_minutes"] == 7
    assert result["clock_seconds"] == 45
    assert result["yard_line"] == 35
    assert result["down"] == 2
    assert result["distance"] == 7
    assert result["yards_gained"] == 12
    assert result["play_type"] == "Rush"
    assert result["scoring"] == 0
    assert result["ppa"] == 1.25


def test_parse_play_missing_id():
    assert parse_play({"driveId": 1, "playNumber": 1}) is None
    assert parse_play({"id": "", "driveId": 1, "playNumber": 1}) is None


def test_insert_drives_idempotent(conn):
    team_a = seed_canonical_team(conn, 1, "Alabama", "ALA")
    team_b = seed_canonical_team(conn, 2, "Auburn", "AUB")
    game_id = seed_canonical_game(
        conn,
        "cfbd:100",
        team_a,
        team_b,
        football_date="2024-11-30",
        kickoff="2024-11-30T17:00:00+00:00",
    )
    row = {
        "drive_id": "cfbd:42",
        "game_id": game_id,
        "drive_number": 1,
        "offense_team_id": team_a,
        "defense_team_id": team_b,
        "scoring": 0,
        "start_period": 1,
        "start_yardline": 75,
        "start_time_minutes": 15,
        "start_time_seconds": 0,
        "end_period": 1,
        "end_yardline": 50,
        "end_time_minutes": 12,
        "end_time_seconds": 30,
        "plays": 3,
        "yards": 25,
        "result": "PUNT",
    }
    written1 = store.insert_drives(conn, [row])
    written2 = store.insert_drives(conn, [row])
    assert written1 >= 1
    assert written2 == 0  # INSERT OR IGNORE: duplicate silently ignored
    count = conn.execute("SELECT COUNT(*) AS n FROM drives").fetchone()["n"]
    assert count == 1


def test_insert_plays_idempotent(conn):
    team_a = seed_canonical_team(conn, 1, "Alabama", "ALA")
    team_b = seed_canonical_team(conn, 2, "Auburn", "AUB")
    game_id = seed_canonical_game(
        conn,
        "cfbd:100",
        team_a,
        team_b,
        football_date="2024-11-30",
        kickoff="2024-11-30T17:00:00+00:00",
    )
    # Need a drive first (FK)
    store.insert_drives(
        conn,
        [
            {
                "drive_id": "cfbd:42",
                "game_id": game_id,
                "drive_number": 1,
                "offense_team_id": team_a,
                "defense_team_id": team_b,
                "scoring": 0,
                "start_period": 1,
                "start_yardline": 75,
                "start_time_minutes": 15,
                "start_time_seconds": 0,
                "end_period": 1,
                "end_yardline": 50,
                "end_time_minutes": 12,
                "end_time_seconds": 30,
                "plays": 3,
                "yards": 25,
                "result": "PUNT",
            }
        ],
    )
    play_row = {
        "play_id": "cfbd:999",
        "drive_id": "cfbd:42",
        "game_id": game_id,
        "offense_team_id": team_a,
        "defense_team_id": team_b,
        "play_number": 1,
        "period": 1,
        "clock_minutes": 14,
        "clock_seconds": 50,
        "yard_line": 25,
        "down": 1,
        "distance": 10,
        "yards_gained": 5,
        "play_type": "Rush",
        "scoring": 0,
        "ppa": 0.5,
    }
    written1 = store.insert_plays(conn, [play_row])
    written2 = store.insert_plays(conn, [play_row])
    assert written1 >= 1
    assert written2 == 0
    count = conn.execute("SELECT COUNT(*) AS n FROM plays").fetchone()["n"]
    assert count == 1


def test_insert_empty_returns_zero(conn):
    assert store.insert_drives(conn, []) == 0
    assert store.insert_plays(conn, []) == 0
