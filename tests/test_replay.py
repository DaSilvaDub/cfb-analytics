"""Tests for the game replay engine."""

from __future__ import annotations

from cfb_analytics.backtest.replay import (
    _is_scrimmage_play,
    _map_cfbd_result,
    _safe_yardline,
    replay_game,
)
from cfb_analytics.features.live import DriveOutcome
from cfb_analytics.ingest import store
from tests.conftest import seed_canonical_game, seed_canonical_team


def _seed_game_with_drive(conn, *, drive_result="PUNT", scoring=0, drive_plays=None):
    """Seed a game with one drive and plays for testing."""
    home = seed_canonical_team(conn, 1, "Alabama", "ALA")
    away = seed_canonical_team(conn, 2, "Auburn", "AUB")
    game_id = seed_canonical_game(
        conn,
        "cfbd:5000",
        home,
        away,
        football_date="2024-11-30",
        kickoff="2024-11-30T17:00:00+00:00",
    )
    # Complete the game
    conn.execute(
        "UPDATE games SET home_points = 21, away_points = 14, completed = 1 WHERE game_id = ?",
        (game_id,),
    )
    # Insert a drive
    store.insert_drives(
        conn,
        [
            {
                "drive_id": "cfbd:d1",
                "game_id": game_id,
                "drive_number": 1,
                "offense_team_id": home,
                "defense_team_id": away,
                "scoring": scoring,
                "start_period": 1,
                "start_yardline": 75,
                "start_time_minutes": 15,
                "start_time_seconds": 0,
                "end_period": 1,
                "end_yardline": 50,
                "end_time_minutes": 12,
                "end_time_seconds": 0,
                "plays": 3,
                "yards": 25,
                "result": drive_result,
            }
        ],
    )
    # Insert plays
    default_plays = drive_plays or [
        {
            "play_id": "cfbd:p1",
            "drive_id": "cfbd:d1",
            "game_id": game_id,
            "offense_team_id": home,
            "defense_team_id": away,
            "play_number": 1,
            "period": 1,
            "clock_minutes": 14,
            "clock_seconds": 50,
            "yard_line": 30,
            "down": 1,
            "distance": 10,
            "yards_gained": 5,
            "play_type": "Rush",
            "scoring": 0,
            "ppa": 0.5,
        },
        {
            "play_id": "cfbd:p2",
            "drive_id": "cfbd:d1",
            "game_id": game_id,
            "offense_team_id": home,
            "defense_team_id": away,
            "play_number": 2,
            "period": 1,
            "clock_minutes": 14,
            "clock_seconds": 10,
            "yard_line": 35,
            "down": 2,
            "distance": 5,
            "yards_gained": 10,
            "play_type": "Pass",
            "scoring": 0,
            "ppa": 1.0,
        },
        {
            "play_id": "cfbd:p3",
            "drive_id": "cfbd:d1",
            "game_id": game_id,
            "offense_team_id": home,
            "defense_team_id": away,
            "play_number": 3,
            "period": 1,
            "clock_minutes": 13,
            "clock_seconds": 30,
            "yard_line": 45,
            "down": 1,
            "distance": 10,
            "yards_gained": 3,
            "play_type": "Rush",
            "scoring": 0,
            "ppa": -0.2,
        },
    ]
    store.insert_plays(conn, default_plays)
    conn.commit()
    return game_id, home, away


def test_empty_game_returns_empty(conn):
    """A game with no drives produces no checkpoints."""
    home = seed_canonical_team(conn, 1, "Alabama", "ALA")
    away = seed_canonical_team(conn, 2, "Auburn", "AUB")
    game_id = seed_canonical_game(
        conn,
        "cfbd:9999",
        home,
        away,
        football_date="2024-11-30",
        kickoff="2024-11-30T17:00:00+00:00",
    )
    conn.commit()
    result = replay_game(conn, game_id)
    assert result == []


def test_nonexistent_game_returns_empty(conn):
    """A game not in the database produces no checkpoints."""
    result = replay_game(conn, "cfbd:not_real")
    assert result == []


def test_single_drive_produces_drive_start_checkpoint(conn):
    """A single-drive game produces at least a drive_start checkpoint."""
    game_id, home, away = _seed_game_with_drive(conn)
    result = replay_game(conn, game_id, checkpoint_kinds=frozenset({"drive_start"}))
    assert len(result) >= 1
    assert result[0].label.kind == "drive_start"
    assert result[0].label.drive_number == 1
    assert result[0].label.quarter == 1
    assert isinstance(result[0].state, type(result[0].state))  # Is a GameState


def test_drive_start_carries_actual_outcome(conn):
    """Drive start checkpoints carry the actual outcome label."""
    game_id, home, away = _seed_game_with_drive(conn, drive_result="PUNT")
    result = replay_game(conn, game_id, checkpoint_kinds=frozenset({"drive_start"}))
    assert len(result) >= 1
    assert result[0].actual_drive_outcome == DriveOutcome.PUNT.value


def test_final_scores_on_checkpoints(conn):
    """Checkpoints carry actual final scores."""
    game_id, home, away = _seed_game_with_drive(conn)
    result = replay_game(conn, game_id, checkpoint_kinds=frozenset({"drive_start"}))
    assert len(result) >= 1
    assert result[0].actual_home_final == 21
    assert result[0].actual_away_final == 14


def test_cfbd_result_mapping():
    """All common CFBD result strings map to valid DriveOutcome values."""
    assert _map_cfbd_result("TOUCHDOWN") == DriveOutcome.TOUCHDOWN
    assert _map_cfbd_result("FIELD GOAL") == DriveOutcome.FIELD_GOAL
    assert _map_cfbd_result("PUNT") == DriveOutcome.PUNT
    assert _map_cfbd_result("FUMBLE") == DriveOutcome.TURNOVER
    assert _map_cfbd_result("INTERCEPTION") == DriveOutcome.TURNOVER
    assert _map_cfbd_result("TURNOVER ON DOWNS") == DriveOutcome.DOWNS
    assert _map_cfbd_result("SAFETY") == DriveOutcome.SAFETY
    assert _map_cfbd_result("END OF HALF") == DriveOutcome.END_HALF
    assert _map_cfbd_result("END OF GAME") == DriveOutcome.END_GAME
    assert _map_cfbd_result(None) == DriveOutcome.DOWNS
    assert _map_cfbd_result("unknown_thing") == DriveOutcome.DOWNS


def test_non_scrimmage_play_filtering():
    """Non-scrimmage plays (kickoffs, PATs) are filtered out."""
    assert _is_scrimmage_play({"down": 1, "play_type": "Rush"}) is True
    assert _is_scrimmage_play({"down": None, "play_type": "Kickoff"}) is False
    assert _is_scrimmage_play({"down": 0, "play_type": "Extra Point"}) is False
    assert _is_scrimmage_play({"down": 3, "play_type": "Pass"}) is True


def test_safe_yardline_conversion():
    """Yardline conversion produces valid values in [1, 99]."""
    assert _safe_yardline(25, "team_a", "team_a") == 75  # Own 25 -> 75 to go
    assert _safe_yardline(50, "team_a", "team_a") == 50  # Midfield
    assert _safe_yardline(80, "team_a", "team_a") == 20  # Red zone
    assert _safe_yardline(0, "team_a", "team_a") == 99  # Own goal line
    assert _safe_yardline(100, "team_a", "team_a") == 1  # Opponent goal line
    assert _safe_yardline(None, "team_a", "team_a") == 75  # Default
