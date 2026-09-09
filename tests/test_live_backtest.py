"""Tests for the live backtest harness walk-forward."""

from __future__ import annotations

from cfb_analytics.backtest.live_harness import (
    run_live_walk_forward,
)
from cfb_analytics.ingest import store
from tests.conftest import seed_canonical_team


def _seed_season_with_pbp(conn, *, season=2024, weeks=2, games_per_week=1):
    """Seed a mini-season with games, drives, and plays for harness testing."""
    home = seed_canonical_team(conn, 1, "Alabama", "ALA")
    away = seed_canonical_team(conn, 2, "Auburn", "AUB")
    game_ids = []

    for week in range(1, weeks + 1):
        for g in range(games_per_week):
            gid = f"cfbd:{season}{week:02d}{g:02d}"
            kickoff = f"{season}-09-{week * 7:02d}T17:00:00+00:00"
            store.upsert_cfbd_game(
                conn,
                {
                    "game_id": gid,
                    "season": season,
                    "week": week,
                    "season_type": "regular",
                    "kickoff_utc": kickoff,
                    "football_date": f"{season}-09-{week * 7:02d}",
                    "neutral_site": 0,
                    "conference_game": 0,
                    "home_team_id": home,
                    "away_team_id": away,
                    "venue_name": "Stadium",
                    "venue_id": None,
                    "status": "completed",
                    "home_points": 28,
                    "away_points": 21,
                    "completed": 1,
                    "source": "cfbd",
                },
            )

            # Add a drive with plays for this game
            did = f"cfbd:d{season}{week:02d}{g:02d}"
            store.insert_drives(
                conn,
                [
                    {
                        "drive_id": did,
                        "game_id": gid,
                        "drive_number": 1,
                        "offense_team_id": home,
                        "defense_team_id": away,
                        "scoring": 0,
                        "start_period": 1,
                        "start_yardline": 75,
                        "start_time_minutes": 15,
                        "start_time_seconds": 0,
                        "end_period": 1,
                        "end_yardline": 50,
                        "end_time_minutes": 12,
                        "end_time_seconds": 0,
                        "plays": 2,
                        "yards": 25,
                        "result": "PUNT",
                    }
                ],
            )
            store.insert_plays(
                conn,
                [
                    {
                        "play_id": f"cfbd:p{season}{week:02d}{g:02d}_1",
                        "drive_id": did,
                        "game_id": gid,
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
                        "play_id": f"cfbd:p{season}{week:02d}{g:02d}_2",
                        "drive_id": did,
                        "game_id": gid,
                        "offense_team_id": home,
                        "defense_team_id": away,
                        "play_number": 2,
                        "period": 1,
                        "clock_minutes": 14,
                        "clock_seconds": 10,
                        "yard_line": 35,
                        "down": 2,
                        "distance": 5,
                        "yards_gained": 3,
                        "play_type": "Rush",
                        "scoring": 0,
                        "ppa": -0.2,
                    },
                ],
            )
            game_ids.append(gid)
    conn.commit()
    return game_ids, home, away


def test_walk_forward_skips_no_pbp(conn):
    """Games without PBP data are counted as skipped."""
    home = seed_canonical_team(conn, 1, "Alabama", "ALA")
    away = seed_canonical_team(conn, 2, "Auburn", "AUB")
    # Game with no drives/plays
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "cfbd:nodata01",
            "season": 2024,
            "week": 1,
            "season_type": "regular",
            "kickoff_utc": "2024-09-07T17:00:00+00:00",
            "football_date": "2024-09-07",
            "neutral_site": 0,
            "conference_game": 0,
            "home_team_id": home,
            "away_team_id": away,
            "venue_name": "Stadium",
            "venue_id": None,
            "status": "completed",
            "home_points": 28,
            "away_points": 21,
            "completed": 1,
            "source": "cfbd",
        },
    )
    conn.commit()
    run = run_live_walk_forward(conn, [2024])
    assert run.games_skipped_no_pbp >= 1
    assert run.games_replayed == 0


def test_walk_forward_produces_predictions(conn):
    """Walk-forward with PBP data produces win prob and totals predictions."""
    game_ids, home, away = _seed_season_with_pbp(conn, weeks=3)
    run = run_live_walk_forward(conn, [2024])
    # At least the games with PBP should produce something
    assert run.games_replayed >= 1
    # With drive_start checkpoints, we should have predictions
    assert len(run.win_prob_predictions) >= 1
    # Each prediction should have valid fields
    for wp in run.win_prob_predictions:
        assert 0.0 <= wp.predicted_home_win_prob <= 1.0
        assert isinstance(wp.actual_home_won, bool)


def test_walk_forward_drive_outcome_predictions(conn):
    """Walk-forward produces drive outcome predictions at drive starts."""
    game_ids, home, away = _seed_season_with_pbp(conn, weeks=3)
    run = run_live_walk_forward(conn, [2024])
    assert len(run.drive_outcome_predictions) >= 1
    for dp in run.drive_outcome_predictions:
        # Probabilities should be valid
        total = dp.predicted_td + dp.predicted_fg + dp.predicted_punt
        total += dp.predicted_turnover + dp.predicted_safety
        assert abs(total - 1.0) < 0.01  # ~sums to 1
        assert dp.actual_outcome != ""


def test_walk_forward_totals_predictions(conn):
    """Walk-forward produces team totals predictions."""
    game_ids, home, away = _seed_season_with_pbp(conn, weeks=3)
    run = run_live_walk_forward(conn, [2024])
    assert len(run.totals_predictions) >= 1
    for tp in run.totals_predictions:
        assert tp.projected_total >= 0
        assert tp.actual_total >= 0


def test_walk_forward_empty_season(conn):
    """Walk-forward on a season with no games produces empty results."""
    run = run_live_walk_forward(conn, [1900])
    assert run.games_replayed == 0
    assert run.games_skipped_no_pbp == 0
    assert len(run.win_prob_predictions) == 0


def test_leakage_guard(conn):
    """Pregame margin uses only prior-week games (leakage cutoff)."""
    # Seed week 1 (will be training data for week 2)
    game_ids, home, away = _seed_season_with_pbp(conn, weeks=2)
    run = run_live_walk_forward(conn, [2024])
    # Week 1 predictions use pregame_margin=0.0 (no prior games)
    # but week 2 might use week 1 data
    week1_preds = [p for p in run.win_prob_predictions if p.week == 1]
    for p in week1_preds:
        assert p.pregame_home_margin == 0.0  # No prior games -> default 0
