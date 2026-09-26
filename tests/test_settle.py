"""Tests for post-game settlement and calibration metrics."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pytest

from cfb_analytics import cli, db
from cfb_analytics.backtest.settle import (
    _compute_calibration_metrics,
    settle_slate,
)


def test_calibration_metrics_empty() -> None:
    brier, ece, log_loss = _compute_calibration_metrics([])
    assert brier is None
    assert ece is None
    assert log_loss is None


def test_calibration_metrics_perfect() -> None:
    obs = [(1.0, 1), (0.0, 0)]
    brier, ece, log_loss = _compute_calibration_metrics(obs)
    assert brier is not None and brier == 0.0
    assert ece is not None and ece == 0.0
    assert log_loss is not None and log_loss < 1e-4


def test_calibration_metrics_uninformative() -> None:
    obs = [(0.5, 1), (0.5, 0)]
    brier, ece, log_loss = _compute_calibration_metrics(obs)
    assert brier is not None and pytest.approx(brier, rel=1e-3) == 0.25
    assert log_loss is not None and pytest.approx(log_loss, rel=1e-3) == 0.6931


@pytest.fixture
def settled_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)

    from cfb_analytics.ingest import store

    now_str = "2026-09-01T00:00:00+00:00"
    for tid, school, alias in [
        ("cfbd:1", "Alabama", "ALA"),
        ("cfbd:2", "Auburn", "AUB"),
        ("cfbd:3", "Georgia", "UGA"),
        ("cfbd:4", "Florida", "UF"),
    ]:
        conn.execute(
            """INSERT INTO teams (team_id, school, alias, first_seen_utc, last_seen_utc)
               VALUES (?, ?, ?, ?, ?)""",
            (tid, school, alias, now_str, now_str),
        )

    # Insert 2 games for 2026-09-12
    # Game 1: ALA (Home) 30, AUB (Away) 20 (Favorite ALA wins)
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "g1",
            "season": 2026,
            "football_date": "2026-09-12",
            "kickoff_utc": "2026-09-12T19:00:00+00:00",
            "home_team_id": "cfbd:1",
            "away_team_id": "cfbd:2",
            "home_points": 30,
            "away_points": 20,
            "completed": 1,
            "status": "completed",
        },
    )
    # Game 2: UGA (Home) 14, UF (Away) 21 (Favorite UGA loses -> Upset)
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "g2",
            "season": 2026,
            "football_date": "2026-09-12",
            "kickoff_utc": "2026-09-12T20:00:00+00:00",
            "home_team_id": "cfbd:3",
            "away_team_id": "cfbd:4",
            "home_points": 14,
            "away_points": 21,
            "completed": 1,
            "status": "completed",
        },
    )

    as_of = "2026-09-12T10:00:00+00:00"
    # Insert ML consensus for Game 1 (ALA 75% fav, AUB 25%)
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g1', 'ML', 'HOME', 0.0, ?, 'all_books', -300, 0.75, 5)""",
        (as_of,),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g1', 'ML', 'AWAY', 0.0, ?, 'all_books', 250, 0.25, 5)""",
        (as_of,),
    )

    # Insert ML consensus for Game 2 (UGA 70% fav, UF 30%)
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g2', 'ML', 'HOME', 0.0, ?, 'all_books', -230, 0.70, 5)""",
        (as_of,),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g2', 'ML', 'AWAY', 0.0, ?, 'all_books', 190, 0.30, 5)""",
        (as_of,),
    )

    # Insert Totals consensus for Game 1 (Line 45.5 -> 30+20=50 is OVER)
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g1', 'TOTAL', 'OVER', 45.5, ?, 'all_books', -110, 0.50, 5)""",
        (as_of,),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g1', 'TOTAL', 'UNDER', 45.5, ?, 'all_books', -110, 0.50, 5)""",
        (as_of,),
    )

    # Insert Totals consensus for Game 2 (Line 42.5 -> 14+21=35 is UNDER)
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g2', 'TOTAL', 'OVER', 42.5, ?, 'all_books', -110, 0.50, 5)""",
        (as_of,),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g2', 'TOTAL', 'UNDER', 42.5, ?, 'all_books', -110, 0.50, 5)""",
        (as_of,),
    )

    # Insert Team Props for Game 1: ALA line 27.5 OVER (prob 0.60, scored 30 -> HIT)
    conn.execute(
        """INSERT INTO team_prop_consensus
           (game_id, team_id, market, line, side, as_of_utc, n_books, consensus_price, prob_shin)
           VALUES ('g1', 'cfbd:1', 'team_total_points', 27.5, 'OVER',
                   '2026-09-12T10:00:00+00:00', 4, -150, 0.60)"""
    )
    conn.execute(
        """INSERT INTO team_prop_consensus
           (game_id, team_id, market, line, side, as_of_utc, n_books, consensus_price, prob_shin)
           VALUES ('g1', 'cfbd:1', 'team_total_points', 27.5, 'UNDER',
                   '2026-09-12T10:00:00+00:00', 4, 120, 0.40)"""
    )

    conn.commit()
    return conn


def test_settle_slate_computation(settled_db: sqlite3.Connection) -> None:
    settlement = settle_slate(settled_db, "2026-09-12")

    assert settlement.slate_date == "2026-09-12"
    assert settlement.games_completed == 2
    assert settlement.ml_games == 2
    assert settlement.ml_fav_total == 2
    assert settlement.ml_fav_correct == 1
    assert settlement.ml_fav_win_pct == 50.0

    # 1 upset: UF over UGA
    assert len(settlement.upsets) == 1
    assert settlement.upsets[0].favorite == "UGA"
    assert settlement.upsets[0].underdog == "UF"

    # Totals: 1 over, 1 under, 0 push
    assert settlement.totals_overs == 1
    assert settlement.totals_unders == 1
    assert settlement.totals_pushes == 0

    # Team props: 1 evaluated, 1 hit
    assert settlement.team_props_evaluated == 1
    assert settlement.team_props_fav_hits == 1

    # Text and Dict formatting
    text = settlement.as_text()
    assert "SLATE SETTLEMENT REPORT - 2026-09-12" in text
    assert "Overall Favorite Record: 1/2" in text
    assert "UF      def. UGA" in text
    assert "Brier Score" in text

    d = settlement.as_dict()
    assert d["slate_date"] == "2026-09-12"
    assert d["is_actionable"] is False
    assert len(d["upsets"]) == 1


def test_settle_cli(settled_db: sqlite3.Connection, capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("cfb_analytics.paths.database_path") as mock_path,
        patch("cfb_analytics.db.open_db") as mock_open,
    ):
        mock_path.return_value.exists.return_value = True
        mock_open.return_value.__enter__.return_value = settled_db
        mock_open.return_value.__exit__.return_value = False

        # Text mode
        rc = cli.main(["settle", "--date", "2026-09-12"])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "SLATE SETTLEMENT REPORT - 2026-09-12" in captured

        # JSON mode
        rc_json = cli.main(["settle", "--date", "2026-09-12", "--json"])
        assert rc_json == 0
        json_str = capsys.readouterr().out
        payload = json.loads(json_str)
        assert payload["slate_date"] == "2026-09-12"
        assert payload["ml_fav_total"] == 2
        assert payload["is_actionable"] is False
