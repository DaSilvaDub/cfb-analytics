"""A tied final score is a push, not a favorite loss."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from cfb_analytics import cli, db
from cfb_analytics.backtest.settle import settle_completed_slates, settle_slate
from cfb_analytics.ingest import store
from cfb_analytics.utils import FOOTBALL_TZ


def _slate(conn: sqlite3.Connection, *, home: int, away: int, prob: float = 0.70) -> None:
    now = "2026-11-01T00:00:00+00:00"
    for tid, school, alias in (("cfbd:1", "Home U", "HOME"), ("cfbd:2", "Away U", "AWAY")):
        conn.execute(
            """INSERT INTO teams (team_id, school, alias, first_seen_utc, last_seen_utc)
               VALUES (?, ?, ?, ?, ?)""",
            (tid, school, alias, now, now),
        )
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "g-tie",
            "season": 2026,
            "football_date": "2026-11-07",
            "kickoff_utc": "2026-11-07T20:00:00+00:00",
            "home_team_id": "cfbd:1",
            "away_team_id": "cfbd:2",
            "home_points": home,
            "away_points": away,
            "completed": 1,
            "status": "completed",
        },
    )
    as_of = "2026-11-07T12:00:00+00:00"
    away_prob = round(1.0 - prob, 2)
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g-tie', 'ML', 'HOME', 0.0, ?, 'all_books', -230, ?, 5)""",
        (as_of, prob),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g-tie', 'ML', 'AWAY', 0.0, ?, 'all_books', 190, ?, 5)""",
        (as_of, away_prob),
    )
    conn.commit()


def test_tie_is_a_push_excluded_from_record_upsets_and_brier(conn) -> None:
    """Audit fixture: 24-24, home shin 0.70. Previously fav_correct=0, upsets=1, brier=0.49."""
    _slate(conn, home=24, away=24, prob=0.70)
    settlement = settle_slate(conn, "2026-11-07")
    assert settlement.ml_pushes == 1
    assert settlement.ml_fav_correct == 0
    assert settlement.ml_fav_total == 0
    assert settlement.upsets == []
    assert settlement.brier_score is None
    assert "excluded from record and Brier" in settlement.as_text()


def test_favorite_win_counts_and_enters_brier(conn) -> None:
    _slate(conn, home=31, away=17, prob=0.70)
    settlement = settle_slate(conn, "2026-11-07")
    assert settlement.ml_pushes == 0
    assert settlement.ml_fav_correct == 1
    assert settlement.ml_fav_total == 1
    assert settlement.upsets == []
    assert settlement.brier_score == (0.70 - 1) ** 2


def test_favorite_loss_is_an_upset(conn) -> None:
    _slate(conn, home=17, away=31, prob=0.70)
    settlement = settle_slate(conn, "2026-11-07")
    assert settlement.ml_pushes == 0
    assert settlement.ml_fav_correct == 0
    assert settlement.ml_fav_total == 1
    assert len(settlement.upsets) == 1
    assert settlement.brier_score == (0.70 - 0) ** 2


def test_settle_recent_exits_zero_when_nothing_is_completed(
    capsys, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("cfb_analytics.paths.database_path", lambda: tmp_path / "missing.sqlite3")
    assert cli._cmd_settle_recent(argparse_namespace()) == 0
    assert "nothing to settle" in capsys.readouterr().out

    path = tmp_path / "empty.sqlite3"
    with db.open_db(path) as empty:
        assert settle_completed_slates(
            empty, now=datetime(2026, 11, 8, 12, tzinfo=FOOTBALL_TZ)
        ) == []
    monkeypatch.setattr("cfb_analytics.paths.database_path", lambda: path)
    assert cli._cmd_settle_recent(argparse_namespace()) == 0


def argparse_namespace():
    import argparse

    return argparse.Namespace()
