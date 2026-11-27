"""Team-prop settlement uses the last pre-kickoff capture and keeps pushes.

Game markets already keep the newest capture strictly before kickoff (#32).
Team props did not: there was no kickoff filter and no ORDER BY, so the row
sqlite happened to return last was the grade. A team-total push stays in the
push count. An OVER-board push stays in the push count and out of the tier
and family hit rates.
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.backtest.settle import settle_slate
from cfb_analytics.features.over_confidence import persist_over_board, settle_over_board
from cfb_analytics.ingest import store
from tests.conftest import seed_canonical_game, seed_canonical_team

SLATE = "2026-10-03"
KICKOFF = "2026-10-03T19:00:00+00:00"
OPEN = "2026-10-03T12:00:00+00:00"
CLOSE = "2026-10-03T18:30:00+00:00"
AFTER = "2026-10-03T21:00:00+00:00"


def _game(conn: sqlite3.Connection, *, home_points: int, away_points: int) -> str:
    home = seed_canonical_team(conn, 1, "Home U", "HOM")
    away = seed_canonical_team(conn, 2, "Away U", "AWY")
    seed_canonical_game(conn, "g-tp", home, away, football_date=SLATE, kickoff=KICKOFF)
    conn.execute(
        """UPDATE games SET home_points = ?, away_points = ?, completed = 1, status = 'completed'
           WHERE game_id = 'g-tp'""",
        (home_points, away_points),
    )
    return home


def _quote(
    conn: sqlite3.Connection,
    *,
    line: float,
    as_of: str,
    over_prob: float,
    n_books: int,
) -> None:
    under_prob = round(1.0 - over_prob, 2)
    for side, prob in (("OVER", over_prob), ("UNDER", under_prob)):
        conn.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books, prob_shin)
               VALUES ('g-tp', 'cfbd:1', 'team_total_points', ?, ?, ?, ?, ?)""",
            (line, side, as_of, n_books, prob),
        )


def test_team_prop_grades_the_last_pre_kickoff_capture(conn: sqlite3.Connection) -> None:
    """Home scored 30. The close is OVER 34.5 (a miss). The wide open and the
    post-kickoff quote are both OVER 20.5 and would have cashed.
    """
    _game(conn, home_points=30, away_points=10)
    _quote(conn, line=34.5, as_of=CLOSE, over_prob=0.62, n_books=3)
    _quote(conn, line=20.5, as_of=AFTER, over_prob=0.80, n_books=5)
    _quote(conn, line=20.5, as_of=OPEN, over_prob=0.70, n_books=8)
    conn.commit()

    settlement = settle_slate(conn, SLATE)

    assert settlement.team_props_fav_hits == 0
    assert settlement.team_props_evaluated == 1
    assert settlement.team_props_pushes == 0


def test_team_prop_push_stays_in_the_record(conn: sqlite3.Connection) -> None:
    """Home scored 30 on a 30-point line. That is a push, not an ungraded skip."""
    _game(conn, home_points=30, away_points=10)
    _quote(conn, line=30.0, as_of=CLOSE, over_prob=0.55, n_books=4)
    conn.commit()

    settlement = settle_slate(conn, SLATE)

    assert settlement.team_props_fav_hits == 0
    assert settlement.team_props_evaluated == 0
    assert settlement.team_props_pushes == 1
    assert "Pushes: 1" in settlement.as_text()


def test_over_board_push_stays_out_of_tier_and_family(conn: sqlite3.Connection) -> None:
    """An exact landing is a push. It must not enter the tier or family denominator."""
    home = seed_canonical_team(conn, 1, "Home U", "HOM")
    away = seed_canonical_team(conn, 2, "Away U", "AWY")
    seed_canonical_game(
        conn, "g-box", home, away, football_date=SLATE, kickoff=KICKOFF
    )
    persist_over_board(
        conn,
        SLATE,
        [
            {
                "rank": 1,
                "game_id": "g-box",
                "game_label": "Away U at Home U",
                "kickoff_et": "3:00 p.m.",
                "kickoff_utc": KICKOFF,
                "market": "team_rushing_yards",
                "pick": "HOM rush",
                "team_id": home,
                "side": "OVER",
                "line": 180.0,
                "projected": 200.0,
                "over_prob": 0.72,
                "confidence": 7.5,
                "tier": "HIGH",
                "family": "favorite_team_rush",
                "inclusion_reasons": ("ap_top_25",),
                "flags": (),
            }
        ],
        as_of_utc=CLOSE,
    )
    store.insert_team_game_boxes(
        conn,
        [
            {
                "game_id": "g-box",
                "team_id": home,
                "rushing_yards": 180.0,
                "net_passing_yards": 200.0,
                "total_yards": 380.0,
                "points": 30,
            }
        ],
    )
    conn.commit()

    report = settle_over_board(conn, SLATE)

    assert report.push == 1
    assert report.cash == 0
    assert report.miss == 0
    assert report.evaluated == 1
    assert report.by_tier["HIGH"] == (0, 0)
    assert "favorite_team_rush" not in (report.by_family or {})
    text = report.as_text()
    assert "push 1" in text
    assert "HIGH" not in text
    assert "favorite_team_rush" not in text
