"""A team-points bet is graded on its own line. An exact integer landing is a push."""

from __future__ import annotations

import sqlite3

from cfb_analytics.backtest.settle import settle_slate
from cfb_analytics.ingest import store

SLATE = "2026-10-17"
AS_OF = "2026-10-17T12:00:00+00:00"


def _team(conn: sqlite3.Connection, team_id: str, school: str, alias: str) -> None:
    now = "2026-10-01T00:00:00+00:00"
    conn.execute(
        """INSERT INTO teams (team_id, school, alias, first_seen_utc, last_seen_utc)
           VALUES (?, ?, ?, ?, ?)""",
        (team_id, school, alias, now, now),
    )


def _game(conn: sqlite3.Connection, *, home_points: int, away_points: int) -> None:
    _team(conn, "cfbd:1", "Home U", "HOME")
    _team(conn, "cfbd:2", "Away U", "AWAY")
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "g-tp",
            "season": 2026,
            "football_date": SLATE,
            "kickoff_utc": "2026-10-17T20:00:00+00:00",
            "home_team_id": "cfbd:1",
            "away_team_id": "cfbd:2",
            "home_points": home_points,
            "away_points": away_points,
            "completed": 1,
            "status": "completed",
        },
    )


def _quote(
    conn: sqlite3.Connection,
    *,
    team_id: str,
    line: float,
    n_books: int,
    over_prob: float,
    over_price: int,
) -> None:
    under_prob = round(1.0 - over_prob, 2)
    under_price = 100 if over_price < 0 else -120
    for side, prob, price in (
        ("OVER", over_prob, over_price),
        ("UNDER", under_prob, under_price),
    ):
        conn.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, prob_shin)
               VALUES ('g-tp', ?, 'team_total_points', ?, ?, ?, ?, ?, ?)""",
            (team_id, line, side, AS_OF, n_books, price, prob),
        )


def test_team_points_bet_settles_only_its_own_line(conn: sqlite3.Connection) -> None:
    """Home scored 21. The bet is OVER 24.5 and loses. The 17.5 alternate would have won."""
    _game(conn, home_points=21, away_points=10)
    _quote(conn, team_id="cfbd:1", line=24.5, n_books=5, over_prob=0.62, over_price=-140)
    _quote(conn, team_id="cfbd:1", line=17.5, n_books=1, over_prob=0.70, over_price=-220)
    conn.commit()

    settlement = settle_slate(conn, SLATE)

    assert settlement.team_props_fav_hits == 0
    assert settlement.team_props_evaluated == 1
    assert settlement.team_props_pushes == 0
    assert settlement.team_props_units == -1.0


def test_integer_team_points_line_is_a_push_refund(conn: sqlite3.Connection) -> None:
    """Away scored 24 on a 24-point line. That is a refund, not an ungraded skip.

    The 17.5 alternate would have cashed. It is not the bet.
    """
    _game(conn, home_points=10, away_points=24)
    _quote(conn, team_id="cfbd:2", line=24.0, n_books=5, over_prob=0.53, over_price=-115)
    _quote(conn, team_id="cfbd:2", line=17.5, n_books=1, over_prob=0.80, over_price=-300)
    conn.commit()

    settlement = settle_slate(conn, SLATE)

    assert settlement.team_props_fav_hits == 0
    assert settlement.team_props_evaluated == 0
    assert settlement.team_props_pushes == 1
    assert settlement.team_props_units == 0.0
    text = settlement.as_text()
    assert "Pushes: 1" in text
    assert "0 units" in text
