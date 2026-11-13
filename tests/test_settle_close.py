"""Settlement grades the last pre-kickoff capture, not the widest one.

Opening moneyline: 8 books, home Shin 70%. Closing capture: 3 books, home Shin
40%. Home wins 21-17. The wide open makes home the favorite (1/1, Brier 0.0900);
the close makes the away side the favorite, so the grade is a loss. A capture
at or after kickoff is not a close.
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.backtest.settle import settle_slate
from cfb_analytics.ingest import store

KICKOFF = "2026-10-03T19:00:00+00:00"
OPEN = "2026-10-03T12:00:00+00:00"
CLOSE = "2026-10-03T18:30:00+00:00"
AT_KICK = "2026-10-03T19:00:00+00:00"
AFTER = "2026-10-03T21:00:00+00:00"


def _consensus(
    conn: sqlite3.Connection,
    *,
    market: str,
    side: str,
    line: float,
    as_of: str,
    price: int,
    prob: float,
    n_books: int,
) -> None:
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price, prob_shin, n_books)
           VALUES ('g-close', ?, ?, ?, ?, 'all_books', ?, ?, ?)""",
        (market, side, line, as_of, price, prob, n_books),
    )


def _slate(conn: sqlite3.Connection) -> None:
    now = "2026-10-01T00:00:00+00:00"
    for tid, school, alias in (("cfbd:1", "Home St", "HOM"), ("cfbd:2", "Away St", "AWY")):
        conn.execute(
            """INSERT INTO teams (team_id, school, alias, first_seen_utc, last_seen_utc)
               VALUES (?, ?, ?, ?, ?)""",
            (tid, school, alias, now, now),
        )
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "g-close",
            "season": 2026,
            "football_date": "2026-10-03",
            "kickoff_utc": KICKOFF,
            "home_team_id": "cfbd:1",
            "away_team_id": "cfbd:2",
            "home_points": 21,
            "away_points": 17,
            "completed": 1,
            "status": "completed",
        },
    )
    # Open is the widest capture and makes home a 70% favorite.
    _consensus(
        conn, market="ML", side="HOME", line=0.0, as_of=OPEN, price=-230, prob=0.70, n_books=8
    )
    _consensus(
        conn, market="ML", side="AWAY", line=0.0, as_of=OPEN, price=190, prob=0.30, n_books=8
    )
    # Close has fewer books and flips the favorite to the away side.
    _consensus(
        conn, market="ML", side="HOME", line=0.0, as_of=CLOSE, price=150, prob=0.40, n_books=3
    )
    _consensus(
        conn, market="ML", side="AWAY", line=0.0, as_of=CLOSE, price=-150, prob=0.60, n_books=3
    )
    # Same width trap on the total: 21+17=38. Open 34.5 is an over; close 45.5 is an under.
    _consensus(
        conn, market="TOTAL", side="OVER", line=34.5, as_of=OPEN, price=-110, prob=0.70, n_books=8
    )
    _consensus(
        conn, market="TOTAL", side="UNDER", line=34.5, as_of=OPEN, price=-110, prob=0.30, n_books=8
    )
    _consensus(
        conn, market="TOTAL", side="OVER", line=45.5, as_of=CLOSE, price=-110, prob=0.40, n_books=3
    )
    _consensus(
        conn,
        market="TOTAL",
        side="UNDER",
        line=45.5,
        as_of=CLOSE,
        price=-110,
        prob=0.60,
        n_books=3,
    )
    # At kickoff and after kickoff would both grade the home favorite and the over.
    for stamp, books, home_prob in ((AT_KICK, 4, 0.95), (AFTER, 5, 0.90)):
        away_prob = round(1.0 - home_prob, 2)
        _consensus(
            conn,
            market="ML",
            side="HOME",
            line=0.0,
            as_of=stamp,
            price=-500,
            prob=home_prob,
            n_books=books,
        )
        _consensus(
            conn,
            market="ML",
            side="AWAY",
            line=0.0,
            as_of=stamp,
            price=400,
            prob=away_prob,
            n_books=books,
        )
        _consensus(
            conn,
            market="TOTAL",
            side="OVER",
            line=32.5,
            as_of=stamp,
            price=-110,
            prob=home_prob,
            n_books=books,
        )
        _consensus(
            conn,
            market="TOTAL",
            side="UNDER",
            line=32.5,
            as_of=stamp,
            price=-110,
            prob=away_prob,
            n_books=books,
        )
    conn.commit()


def test_moneyline_grades_the_closing_capture_not_the_widest(conn: sqlite3.Connection) -> None:
    _slate(conn)
    settlement = settle_slate(conn, "2026-10-03")

    assert settlement.ml_fav_correct == 0
    assert settlement.ml_fav_total == 1
    assert settlement.ml_fav_win_pct == 0.0
    assert len(settlement.upsets) == 1
    upset = settlement.upsets[0]
    assert upset.favorite == "AWY"
    assert upset.underdog == "HOM"
    assert upset.fav_prob == 0.60
    assert upset.fav_price == -150
    assert upset.fav_score == 17
    assert upset.dog_score == 21
    assert settlement.brier_score == (0.40 - 1) ** 2
    text = settlement.as_text()
    assert "Overall Favorite Record: 0/1" in text
    assert "Brier Score : 0.3600" in text
    assert "70.0%" not in text


def test_total_grades_the_closing_line_not_the_widest(conn: sqlite3.Connection) -> None:
    _slate(conn)
    settlement = settle_slate(conn, "2026-10-03")

    assert settlement.totals_overs == 0
    assert settlement.totals_unders == 1
    assert settlement.totals_pushes == 0
