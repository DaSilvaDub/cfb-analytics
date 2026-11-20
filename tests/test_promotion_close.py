"""Promotion close is the last ML quote at or before kickoff.

A later post-kickoff quote must not become the close, even when it is the
newest row in market_consensus.
"""

from __future__ import annotations

from cfb_analytics.backtest.moneyline import _load_market_home_probs
from tests.conftest import seed_canonical_game, seed_canonical_team

KICKOFF = "2026-10-03T19:00:00+00:00"
GAME_ID = "cfbd:close"


def _home_ml(conn, as_of: str, prob: float, n_books: int) -> None:
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, side, line, as_of_utc, anchor, consensus_price,
            prob_shin, n_books)
           VALUES (?, 'ML', 'HOME', 0.0, ?, 'all_books', -110, ?, ?)""",
        (GAME_ID, as_of, prob, n_books),
    )


def test_promotion_close_is_the_last_quote_at_or_before_kickoff(conn) -> None:
    home = seed_canonical_team(conn, 1, "Home St", "HOM")
    away = seed_canonical_team(conn, 2, "Away St", "AWY")
    seed_canonical_game(
        conn,
        GAME_ID,
        home,
        away,
        football_date="2026-10-03",
        kickoff=KICKOFF,
    )
    _home_ml(conn, "2026-10-03T12:00:00+00:00", 0.70, 8)
    _home_ml(conn, "2026-10-03T18:30:00+00:00", 0.40, 3)
    _home_ml(conn, KICKOFF, 0.55, 4)
    _home_ml(conn, "2026-10-03T21:00:00+00:00", 0.90, 9)
    conn.commit()

    home_probs = _load_market_home_probs(conn, [GAME_ID], min_books=1)

    assert home_probs[GAME_ID] == 0.55


def test_promotion_close_omits_a_game_whose_only_quote_is_after_kickoff(conn) -> None:
    home = seed_canonical_team(conn, 1, "Home St", "HOM")
    away = seed_canonical_team(conn, 2, "Away St", "AWY")
    seed_canonical_game(
        conn,
        GAME_ID,
        home,
        away,
        football_date="2026-10-03",
        kickoff=KICKOFF,
    )
    _home_ml(conn, "2026-10-03T21:00:00+00:00", 0.90, 9)
    conn.commit()

    assert _load_market_home_probs(conn, [GAME_ID], min_books=1) == {}
