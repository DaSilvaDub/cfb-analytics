"""Spread and total consensus stop at the last capture strictly before kickoff.

On master, ``LATEST_CAPTURE_SQL`` took ``MAX(as_of_utc)`` with no kickoff
bound, so a close stamped at or after kickoff replaced the pre-game number.
"""

from __future__ import annotations

from cfb_analytics.features.current_market import current_primary_market
from cfb_analytics.ingest import store

SLATE = "2026-09-05"
KICKOFF = "2026-09-05T23:30:00+00:00"
PRE = "2026-09-05T16:00:00+00:00"
AFTER = "2026-09-06T01:00:00+00:00"


def _game(
    conn, *, kickoff: str = KICKOFF, tbd: int = 0, game_id: str = "g1", slate: str = SLATE
) -> None:
    store.upsert_team(conn, {"team_id": "h", "school": "Home U", "alias": "HOME", "market": "H"})
    store.upsert_team(conn, {"team_id": "a", "school": "Away U", "alias": "AWAY", "market": "A"})
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": game_id,
            "season": 2026,
            "week": 1,
            "season_type": "regular",
            "kickoff_utc": kickoff,
            "kickoff_tbd": tbd,
            "football_date": slate,
            "home_team_id": "h",
            "away_team_id": "a",
            "completed": 0,
            "status": "scheduled",
            "source": "cfbd",
        },
    )


def _spread(conn, line: float, as_of: str, *, game_id: str = "g1") -> None:
    for side, signed in (("HOME", line), ("AWAY", -line)):
        conn.execute(
            """INSERT INTO market_consensus
               (game_id, market, line, side, as_of_utc, n_books, consensus_price,
                best_price, best_book, hold, anchor, prob_shin, prob_multiplicative)
               VALUES (?, 'SPREAD', ?, ?, ?, 8, -110, -110, 'DK', 0.04, 'all_books', 0.5, 0.5)""",
            (game_id, signed, side, as_of),
        )


def test_at_and_after_kickoff_captures_are_ignored(conn) -> None:
    _game(conn)
    _spread(conn, -3.5, PRE)
    _spread(conn, -7.0, KICKOFF)
    _spread(conn, -10.0, AFTER)
    rows = current_primary_market(conn, SLATE, ("SPREAD",))
    assert {row["line"] for row in rows} == {-3.5, 3.5}
    assert {row["as_of_utc"] for row in rows} == {PRE}


def test_a_capture_exactly_at_kickoff_is_not_pre_kickoff(conn) -> None:
    _game(conn)
    _spread(conn, -7.0, KICKOFF)
    assert current_primary_market(conn, SLATE, ("SPREAD",)) == []


def test_tbd_placeholder_does_not_drop_the_game_or_cut_off_the_line(conn) -> None:
    """Midnight-ET placeholder. A 2pm capture must still be the current line."""
    placeholder = "2026-11-07T05:00:00+00:00"
    afternoon = "2026-11-07T19:00:00+00:00"
    _game(conn, kickoff=placeholder, tbd=1, slate="2026-11-07")
    _spread(conn, -3.0, "2026-11-06T22:00:00+00:00")
    _spread(conn, -6.5, afternoon)
    rows = current_primary_market(conn, "2026-11-07", ("SPREAD",))
    assert rows, "a TBD game must stay on the slate"
    assert {row["line"] for row in rows} == {-6.5, 6.5}
    assert {row["as_of_utc"] for row in rows} == {afternoon}
