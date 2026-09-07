from __future__ import annotations

from cfb_analytics.features.team_props import build_team_prop_consensus
from cfb_analytics.ingest import store
from cfb_analytics.sources.outlier import parse_team_prop_rows

CAPTURED = "2026-09-05T12:00:00+00:00"


def _market(
    proposition: str = "POINTS",
    *,
    team_id: str = "feed-home",
    periods=None,
    market_type: str = "TEAM_PROP",
):
    return {
        "marketId": "tp-1",
        "marketType": market_type,
        "marketGroupId": "GAME" if periods is None else "QUARTERS",
        "proposition": proposition,
        "teamId": team_id,
        "periods": periods,
        "outcomes": [
            {
                "position": side,
                "line": 27.5,
                "primary": True,
                "odds": [
                    {"book": book, "american": price}
                    for book, price in (
                        ("A", "-110"),
                        ("B", "-108"),
                        ("C", "+100"),
                    )
                ],
            }
            for side in ("OVER", "UNDER")
        ],
    }


def test_parser_requires_authoritative_team_prop_and_canonical_team():
    assert (
        parse_team_prop_rows(
            "g", [_market(market_type="PLAYER_PROP")], CAPTURED, {"feed-home": "team"}
        )
        == []
    )
    assert parse_team_prop_rows("g", [_market()], CAPTURED, {}) == []


def test_parser_whitelists_full_game_families_and_deduplicates_cards():
    markets = [
        _market(),
        _market(),
        _market("PASSING_YARDS"),
        _market(periods=[{"period": 1}]),
    ]
    rows = parse_team_prop_rows("g", markets, CAPTURED, {"feed-home": "canonical-home"})
    assert len(rows) == 6
    assert {row.market for row in rows} == {"team_total_points"}
    assert {row.team_id for row in rows} == {"canonical-home"}
    assert len({row.snapshot_id for row in rows}) == 6


def test_consensus_requires_three_books_and_both_sides(conn, canonical_slate):
    rows = parse_team_prop_rows(
        "cfbd:1001",
        [_market(team_id="feed-home")],
        CAPTURED,
        {"feed-home": canonical_slate["tulsa"]},
    )
    assert store.insert_team_prop_odds(conn, rows) == 6
    written = build_team_prop_consensus(conn, ["cfbd:1001"], as_of_utc="2026-09-05T13:00:00+00:00")
    assert written == 2
    stored = conn.execute(
        "SELECT team_id, market, side, n_books FROM team_prop_consensus ORDER BY side"
    ).fetchall()
    assert [(r["side"], r["n_books"]) for r in stored] == [("OVER", 3), ("UNDER", 3)]
    assert {r["team_id"] for r in stored} == {canonical_slate["tulsa"]}


def test_consensus_fails_closed_below_three_books(conn, canonical_slate):
    rows = parse_team_prop_rows(
        "cfbd:1001",
        [_market()],
        CAPTURED,
        {"feed-home": canonical_slate["tulsa"]},
    )
    store.insert_team_prop_odds(conn, [row for row in rows if row.book != "C"])
    assert (
        build_team_prop_consensus(conn, ["cfbd:1001"], as_of_utc="2026-09-05T13:00:00+00:00") == 0
    )
