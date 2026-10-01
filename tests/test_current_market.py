"""Current lines do not mix historical captures, alternates, or periods."""

from dataclasses import replace

import pytest

from cfb_analytics.features.build_market import build_market_for_slate
from cfb_analytics.features.current_market import load_current_market_rows
from cfb_analytics.ingest import store
from cfb_analytics.sources.outlier import parse_odds_rows
from tests.test_build_market import AFTER, BEFORE, EARLIER, SLATE, odds
from tests.test_build_market import seeded as seeded


def test_current_primary_line_and_both_sides_are_selected(seeded):
    quotes = []
    for book in ("A", "B", "C"):
        for side in ("OVER", "UNDER"):
            quotes += [
                odds(book, side, -110, EARLIER, market="TOTAL", line=55.5),
                odds(book, side, -110, AFTER, market="TOTAL", line=75.5),
                replace(
                    odds(book, side, -110, BEFORE, market="TOTAL", line=18.5), is_primary=False
                ),
            ]
    for side in ("OVER", "UNDER"):
        quotes.append(odds("A", side, -110, BEFORE, market="TOTAL", line=60.5))
    store.insert_odds(seeded, quotes)
    rows = load_current_market_rows(seeded, SLATE, as_of_utc=AFTER)
    assert len(rows) == 2
    assert {r["line"] for r in rows} == {60.5}
    assert {r["side"] for r in rows} == {"OVER", "UNDER"}
    assert {r["as_of_utc"] for r in rows} == {BEFORE}


def test_market_builder_does_not_republish_old_line_as_new(seeded):
    store.insert_odds(
        seeded,
        [
            odds("A", "OVER", -110, EARLIER, market="TOTAL", line=55.5),
            odds("A", "UNDER", -110, EARLIER, market="TOTAL", line=55.5),
        ],
    )
    build_market_for_slate(seeded, SLATE)
    store.insert_odds(
        seeded,
        [
            odds("A", "OVER", -110, BEFORE, market="TOTAL", line=60.5),
            odds("A", "UNDER", -110, BEFORE, market="TOTAL", line=60.5),
        ],
    )
    build_market_for_slate(seeded, SLATE)
    records = seeded.execute("SELECT line, as_of_utc FROM market_consensus").fetchall()
    assert {(r["line"], r["as_of_utc"]) for r in records} == {
        (55.5, EARLIER),
        (60.5, BEFORE),
    }


def test_book_quotes_are_not_mixed_across_captures(seeded):
    store.insert_odds(
        seeded,
        [
            odds("A", "HOME", -200, EARLIER),
            odds("A", "AWAY", 170, EARLIER),
            odds("B", "HOME", -120, BEFORE),
            odds("B", "AWAY", 100, BEFORE),
        ],
    )
    rows = load_current_market_rows(seeded, SLATE, as_of_utc=BEFORE)
    assert all(r["n_books"] == 1 and r["best_book"] == "B" for r in rows)


@pytest.mark.parametrize(
    "period",
    [
        {"periods": [1]},
        {"periodLabel": "1st Half"},
        {"marketGroupId": "FIRST_HALF"},
    ],
)
def test_gameline_parser_excludes_partial_game_periods(moneyline_market, period):
    assert parse_odds_rows("g1", [{**moneyline_market, **period}], BEFORE) == []
    assert parse_odds_rows("g1", [moneyline_market], BEFORE)
