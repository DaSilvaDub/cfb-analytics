"""Current-market selection, the full-game gameline filter and the slate report."""

from __future__ import annotations

import json

import pytest

from cfb_analytics.features.build_market import build_market_for_slate
from cfb_analytics.features.current_market import (
    current_consensus,
    current_primary_market,
    primary_lines,
)
from cfb_analytics.ingest import store
from cfb_analytics.reporting.slate import (
    build_slate_report,
    render_slate_report,
    slate_report_json,
)
from cfb_analytics.sources.outlier import OddsRow, is_full_game_market, parse_odds_rows

SLATE = "2026-09-05"
KICKOFF = "2026-09-05T23:30:00+00:00"
EARLY = "2026-09-05T13:00:00+00:00"
LATE = "2026-09-05T18:00:00+00:00"


@pytest.fixture
def seeded(conn):
    store.upsert_team(conn, {"team_id": "h", "school": "Home U", "alias": "HOME", "market": "H"})
    store.upsert_team(conn, {"team_id": "a", "school": "Away U", "alias": "AWAY", "market": "A"})
    store.upsert_game(conn, {
        "game_id": "g1", "season": 2026, "kickoff_utc": KICKOFF,
        "football_date": SLATE, "day_of_week": 5,
        "home_team_id": "h", "away_team_id": "a",
        "venue_name": None, "network": None, "status": "pregame",
    })
    return conn


def odds(book, side, price, captured, market="TOTAL", line=55.5):
    return OddsRow(game_id="g1", market_id="m", book=book, market=market, side=side,
                   line=line, price_american=price, price_decimal=None,
                   is_primary=True, captured_utc=captured)


def _consensus(conn, market, line, side, as_of, *, prob=0.5, n_books=5, price=-110):
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES ('g1', ?, ?, ?, ?, ?, ?, ?, 'DK', 0.04, 'all_books', ?, ?, ?, 0.01, '[]')""",
        (market, line, side, as_of, n_books, price, price, prob, prob, prob),
    )


def _period_market(proposition, group, label):
    return {
        "marketId": f"{proposition}-{label}",
        "proposition": proposition,
        "marketGroupId": group,
        "periodLabel": label,
        "periods": [label],
        "outcomes": [
            {"position": "OVER", "line": 13.5,
             "odds": [{"book": "DK", "american": -110, "decimal": 1.91}]},
            {"position": "UNDER", "line": 13.5,
             "odds": [{"book": "DK", "american": -110, "decimal": 1.91}]},
        ],
    }


class TestFullGameGamelines:
    def test_period_markets_are_not_full_game(self):
        assert not is_full_game_market(_period_market("TOTAL", "QUARTERS", "1Q"))
        assert not is_full_game_market(_period_market("SPREAD", "HALVES", "1H"))
        assert is_full_game_market({"proposition": "TOTAL", "marketGroupId": "GAMELINES"})
        assert is_full_game_market({"proposition": "TOTAL"})

    def test_quarter_total_never_parses_as_game_total(self, moneyline_market):
        rows = parse_odds_rows(
            "g1", [moneyline_market, _period_market("TOTAL", "QUARTERS", "1Q")], EARLY
        )
        assert rows, "the full-game moneyline must still parse"
        assert all(r.market == "ML" for r in rows)


class TestSupersededLineGroups:
    def test_pulled_line_is_not_restamped_as_current(self, seeded):
        store.insert_odds(seeded, [
            # 9am ladder: 55.5 and an alternate 58.5
            odds("A", "OVER", -110, EARLY), odds("A", "UNDER", -110, EARLY),
            odds("A", "OVER", 120, EARLY, line=58.5), odds("A", "UNDER", -140, EARLY, line=58.5),
            # Later capture: the line moved to 56.5 and the 58.5 alternate was pulled.
            odds("A", "OVER", -110, LATE, line=56.5), odds("A", "UNDER", -110, LATE, line=56.5),
        ])
        summary = build_market_for_slate(seeded, SLATE)
        assert summary.stale_rows_skipped == 4
        lines = {r["line"] for r in seeded.execute(
            "SELECT line FROM market_consensus WHERE market='TOTAL'")}
        assert lines == {56.5}

    def test_sources_capture_independently(self, seeded):
        cfbd = OddsRow(game_id="g1", market_id=None, book="CONSENSUS", market="TOTAL",
                       side="OVER", line=57.0, price_american=None, price_decimal=None,
                       is_primary=True, captured_utc=LATE, source="cfbd")
        store.insert_odds(seeded, [
            odds("A", "OVER", -110, EARLY), odds("A", "UNDER", -110, EARLY), cfbd,
        ])
        build_market_for_slate(seeded, SLATE)
        lines = {r["line"] for r in seeded.execute(
            "SELECT line FROM market_consensus WHERE market='TOTAL'")}
        assert 55.5 in lines, "a later CFBD pull must not erase the priced Outlier capture"


class TestCurrentConsensus:
    def test_only_newest_capture_per_game_market(self, seeded):
        _consensus(seeded, "TOTAL", 58.5, "OVER", EARLY)
        _consensus(seeded, "TOTAL", 58.5, "UNDER", EARLY)
        _consensus(seeded, "TOTAL", 56.5, "OVER", LATE)
        _consensus(seeded, "TOTAL", 56.5, "UNDER", LATE)
        _consensus(seeded, "ML", 0.0, "HOME", EARLY, prob=0.6)
        _consensus(seeded, "ML", 0.0, "AWAY", EARLY, prob=0.4)
        rows = current_consensus(seeded, SLATE)
        assert {(r["market"], r["line"]) for r in rows} == {("TOTAL", 56.5), ("ML", 0.0)}

    def test_primary_line_is_most_booked_then_balanced(self):
        def row(line, side, n_books, prob):
            return {"game_id": "g1", "market": "SPREAD", "line": line, "side": side,
                    "n_books": n_books, "prob_shin": prob, "hold": 0.04}

        rows = [
            row(-7.0, "HOME", 12, 0.50), row(7.0, "AWAY", 12, 0.50),
            row(-3.5, "HOME", 4, 0.62), row(3.5, "AWAY", 4, 0.38),
            row(-10.5, "HOME", 12, 0.40), row(10.5, "AWAY", 12, 0.60),
        ]
        picked = primary_lines(rows)
        assert {r["line"] for r in picked} == {-7.0, 7.0}

    def test_board_and_report_agree_on_current(self, seeded):
        _consensus(seeded, "SPREAD", -3.0, "HOME", EARLY)
        _consensus(seeded, "SPREAD", 3.0, "AWAY", EARLY)
        _consensus(seeded, "SPREAD", -6.5, "HOME", LATE)
        _consensus(seeded, "SPREAD", 6.5, "AWAY", LATE)
        primary = current_primary_market(seeded, SLATE, ("SPREAD",))
        assert {r["line"] for r in primary} == {-6.5, 6.5}


class TestSlateReport:
    def test_every_family_prints_even_when_empty(self, seeded):
        text = render_slate_report(build_slate_report(seeded, SLATE))
        for heading in ("MONEYLINES", "SPREADS", "GAME TOTALS", "RUSHING / RECEIVING OVERS"):
            assert f"== {heading}" in text
        assert "(no current moneyline market)" in text
        assert "(no current spread market)" in text
        assert "(no current game-total market)" in text

    def test_game_lines_appear_once_per_game(self, seeded):
        for as_of in (EARLY, LATE):
            _consensus(seeded, "ML", 0.0, "HOME", as_of, prob=0.6, price=-150)
            _consensus(seeded, "ML", 0.0, "AWAY", as_of, prob=0.4, price=130)
        _consensus(seeded, "SPREAD", -4.5, "HOME", LATE)
        _consensus(seeded, "SPREAD", 4.5, "AWAY", LATE)
        _consensus(seeded, "TOTAL", 55.5, "OVER", LATE)
        _consensus(seeded, "TOTAL", 55.5, "UNDER", LATE)
        _consensus(seeded, "TOTAL", 49.5, "OVER", LATE, n_books=2, prob=0.7)
        _consensus(seeded, "TOTAL", 49.5, "UNDER", LATE, n_books=2, prob=0.3)

        report = build_slate_report(seeded, SLATE)
        counts = report.counts()
        assert (counts["ML"], counts["SPREAD"], counts["TOTAL"]) == (1, 1, 1)
        assert report.rows["TOTAL"][0].line == 55.5
        assert report.rows["SPREAD"][0].line == -4.5

        payload = slate_report_json(report)
        json.dumps(payload)
        assert payload["moneylines"][0]["prices"] == {"HOME": -150, "AWAY": 130}
        assert payload["stamp"]
