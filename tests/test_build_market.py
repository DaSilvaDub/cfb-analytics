"""The market build pipeline, including its leakage wiring."""

from __future__ import annotations

import pytest

from cfb_analytics.features.build_market import build_market_for_slate
from cfb_analytics.ingest import store
from cfb_analytics.sources.outlier import OddsRow

KICKOFF = "2026-09-05T23:30:00+00:00"
SLATE = "2026-09-05"
BEFORE = "2026-09-05T20:00:00+00:00"
EARLIER = "2026-09-04T20:00:00+00:00"
AFTER = "2026-09-06T04:00:00+00:00"  # a closing line, recorded post-kickoff


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


def odds(book, side, price, captured, market="ML", line=0.0):
    return OddsRow(game_id="g1", market_id="m", book=book, market=market, side=side,
                   line=line, price_american=price, price_decimal=None,
                   is_primary=True, captured_utc=captured)


class TestLeakageWiring:
    def test_post_kickoff_prices_are_dropped_not_used(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "HOME", -200, BEFORE), odds("A", "AWAY", 170, BEFORE),
            odds("B", "HOME", -210, BEFORE), odds("B", "AWAY", 175, BEFORE),
            # A closing line recorded after kickoff must never reach a feature.
            odds("A", "HOME", -400, AFTER), odds("A", "AWAY", 320, AFTER),
        ])
        summary = build_market_for_slate(seeded, SLATE)
        assert summary.leaked_rows_dropped == 2

        row = seeded.execute(
            "SELECT consensus_price FROM market_consensus WHERE side='HOME'").fetchone()
        assert row["consensus_price"] > -300, "the -400 closing line leaked into consensus"

    def test_a_game_with_only_post_kickoff_odds_produces_nothing(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "HOME", -200, AFTER), odds("A", "AWAY", 170, AFTER)])
        summary = build_market_for_slate(seeded, SLATE)
        assert summary.consensus_rows == 0
        assert summary.leaked_rows_dropped == 2


class TestConsensusPersistence:
    def test_writes_one_row_per_side(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "HOME", -200, BEFORE), odds("A", "AWAY", 170, BEFORE),
            odds("B", "HOME", -210, BEFORE), odds("B", "AWAY", 175, BEFORE),
        ])
        summary = build_market_for_slate(seeded, SLATE)
        assert summary.consensus_rows == 2
        sides = {r["side"] for r in seeded.execute("SELECT side FROM market_consensus")}
        assert sides == {"HOME", "AWAY"}

    def test_stores_all_three_devig_methods(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "HOME", -200, BEFORE), odds("A", "AWAY", 170, BEFORE),
            odds("B", "HOME", -210, BEFORE), odds("B", "AWAY", 175, BEFORE),
        ])
        build_market_for_slate(seeded, SLATE)
        row = seeded.execute(
            """SELECT prob_multiplicative, prob_shin, prob_power, prob_spread
               FROM market_consensus WHERE side='HOME'""").fetchone()
        assert all(row[k] is not None for k in
                   ("prob_multiplicative", "prob_shin", "prob_power", "prob_spread"))

    def test_rebuild_replaces_rather_than_duplicates(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "HOME", -200, BEFORE), odds("A", "AWAY", 170, BEFORE),
            odds("B", "HOME", -210, BEFORE), odds("B", "AWAY", 175, BEFORE),
        ])
        build_market_for_slate(seeded, SLATE)
        build_market_for_slate(seeded, SLATE)
        count = seeded.execute("SELECT COUNT(*) n FROM market_consensus").fetchone()["n"]
        assert count == 2

    def test_separate_lines_are_separate_rows(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "OVER", -110, BEFORE, market="TOTAL", line=55.5),
            odds("A", "UNDER", -110, BEFORE, market="TOTAL", line=55.5),
            odds("A", "OVER", -130, BEFORE, market="TOTAL", line=52.5),
            odds("A", "UNDER", 110, BEFORE, market="TOTAL", line=52.5),
        ])
        build_market_for_slate(seeded, SLATE)
        lines = {r["line"] for r in seeded.execute(
            "SELECT DISTINCT line FROM market_consensus WHERE market='TOTAL'")}
        assert lines == {55.5, 52.5}


class TestLineMovement:
    def test_records_movement_between_captures(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "HOME", -150, EARLIER, market="SPREAD", line=-3.0),
            odds("A", "AWAY", 130, EARLIER, market="SPREAD", line=3.0),
            odds("A", "HOME", -150, BEFORE, market="SPREAD", line=-6.5),
            odds("A", "AWAY", 130, BEFORE, market="SPREAD", line=6.5),
        ])
        build_market_for_slate(seeded, SLATE)
        rows = seeded.execute(
            "SELECT side, move_magnitude, rlm_basis FROM line_movement").fetchall()
        assert rows, "expected movement rows"
        assert all(r["rlm_basis"] == "line_only" for r in rows)

    def test_a_single_capture_reports_no_movement(self, seeded):
        """Emitting a zero would look like a measured flat line."""
        store.insert_odds(seeded, [
            odds("A", "HOME", -200, BEFORE), odds("A", "AWAY", 170, BEFORE)])
        summary = build_market_for_slate(seeded, SLATE)
        assert summary.movement_rows == 0


class TestSummary:
    def test_counts_unpriced_groups(self, seeded):
        store.insert_odds(seeded, [odds("A", "HOME", -200, BEFORE)])
        summary = build_market_for_slate(seeded, SLATE)
        assert summary.unpriced_groups == 1
        assert summary.consensus_rows == 0

    def test_reports_the_anchor_used(self, seeded):
        store.insert_odds(seeded, [
            odds("A", "HOME", -200, BEFORE), odds("A", "AWAY", 170, BEFORE),
            odds("B", "HOME", -210, BEFORE), odds("B", "AWAY", 175, BEFORE),
        ])
        summary = build_market_for_slate(seeded, SLATE)
        assert summary.anchors.get("all_books") == 1, "no sharp books in this fixture"

    def test_empty_slate_is_reported_not_crashed(self, conn):
        summary = build_market_for_slate(conn, "2030-01-01")
        assert summary.games == 0
        assert "games            : 0" in summary.as_text()
