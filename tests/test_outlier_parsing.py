from __future__ import annotations

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.sources.outlier import (
    parse_event,
    parse_injury_rows,
    parse_odds_rows,
)

CAPTURED = "2026-09-01T00:00:00+00:00"


class TestBookAttribution:
    """The live feed's `books` list is NOT parallel to its `odds` list.

    Zipping them by index mis-attributes every price to the wrong book and
    nothing downstream would catch it, so this is a regression test with teeth.
    """

    def test_price_is_attributed_to_the_book_named_inside_the_odds_entry(self, moneyline_market):
        rows = parse_odds_rows("evt-1", [moneyline_market], CAPTURED)
        by_book = {(r.book, r.side): r.price_american for r in rows}

        # books[0] is FLIFF but odds[0] belongs to FANATICS.
        assert by_book[("FANATICS", "AWAY")] == -260
        assert by_book[("BETRIVERS", "AWAY")] == -265
        assert by_book[("FLIFF", "AWAY")] == -250

    def test_index_zip_would_have_produced_a_different_answer(self, moneyline_market):
        """Guards the guard: if the fixture ever loses its ordering mismatch,
        the test above stops proving anything."""
        outcome = moneyline_market["outcomes"][0]
        zipped = {b: o["american"] for b, o in zip(outcome["books"], outcome["odds"], strict=False)}
        assert zipped["FLIFF"] == "-260"  # the wrong answer index-zipping gives
        rows = parse_odds_rows("evt-1", [moneyline_market], CAPTURED)
        actual = {r.book: r.price_american for r in rows if r.side == "AWAY"}
        assert actual["FLIFF"] == -250 != int(zipped["FLIFF"])


class TestParseOddsRows:
    def test_maps_proposition_to_market_code(self, moneyline_market):
        rows = parse_odds_rows("evt-1", [moneyline_market], CAPTURED)
        assert {r.market for r in rows} == {"ML"}

    def test_skips_out_of_scope_propositions(self):
        markets = [{"proposition": "WINNING_MARGIN", "outcomes": [
            {"position": "HOME", "odds": [{"american": "+300", "book": "FANDUEL"}]}]}]
        assert parse_odds_rows("evt-1", markets, CAPTURED) == []

    def test_unions_across_market_rows_of_the_same_proposition(self, moneyline_market):
        """A proposition spans several market rows, each with its own books."""
        second_row = {
            "marketId": "mkt-2",
            "proposition": "MONEYLINE",
            "outcomes": [{
                "position": "AWAY", "line": 0.0, "primary": True,
                "odds": [
                    {"american": "-255", "decimal": 1.392, "book": "PS3838"},
                    {"american": "-258", "decimal": 1.388, "book": "CIRCA"},
                ],
            }],
        }
        rows = parse_odds_rows("evt-1", [moneyline_market, second_row], CAPTURED)
        away_books = {r.book for r in rows if r.side == "AWAY"}
        assert {"FANATICS", "BETRIVERS", "FLIFF", "PS3838", "CIRCA"} == away_books

    def test_dedupes_repeated_book_side_line(self, moneyline_market):
        rows = parse_odds_rows("evt-1", [moneyline_market, moneyline_market], CAPTURED)
        keys = [(r.book, r.market, r.side, r.line) for r in rows]
        assert len(keys) == len(set(keys))

    def test_derives_decimal_when_feed_omits_it(self):
        markets = [{"proposition": "MONEYLINE", "outcomes": [
            {"position": "HOME", "line": 0.0, "odds": [{"american": "-110", "book": "BOVADA"}]}]}]
        row = parse_odds_rows("evt-1", markets, CAPTURED)[0]
        assert row.price_decimal == pytest.approx(1.909090, abs=1e-5)

    def test_snapshot_id_is_stable_and_distinguishes_books(self, moneyline_market):
        rows = parse_odds_rows("evt-1", [moneyline_market], CAPTURED)
        again = parse_odds_rows("evt-1", [moneyline_market], CAPTURED)
        assert [r.snapshot_id for r in rows] == [r.snapshot_id for r in again]
        assert len({r.snapshot_id for r in rows}) == len(rows)

    def test_ignores_odds_entry_with_no_book(self):
        markets = [{"proposition": "TOTAL", "outcomes": [
            {"position": "OVER", "line": 55.5, "odds": [{"american": "-110", "book": ""}]}]}]
        assert parse_odds_rows("evt-1", markets, CAPTURED) == []


class TestParseEvent:
    def test_normalises_kickoff_to_utc(self, schedule_event):
        assert parse_event(schedule_event)["kickoff_utc"] == "2026-09-05T23:30:00+00:00"

    def test_normalises_an_offset_kickoff_if_the_feed_ever_sends_one(self, schedule_event):
        """scheduledTime is UTC today, but injury.lastUpdated uses -0700, so the
        parser must keep handling offset forms."""
        schedule_event["scheduledTime"] = "2026-09-05T19:30:00-0700"
        assert parse_event(schedule_event)["kickoff_utc"] == "2026-09-06T02:30:00+00:00"

    def test_extracts_both_teams(self, schedule_event):
        record = parse_event(schedule_event)
        assert record["home"]["alias"] == "TLSA"
        assert record["away"]["team_id"] == "t-away"

    def test_raises_rather_than_defaulting_a_bad_kickoff(self, schedule_event):
        schedule_event["scheduledTime"] = "garbage"
        with pytest.raises(SchemaError, match="scheduledTime"):
            parse_event(schedule_event)

    def test_raises_when_teams_missing(self):
        with pytest.raises(SchemaError):
            parse_event({"eventId": "x", "scheduledTime": "2026-09-05T12:00:00Z"})


class TestParseInjuryRows:
    @pytest.fixture
    def players(self):
        return [
            {
                "playerId": "p1", "firstName": "A", "lastName": "B",
                "jerseyNumber": "7", "position": "QB",
                "injury": {
                    "status": "Questionable", "injury": "Knee",
                    "returnDate": "2026-09-05", "lastUpdated": "2026-08-29T11:02:11-0700",
                    "hasNews": True,
                },
            },
            {
                "playerId": "p2", "firstName": "C", "lastName": "D",
                "position": "WR",
                "injury": {"status": "Out", "injury": "Undisclosed", "hasNews": False},
            },
        ]

    def test_does_not_store_player_names(self, players):
        rows = parse_injury_rows("evt-1", "t-home", players, CAPTURED)
        for row in rows:
            assert "firstName" not in row and "lastName" not in row
            assert "A" not in row.values() and "B" not in row.values()

    def test_maps_designation_and_position_group(self, players):
        rows = parse_injury_rows("evt-1", "t-home", players, CAPTURED)
        assert rows[0]["designation"] == "Questionable"
        assert rows[0]["position_group"] == "QB"
        assert rows[1]["position_group"] == "SKILL"

    def test_normalises_last_updated_to_utc(self, players):
        rows = parse_injury_rows("evt-1", "t-home", players, CAPTURED)
        assert rows[0]["last_updated_utc"] == "2026-08-29T18:02:11+00:00"

    def test_skips_rows_without_a_player_id(self):
        rows = parse_injury_rows("evt-1", "t", [{"position": "QB", "injury": {}}], CAPTURED)
        assert rows == []

    def test_tolerates_missing_injury_object(self):
        rows = parse_injury_rows("evt-1", "t", [{"playerId": "p", "position": "RB"}], CAPTURED)
        assert rows[0]["designation"] is None
