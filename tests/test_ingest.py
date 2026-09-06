"""Ingestion behaviour, including partial-failure degradation.

Uses a fake client rather than the network: the point is the orchestration
contract, not the HTTP layer (covered in test_http.py).
"""

from __future__ import annotations

import pytest

from cfb_analytics.errors import SourceError
from cfb_analytics.ingest.outlier_ingest import ingest_slate


class FakeClient:
    def __init__(self, events, markets=None, injuries=None,
                 market_errors=(), injury_errors=()):
        self._events = events
        self._markets = markets or []
        self._injuries = injuries or []
        self._market_errors = set(market_errors)
        self._injury_errors = set(injury_errors)

    def fetch_schedule(self, league=None):
        return self._events

    def fetch_event_markets(self, event_id, market_type="GAMELINE"):
        if event_id in self._market_errors:
            raise SourceError(f"HTTP 500 for {event_id}")
        return self._markets

    def fetch_team_injuries(self, team_id, league=None):
        if team_id in self._injury_errors:
            raise SourceError(f"HTTP 500 for {team_id}")
        return self._injuries


@pytest.fixture(autouse=True)
def _canonical(canonical_slate):
    """Outlier resolves onto canonical rows, so every test needs them present.

    Autouse rather than per-test: an ingest run against an empty canonical
    store resolves nothing, which would make each assertion below fail for a
    reason unrelated to what it is testing.
    """
    return canonical_slate


@pytest.fixture
def two_events(schedule_event):
    second = {
        **schedule_event,
        "eventId": "evt-2",
        "home": {"teamId": "t-h2", "name": "Duke", "alias": "DUKE", "market": "Duke"},
        "away": {"teamId": "t-a2", "name": "Tulane", "alias": "TULN", "market": "Tulane"},
    }
    return [schedule_event, second]


SLATE = "2026-09-05"  # US Eastern slate date; see utils.football_date


class TestSlateFiltering:
    def test_selects_only_the_requested_slate(self, conn, two_events, schedule_event):
        off_slate = {**schedule_event, "eventId": "evt-x",
                     "scheduledTime": "2026-09-12T23:30:00+00:00"}
        client = FakeClient(two_events + [off_slate])
        summary = ingest_slate(conn, client, SLATE, with_odds=False, with_injuries=False)
        assert summary.events_seen == 2
        assert summary.games_matched == 2

    def test_empty_slate_is_not_an_error(self, conn, two_events):
        summary = ingest_slate(conn, FakeClient(two_events), "2030-01-01")
        assert summary.events_seen == 0
        assert summary.odds_rows == 0

    def test_keeps_late_night_west_coast_game_on_the_saturday_slate(
        self, conn, two_events, schedule_event
    ):
        """UCLA@CAL kicks 02:30Z Sunday but belongs to Saturday's slate."""
        late = {**schedule_event, "eventId": "evt-late",
                "scheduledTime": "2026-09-06T02:30:00+00:00", "dayOfWeek": 5}
        summary = ingest_slate(conn, FakeClient(two_events + [late]), SLATE,
                               with_odds=False, with_injuries=False)
        assert summary.events_seen == 3

    def test_excludes_friday_night_game_from_the_saturday_slate(
        self, conn, two_events, schedule_event
    ):
        """UTEP@OKLA kicks 00:00Z on the 5th but is a Friday-night game."""
        friday = {**schedule_event, "eventId": "evt-fri",
                  "scheduledTime": "2026-09-05T00:00:00+00:00", "dayOfWeek": 4}
        summary = ingest_slate(conn, FakeClient(two_events + [friday]), SLATE,
                               with_odds=False, with_injuries=False)
        assert summary.events_seen == 2

    def test_records_weekday_cross_check_mismatch(self, conn, schedule_event):
        """A feed dayOfWeek that disagrees with the Eastern weekday is surfaced."""
        wrong = {**schedule_event, "eventId": "evt-w", "dayOfWeek": 2}
        summary = ingest_slate(conn, FakeClient([wrong]), SLATE,
                               with_odds=False, with_injuries=False)
        assert len(summary.weekday_mismatches) == 1
        assert "weekday cross-check mismatches: 1" in summary.as_text()


class TestPartialFailure:
    def test_one_event_market_failure_does_not_lose_the_others(
        self, conn, two_events, moneyline_market
    ):
        client = FakeClient(two_events, markets=[moneyline_market],
                            market_errors={"evt-1"})
        summary = ingest_slate(conn, client, SLATE, with_injuries=False)

        assert summary.games_matched == 2, "both games still stored"
        assert len(summary.market_failures) == 1
        assert summary.odds_rows > 0, "the healthy event's odds were kept"

    def test_injury_failure_is_recorded_not_raised(self, conn, two_events):
        client = FakeClient(two_events, injury_errors={"t-home"})
        summary = ingest_slate(conn, client, SLATE, with_odds=False)
        assert len(summary.injury_failures) == 1
        assert summary.games_matched == 2

    def test_failures_land_in_source_health(self, conn, two_events, moneyline_market):
        client = FakeClient(two_events, markets=[moneyline_market], market_errors={"evt-1"})
        ingest_slate(conn, client, SLATE, with_injuries=False)
        failed = conn.execute("SELECT COUNT(*) AS n FROM source_health WHERE ok = 0").fetchone()
        assert failed["n"] == 1

    def test_malformed_event_is_counted_not_fatal(self, conn, two_events):
        client = FakeClient(two_events + [{"eventId": "bad", "scheduledTime": "nope"}])
        summary = ingest_slate(conn, client, SLATE, with_odds=False, with_injuries=False)
        assert len(summary.schema_failures) == 1
        assert summary.games_matched == 2


class TestIngestOutputs:
    def test_collects_the_book_set(self, conn, two_events, moneyline_market):
        client = FakeClient(two_events, markets=[moneyline_market])
        summary = ingest_slate(conn, client, SLATE, with_injuries=False)
        assert {"FANATICS", "BETRIVERS", "FLIFF"} <= summary.books

    def test_rerun_writes_no_duplicate_odds(self, conn, two_events, moneyline_market):
        client = FakeClient(two_events, markets=[moneyline_market])
        ingest_slate(conn, client, SLATE, with_injuries=False)
        before = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
        ingest_slate(conn, client, SLATE, with_injuries=False)
        after = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
        # A rerun in the same second is a genuine duplicate and is ignored.
        assert after == before

    def test_limit_caps_events(self, conn, two_events, moneyline_market):
        client = FakeClient(two_events, markets=[moneyline_market])
        summary = ingest_slate(conn, client, SLATE, limit=1, with_injuries=False)
        assert summary.games_matched == 1
        assert summary.events_seen == 2, "events_seen reports the slate, not the capped subset"

    def test_summary_text_mentions_failures(self, conn, two_events, moneyline_market):
        client = FakeClient(two_events, markets=[moneyline_market], market_errors={"evt-1"})
        text = ingest_slate(conn, client, SLATE, with_injuries=False).as_text()
        assert "market fetch failures: 1" in text


class TestCanonicalIdentity:
    """The defect this ingest was rewritten to prevent: one game, one row.

    Before this, ingesting a slate wrote a second `games` row under Outlier's
    `eventId` beside the CFBD row for the same fixture, so the board listed
    every side twice and each key held only half the books.
    """

    def test_ingest_writes_no_second_game_row(self, conn, two_events, moneyline_market):
        client = FakeClient(two_events, markets=[moneyline_market])
        summary = ingest_slate(conn, client, SLATE, with_injuries=False)

        rows = conn.execute("SELECT game_id, source FROM games ORDER BY game_id").fetchall()
        assert [r["game_id"] for r in rows] == ["cfbd:1001", "cfbd:1002"]
        assert {r["source"] for r in rows} == {"cfbd"}
        assert summary.games_matched == 2

    def test_odds_attach_to_the_canonical_game(self, conn, two_events, moneyline_market):
        client = FakeClient(two_events, markets=[moneyline_market])
        ingest_slate(conn, client, SLATE, with_injuries=False)
        keys = {r["game_id"] for r in conn.execute("SELECT DISTINCT game_id FROM odds_snapshots")}
        assert keys <= {"cfbd:1001", "cfbd:1002"}, "odds must not hang off a feed-native id"

    def test_no_feed_native_team_rows_are_created(self, conn, two_events):
        ingest_slate(conn, FakeClient(two_events), SLATE,
                     with_odds=False, with_injuries=False)
        stray = conn.execute(
            "SELECT COUNT(*) AS n FROM teams WHERE cfbd_id IS NULL").fetchone()["n"]
        assert stray == 0

    def test_unresolvable_event_is_named_not_materialised(self, conn, schedule_event):
        """An event with no canonical game is reported, and creates nothing.

        Inventing a row to hold the odds is exactly what produced the duplicate
        in the first place, so the data is refused and the reason names the
        teams and the fix.
        """
        stranger = {
            **schedule_event, "eventId": "evt-fcs",
            "home": {"teamId": "t-x", "name": "Bison", "alias": "NDSU",
                     "market": "North Dakota State"},
            "away": {"teamId": "t-y", "name": "Jackrabbits", "alias": "SDST",
                     "market": "South Dakota State"},
        }
        before = conn.execute("SELECT COUNT(*) AS n FROM games").fetchone()["n"]
        summary = ingest_slate(conn, FakeClient([stranger]), SLATE,
                               with_odds=False, with_injuries=False)

        assert summary.games_matched == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM games").fetchone()["n"] == before
        assert len(summary.unresolved_events) == 1
        assert "North Dakota State" in summary.unresolved_events[0]
        assert "unresolved events: 1" in summary.as_text()

    def test_feed_naming_is_kept_as_an_alias_on_the_canonical_team(
        self, conn, two_events
    ):
        ingest_slate(conn, FakeClient(two_events), SLATE,
                     with_odds=False, with_injuries=False)
        aliases = {
            (r["team_id"], r["alias"]) for r in conn.execute(
                "SELECT team_id, alias FROM team_aliases WHERE source = 'outlier'")
        }
        assert ("cfbd:202", "Tulsa") in aliases
        assert ("cfbd:202", "TLSA") in aliases
