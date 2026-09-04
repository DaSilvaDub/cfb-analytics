"""The scheduled job and the CFBD lines ingest."""

from __future__ import annotations

from datetime import datetime

import pytest

from cfb_analytics import daily
from cfb_analytics.errors import AuthRequiredError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_lines import (
    build_game_index,
    ingest_lines,
    parse_line_rows,
    parse_movement_rows,
)
from cfb_analytics.utils import FOOTBALL_TZ

STAMP = "2026-09-04T11:00:00+00:00"


def provider(**overrides):
    row = {
        "provider": "Bovada",
        "spread": -6.5, "spreadOpen": -3.0,
        "overUnder": 55.5, "overUnderOpen": 52.5,
        "homeMoneyline": -260, "awayMoneyline": 210,
    }
    row.update(overrides)
    return row


class TestParseLineRows:
    def test_moneyline_carries_prices(self):
        rows = parse_line_rows("g1", provider(), STAMP)
        ml = {r.side: r.price_american for r in rows if r.market == "ML"}
        assert ml == {"HOME": -260, "AWAY": 210}

    def test_spread_sides_get_opposite_signs(self):
        rows = parse_line_rows("g1", provider(), STAMP)
        spread = {r.side: r.line for r in rows if r.market == "SPREAD"}
        assert spread == {"HOME": -6.5, "AWAY": 6.5}

    def test_spread_and_total_have_no_price(self):
        """CFBD does not publish juice, so these are movement-only rows."""
        rows = parse_line_rows("g1", provider(), STAMP)
        for row in rows:
            if row.market in ("SPREAD", "TOTAL"):
                assert row.price_american is None

    def test_total_sides_share_the_line(self):
        rows = parse_line_rows("g1", provider(), STAMP)
        totals = {r.side: r.line for r in rows if r.market == "TOTAL"}
        assert totals == {"OVER": 55.5, "UNDER": 55.5}

    def test_missing_moneyline_omits_the_market(self):
        rows = parse_line_rows("g1", provider(homeMoneyline=None), STAMP)
        assert not [r for r in rows if r.market == "ML"]

    def test_provider_becomes_the_book_upper_cased(self):
        rows = parse_line_rows("g1", provider(), STAMP)
        assert {r.book for r in rows} == {"BOVADA"}

    def test_row_without_a_provider_is_dropped(self):
        assert parse_line_rows("g1", provider(provider=""), STAMP) == []


class TestParseMovementRows:
    def test_reports_spread_movement_from_a_single_call(self):
        """CFBD ships open and current together, so day one already has data."""
        rows = {(r["market"], r["side"]): r for r in parse_movement_rows("g1", provider(), STAMP)}
        home = rows[("SPREAD", "HOME")]
        assert home["open_line"] == -3.0
        assert home["current_line"] == -6.5
        assert home["move_magnitude"] == pytest.approx(-3.5)
        assert home["move_direction"] == "toward"

    def test_away_side_mirrors_the_home_move(self):
        rows = {(r["market"], r["side"]): r for r in parse_movement_rows("g1", provider(), STAMP)}
        away = rows[("SPREAD", "AWAY")]
        assert away["open_line"] == 3.0 and away["current_line"] == 6.5
        assert away["move_magnitude"] == pytest.approx(3.5)

    def test_total_movement(self):
        rows = {(r["market"], r["side"]): r for r in parse_movement_rows("g1", provider(), STAMP)}
        assert rows[("TOTAL", "OVER")]["move_magnitude"] == pytest.approx(3.0)

    def test_rlm_stays_line_only(self):
        for row in parse_movement_rows("g1", provider(), STAMP):
            assert row["rlm_basis"] == "line_only"
            assert row["rlm_flag"] == 0

    def test_missing_opening_number_yields_no_movement_rather_than_a_guess(self):
        rows = parse_movement_rows("g1", provider(spreadOpen=None, overUnderOpen=None), STAMP)
        assert rows == []

    def test_unmoved_line_is_flat(self):
        rows = parse_movement_rows("g1", provider(spreadOpen=-6.5), STAMP)
        home = next(r for r in rows if r["market"] == "SPREAD" and r["side"] == "HOME")
        assert home["move_direction"] == "flat"


class FakeLinesClient:
    def __init__(self, games):
        self.games = games

    def fetch_lines(self, year, *, week=None, season_type="regular"):
        return self.games


@pytest.fixture
def seeded(conn):
    store.upsert_team(conn, {"team_id": "h", "school": "Tulsa", "alias": "TLSA", "market": "Tulsa"})
    store.upsert_team(
        conn, {"team_id": "a", "school": "Oklahoma State", "alias": "OKST", "market": "Okla St"})
    store.upsert_game(conn, {
        "game_id": "g1", "season": 2026, "kickoff_utc": "2026-09-05T23:30:00+00:00",
        "football_date": "2026-09-05", "day_of_week": 5,
        "home_team_id": "h", "away_team_id": "a",
        "venue_name": None, "network": None, "status": "pregame",
    })
    return conn


def cfbd_game(**overrides):
    row = {
        "id": 1, "season": 2026, "week": 2, "startDate": "2026-09-05T23:30:00.000Z",
        "homeTeam": "Tulsa", "awayTeam": "Oklahoma State",
        "lines": [provider()],
    }
    row.update(overrides)
    return row


class TestIngestLines:
    def test_matches_a_game_by_school_name(self, seeded):
        summary = ingest_lines(seeded, FakeLinesClient([cfbd_game()]), 2026, captured_utc=STAMP)
        assert summary.games_matched == 1
        assert summary.odds_rows > 0
        assert summary.movement_rows > 0

    def test_matches_through_an_alias(self, seeded):
        game = cfbd_game(homeTeam="TLSA", awayTeam="OKST")
        assert ingest_lines(
            seeded, FakeLinesClient([game]), 2026, captured_utc=STAMP).games_matched == 1

    def test_unmatched_game_is_counted_not_guessed(self, seeded):
        game = cfbd_game(homeTeam="Nowhere State", awayTeam="Elsewhere Tech")
        summary = ingest_lines(seeded, FakeLinesClient([game]), 2026, captured_utc=STAMP)
        assert summary.games_matched == 0
        assert summary.games_unmatched == 1
        assert summary.odds_rows == 0

    def test_wrong_date_does_not_match(self, seeded):
        """The key includes the slate, so a same-teams game in another week
        cannot be joined onto this one."""
        game = cfbd_game(startDate="2026-10-10T23:30:00.000Z")
        assert ingest_lines(
            seeded, FakeLinesClient([game]), 2026, captured_utc=STAMP).games_unmatched == 1

    def test_rerun_is_idempotent(self, seeded):
        client = FakeLinesClient([cfbd_game()])
        ingest_lines(seeded, client, 2026, captured_utc=STAMP)
        before = seeded.execute("SELECT COUNT(*) n FROM odds_snapshots").fetchone()["n"]
        ingest_lines(seeded, client, 2026, captured_utc=STAMP)
        after = seeded.execute("SELECT COUNT(*) n FROM odds_snapshots").fetchone()["n"]
        assert after == before

    def test_movement_lands_in_the_table(self, seeded):
        ingest_lines(seeded, FakeLinesClient([cfbd_game()]), 2026, captured_utc=STAMP)
        row = seeded.execute(
            """SELECT move_magnitude, rlm_basis FROM line_movement
               WHERE market='SPREAD' AND side='HOME'""").fetchone()
        assert row["move_magnitude"] == pytest.approx(-3.5)
        assert row["rlm_basis"] == "line_only"

    def test_game_index_covers_every_name_form(self, seeded):
        index = build_game_index(seeded)
        assert ("2026-09-05", "TULSA", "OKLAHOMA STATE") in index
        assert ("2026-09-05", "TLSA", "OKST") in index


class TestSlateWindow:
    def test_selects_only_slates_inside_the_operating_window(self, seeded):
        store.upsert_game(seeded, {
            "game_id": "far", "season": 2026, "kickoff_utc": "2026-11-01T23:30:00+00:00",
            "football_date": "2026-11-01", "day_of_week": 6,
            "home_team_id": "h", "away_team_id": "a",
            "venue_name": None, "network": None, "status": "pregame"})
        now = datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ)
        assert daily.slates_in_window(seeded, now=now) == ["2026-09-05"]

    def test_past_slates_are_excluded(self, seeded):
        now = datetime(2026, 9, 20, 12, tzinfo=FOOTBALL_TZ)
        assert daily.slates_in_window(seeded, now=now) == []


class TestDailyDegradation:
    def test_missing_cfbd_key_is_a_skip_not_a_failure(self, seeded, monkeypatch):
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
        report = daily.run_daily(
            seeded, with_outlier=False, now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ))
        cfbd = next(o for o in report.outcomes if o.name == "cfbd")
        assert cfbd.status == "skipped"
        assert "CFBD_API_KEY" in cfbd.detail

    def test_an_expired_outlier_session_is_a_skip_with_the_reason(
        self, seeded, monkeypatch
    ):
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)

        def boom(*a, **k):
            raise AuthRequiredError("HTTP 403")

        monkeypatch.setattr("cfb_analytics.sources.outlier.OutlierClient.__init__", boom)
        report = daily.run_daily(
            seeded, with_outlier=True, now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ))
        outlier = next(o for o in report.outcomes if o.name == "outlier")
        assert outlier.status == "skipped"
        assert "24h" in outlier.detail

    def test_all_sources_skipped_still_reports_ok(self, seeded, monkeypatch):
        """A known-missing credential must not turn into a red build daily."""
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
        report = daily.run_daily(
            seeded, with_outlier=False, now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ))
        assert report.ok is True

    def test_summary_names_every_source_and_its_state(self, seeded, monkeypatch):
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
        text = daily.run_daily(
            seeded, with_outlier=False,
            now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ)).as_text()
        assert "cfbd" in text
        assert "SKIP" in text
        assert "UNPROMOTED" in text, "shadow mode must be stamped on scheduled output"

    def test_market_is_rebuilt_for_slates_in_window(self, seeded, monkeypatch):
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
        ingest_lines(seeded, FakeLinesClient([cfbd_game()]), 2026, captured_utc=STAMP)
        report = daily.run_daily(
            seeded, with_outlier=False, now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ))
        assert report.games == 1


class TestProvenance:
    """Every odds row must record which feed produced it.

    `insert_odds` previously hardcoded 'outlier', so CFBD rows were written
    with false provenance -- destroying the only column that distinguishes a
    juice-free CFBD spread from a priced Outlier one, and corrupting the book
    attribution in `coverage`.
    """

    def test_cfbd_rows_are_labelled_cfbd(self, seeded):
        ingest_lines(seeded, FakeLinesClient([cfbd_game()]), 2026, captured_utc=STAMP)
        sources = {r["source"] for r in seeded.execute(
            "SELECT DISTINCT source FROM odds_snapshots")}
        assert sources == {"cfbd"}

    def test_outlier_rows_keep_the_outlier_label(self, seeded, moneyline_market):
        from cfb_analytics.sources.outlier import parse_odds_rows

        store.insert_odds(seeded, parse_odds_rows("g1", [moneyline_market], STAMP))
        sources = {r["source"] for r in seeded.execute(
            "SELECT DISTINCT source FROM odds_snapshots")}
        assert sources == {"outlier"}

    def test_same_line_from_two_feeds_does_not_collide(self, seeded, moneyline_market):
        """Provenance is part of the snapshot identity, so two feeds quoting
        the same book and price both survive rather than one overwriting the
        other."""
        from cfb_analytics.sources.outlier import parse_odds_rows

        outlier_rows = parse_odds_rows("g1", [moneyline_market], STAMP)
        store.insert_odds(seeded, outlier_rows)
        cloned = [type(r)(**{**r.__dict__, "source": "cfbd"}) for r in outlier_rows]
        added = store.insert_odds(seeded, cloned)
        assert added == len(cloned)


class FakeBackfillClient:
    """Enough of CFBDClient for the bootstrap path."""

    def __init__(self, games):
        self._games = games

    def fetch_venues(self):
        return []

    def fetch_fbs_teams(self, year):
        return [
            {"id": 1, "school": "Tulsa", "abbreviation": "TLSA", "conference": "AAC",
             "classification": "fbs"},
            {"id": 2, "school": "Oklahoma State", "abbreviation": "OKST", "conference": "Big 12",
             "classification": "fbs"},
        ]

    def fetch_games(self, year, *, season_type="both", classification="fbs"):
        return self._games


class TestBootstrap:
    """A fresh cloud runner has no data branch, so the store starts empty."""

    def test_reports_skip_when_empty_and_no_key(self, conn, monkeypatch):
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
        report = daily.run_daily(
            conn, season=2026, with_outlier=False,
            now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ))
        boot = next(o for o in report.outcomes if o.name == "bootstrap")
        assert boot.status == "skipped"
        assert "no 2026 games" in boot.detail

    def test_does_not_run_when_the_season_already_has_games(self, seeded, monkeypatch):
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
        report = daily.run_daily(
            seeded, season=2026, with_outlier=False,
            now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ))
        assert not [o for o in report.outcomes if o.name == "bootstrap"]
        assert report.bootstrapped is False

    def test_can_be_disabled(self, conn, monkeypatch):
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
        report = daily.run_daily(
            conn, season=2026, with_outlier=False, bootstrap=False,
            now=datetime(2026, 9, 4, 12, tzinfo=FOOTBALL_TZ))
        assert not [o for o in report.outcomes if o.name == "bootstrap"]

    def test_counts_games_for_the_season(self, seeded):
        assert daily.games_for_season(seeded, 2026) == 1
        assert daily.games_for_season(seeded, 2025) == 0

    def test_empty_store_with_a_key_loads_the_season(self, conn, monkeypatch):
        """The self-healing path: no data branch on the first cloud run."""
        games = [{
            "id": 900, "season": 2026, "week": 2, "seasonType": "regular",
            "startDate": "2026-09-05T23:30:00.000Z", "neutralSite": False,
            "conferenceGame": False, "completed": False,
            "homeTeam": "Tulsa", "homeId": 1, "awayTeam": "Oklahoma State", "awayId": 2,
            "venue": "Chapman Stadium",
        }]
        monkeypatch.setattr(
            "cfb_analytics.sources.cfbd.CFBDClient",
            lambda *a, **k: FakeBackfillClient(games))
        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: True)
        # Keep the lines leg out of it; this test is about the bootstrap alone.
        monkeypatch.setattr(
            "cfb_analytics.ingest.cfbd_lines.ingest_lines",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
            raising=False)

        report = daily.DailyReport(started_utc="x")
        daily._bootstrap_if_empty(conn, report, 2026)

        boot = next(o for o in report.outcomes if o.name == "bootstrap")
        assert boot.status == "ok", boot.detail
        assert report.bootstrapped is True
        assert daily.games_for_season(conn, 2026) == 1

    def test_bootstrap_failure_is_reported_not_raised(self, conn, monkeypatch):
        from cfb_analytics.errors import SourceError

        monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: True)

        def boom(*a, **k):
            raise SourceError("CFBD is down")

        monkeypatch.setattr("cfb_analytics.sources.cfbd.CFBDClient", boom)
        report = daily.DailyReport(started_utc="x")
        daily._bootstrap_if_empty(conn, report, 2026)
        boot = next(o for o in report.outcomes if o.name == "bootstrap")
        assert boot.status == "failed"
        assert "CFBD is down" in boot.detail
