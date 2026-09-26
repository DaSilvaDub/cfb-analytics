"""Acceptance tests A1–A5 for historical CFBD lines dual-stamp ingest."""

from __future__ import annotations

import pytest

from cfb_analytics.errors import LeakageError
from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.features.build_market import build_market_for_slate
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_lines import ingest_lines
from cfb_analytics.ingest.cfbd_lines_historical import (
    SOURCE,
    backfill_historical_lines,
    ingest_historical_lines,
    parse_close_rows,
    parse_open_rows,
    rebuild_market_for_season,
    synthetic_close_utc,
    synthetic_open_utc,
)
from cfb_analytics.sources.outlier import OddsRow

KICKOFF = "2024-09-07T19:30:00+00:00"
STAMP_LIVE = "2026-09-04T11:00:00+00:00"


def provider(**overrides):
    row = {
        "provider": "Bovada",
        "spread": -6.5,
        "spreadOpen": -3.0,
        "overUnder": 55.5,
        "overUnderOpen": 52.5,
        "homeMoneyline": -260,
        "awayMoneyline": 210,
    }
    row.update(overrides)
    return row


class FakeLinesClient:
    def __init__(self, games):
        self.games = games
        self.calls: list[tuple] = []

    def fetch_lines(self, year, *, week=None, season_type="regular"):
        self.calls.append((year, week, season_type))
        return self.games


def cfbd_game(**overrides):
    row = {
        "id": 1,
        "season": 2024,
        "week": 2,
        "startDate": "2024-09-07T19:30:00.000Z",
        "homeTeam": "Tulsa",
        "awayTeam": "Oklahoma State",
        "lines": [provider()],
    }
    row.update(overrides)
    return row


@pytest.fixture
def seeded(conn):
    store.upsert_team(
        conn, {"team_id": "h", "school": "Tulsa", "alias": "TLSA", "market": "Tulsa",
               "classification": "fbs"}
    )
    store.upsert_team(
        conn,
        {
            "team_id": "a",
            "school": "Oklahoma State",
            "alias": "OKST",
            "market": "Okla St",
            "classification": "fbs",
        },
    )
    store.upsert_game(
        conn,
        {
            "game_id": "g2024",
            "season": 2024,
            "kickoff_utc": KICKOFF,
            "football_date": "2024-09-07",
            "day_of_week": 5,
            "home_team_id": "h",
            "away_team_id": "a",
            "venue_name": None,
            "network": None,
            "status": "final",
        },
    )
    return conn


class TestSyntheticStamps:
    def test_open_is_seven_days_before_kickoff(self):
        assert synthetic_open_utc(KICKOFF) == "2024-08-31T19:30:00+00:00"

    def test_close_is_sixty_seconds_before_kickoff(self):
        assert synthetic_close_utc(KICKOFF) == "2024-09-07T19:29:00+00:00"

    def test_both_stamps_strictly_before_kickoff(self):
        open_s = synthetic_open_utc(KICKOFF)
        close_s = synthetic_close_utc(KICKOFF)
        assert open_s < KICKOFF
        assert close_s < KICKOFF
        assert open_s < close_s

    def test_unparseable_kickoff_is_leakage(self):
        with pytest.raises(LeakageError):
            synthetic_close_utc("not-a-timestamp")


class TestParseRoles:
    def test_open_has_no_moneyline(self):
        rows = parse_open_rows("g1", provider(), "2024-08-31T19:30:00+00:00")
        assert all(r.market != "ML" for r in rows)
        assert {r.market for r in rows} == {"SPREAD", "TOTAL"}
        assert {r.source for r in rows} == {SOURCE}

    def test_open_uses_open_fields(self):
        rows = parse_open_rows("g1", provider(), "2024-08-31T19:30:00+00:00")
        spread = {r.side: r.line for r in rows if r.market == "SPREAD"}
        assert spread == {"HOME": -3.0, "AWAY": 3.0}

    def test_close_includes_moneyline_and_close_fields(self):
        rows = parse_close_rows("g1", provider(), "2024-09-07T19:29:00+00:00")
        ml = {r.side: r.price_american for r in rows if r.market == "ML"}
        assert ml == {"HOME": -260, "AWAY": 210}
        spread = {r.side: r.line for r in rows if r.market == "SPREAD"}
        assert spread == {"HOME": -6.5, "AWAY": 6.5}
        assert {r.source for r in rows} == {SOURCE}

    def test_close_omits_ml_when_one_side_missing(self):
        rows = parse_close_rows(
            "g1", provider(homeMoneyline=None), "2024-09-07T19:29:00+00:00"
        )
        assert not [r for r in rows if r.market == "ML"]


class TestA1AsOfGuard:
    def test_admissible_keeps_open_and_close_before_kickoff(self):
        reader = AsOfReader(game_id="g2024", kickoff_utc=KICKOFF, season=2024)
        rows = [
            {"captured_utc": synthetic_open_utc(KICKOFF), "role": "open"},
            {"captured_utc": synthetic_close_utc(KICKOFF), "role": "close"},
            {"captured_utc": KICKOFF, "role": "at_kick"},
            {"captured_utc": "2024-09-07T20:00:00+00:00", "role": "after"},
        ]
        kept = reader.admissible(rows, what="odds", as_of_field="captured_utc")
        assert {r["role"] for r in kept} == {"open", "close"}

    def test_check_raises_leakage_for_stamp_at_or_after_kickoff(self):
        reader = AsOfReader(game_id="g2024", kickoff_utc=KICKOFF, season=2024)
        with pytest.raises(LeakageError, match="at or after kickoff"):
            reader.check(KICKOFF, what="close")
        with pytest.raises(LeakageError, match="at or after kickoff"):
            reader.check("2024-09-07T19:30:01+00:00", what="close")

    def test_fixture_row_stamped_at_kickoff_dropped_from_consensus_path(self, seeded):
        """A1: close stamp >= kickoff must not reach market consensus."""
        # Manually insert a leaking close row and a valid open row.
        store.insert_odds(
            seeded,
            [
                OddsRow(
                    game_id="g2024",
                    market_id=None,
                    book="BOVADA",
                    market="ML",
                    side="HOME",
                    line=0.0,
                    price_american=-200,
                    price_decimal=1.5,
                    is_primary=True,
                    captured_utc=KICKOFF,  # leak
                    source=SOURCE,
                ),
                OddsRow(
                    game_id="g2024",
                    market_id=None,
                    book="BOVADA",
                    market="ML",
                    side="AWAY",
                    line=0.0,
                    price_american=170,
                    price_decimal=2.7,
                    is_primary=True,
                    captured_utc=KICKOFF,
                    source=SOURCE,
                ),
            ],
        )
        summary = build_market_for_slate(seeded, "2024-09-07", min_books_for_consensus=1)
        # Leaked rows dropped; no admissible priced sides → no consensus.
        assert summary.leaked_rows_dropped >= 2
        assert summary.consensus_rows == 0


class TestA2StampInvariant:
    def test_every_historical_row_captured_before_kickoff(self, seeded):
        ingest_historical_lines(seeded, FakeLinesClient([cfbd_game()]), 2024, week=2)
        rows = seeded.execute(
            """SELECT o.captured_utc, g.kickoff_utc
               FROM odds_snapshots o
               JOIN games g ON g.game_id = o.game_id
               WHERE o.source = ?""",
            (SOURCE,),
        ).fetchall()
        assert rows
        for row in rows:
            assert row["captured_utc"] < row["kickoff_utc"]


class TestA3SeasonIsolation:
    def test_historical_ingest_does_not_mutate_live_cfbd_rows(self, seeded):
        # Seed a live-path row for a 2026 game on the same teams.
        store.upsert_game(
            seeded,
            {
                "game_id": "g2026",
                "season": 2026,
                "kickoff_utc": "2026-09-05T23:30:00+00:00",
                "football_date": "2026-09-05",
                "day_of_week": 5,
                "home_team_id": "h",
                "away_team_id": "a",
                "venue_name": None,
                "network": None,
                "status": "pregame",
            },
        )
        live_game = {
            "id": 99,
            "season": 2026,
            "week": 2,
            "startDate": "2026-09-05T23:30:00.000Z",
            "homeTeam": "Tulsa",
            "awayTeam": "Oklahoma State",
            "lines": [provider()],
        }
        ingest_lines(
            seeded, FakeLinesClient([live_game]), 2026, captured_utc=STAMP_LIVE
        )
        live_before = seeded.execute(
            "SELECT snapshot_id, source, captured_utc, market, side, price_american "
            "FROM odds_snapshots WHERE source = 'cfbd' ORDER BY snapshot_id"
        ).fetchall()
        assert live_before

        ingest_historical_lines(seeded, FakeLinesClient([cfbd_game()]), 2024, week=2)

        live_after = seeded.execute(
            "SELECT snapshot_id, source, captured_utc, market, side, price_american "
            "FROM odds_snapshots WHERE source = 'cfbd' ORDER BY snapshot_id"
        ).fetchall()
        assert [tuple(r) for r in live_after] == [tuple(r) for r in live_before]

        hist = seeded.execute(
            "SELECT COUNT(*) n FROM odds_snapshots WHERE source = ?", (SOURCE,)
        ).fetchone()["n"]
        assert hist > 0


class TestIngestAndIdempotency:
    def test_dual_stamp_writes_open_and_close(self, seeded):
        summary = ingest_historical_lines(
            seeded, FakeLinesClient([cfbd_game()]), 2024, week=2
        )
        assert summary.games_matched == 1
        assert summary.open_odds_rows > 0
        assert summary.close_odds_rows > 0
        assert summary.games_with_ml_close == 1
        stamps = {
            r["captured_utc"]
            for r in seeded.execute(
                "SELECT DISTINCT captured_utc FROM odds_snapshots WHERE source = ?",
                (SOURCE,),
            )
        }
        assert stamps == {synthetic_open_utc(KICKOFF), synthetic_close_utc(KICKOFF)}

    def test_rerun_is_idempotent(self, seeded):
        client = FakeLinesClient([cfbd_game()])
        ingest_historical_lines(seeded, client, 2024, week=2)
        before = seeded.execute(
            "SELECT COUNT(*) n FROM odds_snapshots WHERE source = ?", (SOURCE,)
        ).fetchone()["n"]
        ingest_historical_lines(seeded, client, 2024, week=2)
        after = seeded.execute(
            "SELECT COUNT(*) n FROM odds_snapshots WHERE source = ?", (SOURCE,)
        ).fetchone()["n"]
        assert after == before

    def test_backfill_loops_weeks(self, seeded):
        client = FakeLinesClient([cfbd_game()])
        backfill_historical_lines(seeded, client, 2024, weeks=range(1, 4))
        assert [c[1] for c in client.calls] == [1, 2, 3]


class TestA5ConsensusRebuild:
    def test_build_market_for_slate_produces_pre_kickoff_consensus(self, seeded):
        ingest_historical_lines(seeded, FakeLinesClient([cfbd_game()]), 2024, week=2)
        summary = build_market_for_slate(
            seeded, "2024-09-07", min_books_for_consensus=1
        )
        assert summary.consensus_rows > 0
        rows = seeded.execute(
            """SELECT c.as_of_utc, g.kickoff_utc
               FROM market_consensus c
               JOIN games g ON g.game_id = c.game_id
               WHERE g.football_date = '2024-09-07'"""
        ).fetchall()
        assert rows
        for row in rows:
            assert row["as_of_utc"] < row["kickoff_utc"]

    def test_rebuild_market_for_season_covers_slate(self, seeded):
        ingest_historical_lines(seeded, FakeLinesClient([cfbd_game()]), 2024, week=2)
        totals = rebuild_market_for_season(
            seeded, 2024, min_books_for_consensus=1
        )
        assert totals["slates"] == 1
        assert totals["consensus_rows"] > 0


class TestA6MoneylineWiring:
    def test_report_states_market_na_when_no_coverage(self, conn):
        from cfb_analytics.backtest.moneyline import run_moneyline_backtest

        # Minimal season seed reused from moneyline tests pattern.
        for team_id, school in [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]:
            store.upsert_team(
                conn,
                {
                    "team_id": f"{team_id}2019",
                    "school": school,
                    "alias": None,
                    "market": None,
                    "classification": "fbs",
                },
            )
        gid = 0
        for week in range(1, 5):
            for i, home in enumerate("abcd"):
                for away in "abcd"[i + 1 :]:
                    gid += 1
                    store.upsert_cfbd_game(
                        conn,
                        {
                            "game_id": f"g2019-{gid}",
                            "season": 2019,
                            "week": week,
                            "season_type": "regular",
                            "kickoff_utc": f"2019-09-{week:02d}T00:00:00+00:00",
                            "football_date": f"2019-09-{week:02d}",
                            "neutral_site": 0,
                            "conference_game": 0,
                            "home_team_id": f"{home}2019",
                            "away_team_id": f"{away}2019",
                            "venue_name": None,
                            "venue_id": None,
                            "status": "final",
                            "home_points": 24 + gid % 10,
                            "away_points": 14 + gid % 7,
                            "completed": 1,
                            "source": "cfbd",
                        },
                    )
        text = run_moneyline_backtest(conn, seasons=(2019,), min_games=1).as_text()
        assert "market: N/A (coverage)" in text

    def test_report_scores_market_when_close_consensus_exists(self, seeded):
        from cfb_analytics.backtest.harness import GamePrediction
        from cfb_analytics.backtest.moneyline import (
            _load_market_home_probs,
            _market_probs,
        )

        ingest_historical_lines(seeded, FakeLinesClient([cfbd_game()]), 2024, week=2)
        build_market_for_slate(seeded, "2024-09-07", min_books_for_consensus=1)
        home = _load_market_home_probs(seeded, ["g2024"], min_books=1)
        assert "g2024" in home
        preds = [
            GamePrediction(
                game_id="g2024",
                season=2024,
                week=2,
                home_team_id="h",
                away_team_id="a",
                neutral_site=False,
                predicted_margin=7.0,
                actual_margin=10.0,
                elo_home_rating=None,
                elo_away_rating=None,
                internal_elo_win_prob=None,
                logit_win_prob=None,
            )
        ]
        probs, skipped = _market_probs(preds, home)
        assert skipped == 0
        assert len(probs) == 1
