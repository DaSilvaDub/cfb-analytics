"""TBD kickoffs are not real start times, and the daily job refreshes them."""

from __future__ import annotations

from datetime import datetime, timedelta

from cfb_analytics import daily
from cfb_analytics.features.over_confidence import OverConfidencePick, persist_over_board
from cfb_analytics.ingest.cfbd_fundamentals import _cutoffs
from cfb_analytics.ingest.store import upsert_cfbd_game
from cfb_analytics.sources.cfbd import parse_game
from cfb_analytics.utils import FOOTBALL_TZ
from tests.conftest import seed_canonical_team

PLACEHOLDER = "2026-11-07T05:00:00+00:00"  # midnight ET
TWO_PM_ET = "2026-11-07T19:00:00+00:00"
REAL_KICKOFF = "2026-11-07T20:30:00+00:00"  # 3:30 PM ET


def _pick(game_id: str, kickoff: str) -> OverConfidencePick:
    return OverConfidencePick(
        rank=1,
        game_id=game_id,
        game_label="LIB at NMSU",
        kickoff_et="midnight",
        kickoff_utc=kickoff,
        market="team_rushing_yards",
        pick="LIB rush",
        team_id="cfbd:2335",
        side="OVER",
        line=155.5,
        projected=180.0,
        over_prob=0.7,
        confidence=7.0,
        tier="HIGH",
        family="favorite_team_rush",
        inclusion_reasons=("conference_top_2",),
        flags=(),
    )


def _teams(conn) -> None:
    seed_canonical_team(conn, 2335, "Liberty", "LIB")
    seed_canonical_team(conn, 166, "New Mexico State", "NMSU")


def _raw(*, tbd: bool, start: str, completed: bool = False) -> dict:
    return {
        "id": 401870795,
        "season": 2026,
        "week": 10,
        "seasonType": "regular",
        "startDate": start,
        "startTimeTBD": tbd,
        "completed": completed,
        "neutralSite": False,
        "conferenceGame": True,
        "homeId": 166,
        "homeTeam": "New Mexico State",
        "awayId": 2335,
        "awayTeam": "Liberty",
    }


def test_parse_game_flags_start_time_tbd() -> None:
    flagged = parse_game(_raw(tbd=True, start=PLACEHOLDER))
    assert flagged["kickoff_tbd"] == 1
    assert flagged["kickoff_utc"] == PLACEHOLDER
    real = parse_game(_raw(tbd=False, start=REAL_KICKOFF))
    assert real["kickoff_tbd"] == 0


def test_tbd_placeholder_accepts_a_2pm_et_rewrite(conn) -> None:
    _teams(conn)
    upsert_cfbd_game(
        conn,
        {
            "game_id": "cfbd:401870795",
            "season": 2026,
            "week": 10,
            "season_type": "regular",
            "kickoff_utc": PLACEHOLDER,
            "kickoff_tbd": 1,
            "football_date": "2026-11-07",
            "home_team_id": "cfbd:166",
            "away_team_id": "cfbd:2335",
            "completed": 0,
            "status": "scheduled",
            "source": "cfbd",
        },
    )
    rejected: list[str] = []
    written = persist_over_board(
        conn,
        "2026-11-07",
        [_pick("cfbd:401870795", PLACEHOLDER)],
        as_of_utc=TWO_PM_ET,
        rejected=rejected,
    )
    assert written == 1
    assert rejected == []


def test_refresh_replaces_a_tbd_placeholder_with_the_real_kickoff(conn, monkeypatch) -> None:
    _teams(conn)
    upsert_cfbd_game(
        conn,
        {
            "game_id": "cfbd:401870795",
            "season": 2026,
            "week": 10,
            "season_type": "regular",
            "kickoff_utc": PLACEHOLDER,
            "kickoff_tbd": 1,
            "football_date": "2026-11-07",
            "home_team_id": "cfbd:166",
            "away_team_id": "cfbd:2335",
            "completed": 0,
            "status": "scheduled",
            "source": "cfbd",
        },
    )

    class Client:
        def fetch_games(self, season, *, season_type="both", classification="fbs", refresh=False):
            return [_raw(tbd=False, start="2026-11-07T20:30:00.000Z")]

    monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: True)
    monkeypatch.setattr("cfb_analytics.sources.cfbd.CFBDClient", lambda *a, **k: Client())
    report = daily.DailyReport(started_utc="2026-11-07T16:00:00+00:00")
    daily._run_refresh_scores(
        conn, report, 2026, now=datetime(2026, 11, 7, 12, tzinfo=FOOTBALL_TZ)
    )
    stored = conn.execute(
        "SELECT kickoff_utc, kickoff_tbd FROM games WHERE game_id = 'cfbd:401870795'"
    ).fetchone()
    assert stored["kickoff_tbd"] == 0
    assert stored["kickoff_utc"].startswith("2026-11-07T20:30:00")


def test_real_kickoff_still_rejects_a_post_kickoff_write(conn) -> None:
    _teams(conn)
    upsert_cfbd_game(
        conn,
        {
            "game_id": "cfbd:401870795",
            "season": 2026,
            "week": 10,
            "season_type": "regular",
            "kickoff_utc": REAL_KICKOFF,
            "kickoff_tbd": 0,
            "football_date": "2026-11-07",
            "home_team_id": "cfbd:166",
            "away_team_id": "cfbd:2335",
            "completed": 0,
            "status": "scheduled",
            "source": "cfbd",
        },
    )
    rejected: list[str] = []
    written = persist_over_board(
        conn,
        "2026-11-07",
        [_pick("cfbd:401870795", REAL_KICKOFF)],
        as_of_utc=REAL_KICKOFF,
        rejected=rejected,
    )
    assert written == 0
    assert rejected == ["cfbd:401870795"]


def test_asof_cutoff_ignores_a_tbd_midnight_placeholder(conn) -> None:
    _teams(conn)
    upsert_cfbd_game(
        conn,
        {
            "game_id": "cfbd:tbd",
            "season": 2026,
            "week": 10,
            "season_type": "regular",
            "kickoff_utc": PLACEHOLDER,
            "kickoff_tbd": 1,
            "football_date": "2026-11-07",
            "home_team_id": "cfbd:166",
            "away_team_id": "cfbd:2335",
            "completed": 0,
            "source": "cfbd",
        },
    )
    upsert_cfbd_game(
        conn,
        {
            "game_id": "cfbd:real",
            "season": 2026,
            "week": 10,
            "season_type": "regular",
            "kickoff_utc": "2026-11-08T01:00:00+00:00",
            "kickoff_tbd": 0,
            "football_date": "2026-11-07",
            "home_team_id": "cfbd:2335",
            "away_team_id": "cfbd:166",
            "completed": 0,
            "source": "cfbd",
        },
    )
    weekly, _season_final = _cutoffs(conn, 2026)
    real = datetime.fromisoformat("2026-11-08T01:00:00+00:00")
    assert weekly[10] == (real + timedelta(days=1)).isoformat()
