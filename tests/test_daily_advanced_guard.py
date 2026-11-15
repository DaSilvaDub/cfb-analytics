"""The daily job fails when the current season has no advanced-stat rows.

The scheduled job grew that guard without a fetch. These tests drive the
current-season fetch through the CFBD HTTP replay cache (no live API) and
keep the zero-row failure when that fetch returns nothing.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

from cfb_analytics import daily, paths
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_boxes import BoxSummary
from cfb_analytics.ingest.cfbd_fundamentals import EloBackfillSummary
from cfb_analytics.ingest.cfbd_lines import LinesIngestSummary
from cfb_analytics.ingest.cfbd_rankings import RankingsSummary
from cfb_analytics.sources.cfbd import CFBDClient
from cfb_analytics.sources.http import cache_key
from cfb_analytics.utils import FOOTBALL_TZ

NOW = datetime(2026, 11, 14, 12, tzinfo=FOOTBALL_TZ)
WEEK_11_KICKOFF = "2026-11-07T23:00:00+00:00"


def test_zero_team_season_advanced_rows_fail_with_an_error_annotation(conn, capsys, monkeypatch):
    monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
    report = daily.run_daily(
        conn,
        season=2026,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=datetime(2026, 11, 8, 12, tzinfo=FOOTBALL_TZ),
    )
    code = daily.daily_exit_code(report)
    outcome = next(item for item in report.outcomes if item.name == "team_advanced")
    assert outcome.status == "failed"
    assert "0 team_season_advanced" in outcome.detail
    assert report.ok is False
    assert code == 1
    assert "::error::mandatory CFBD source team_advanced failed" in capsys.readouterr().out


def _record_replay(url: str, payload: list) -> None:
    path = paths.cache_dir() / "cfbd" / f"{cache_key(url)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"url": url, "fetched_at": time.time(), "payload": payload}),
        encoding="utf-8",
    )


def _side(ppa: float) -> dict:
    return {
        "plays": 80,
        "drives": 15,
        "ppa": ppa,
        "totalPPA": ppa * 80,
        "successRate": 0.45,
        "explosiveness": 1.1,
        "pointsPerOpportunity": 3.2,
        "havoc": {"total": 0.12},
        "lineYards": 2.7,
        "stuffRate": 0.19,
        "passingPlays": {"ppa": ppa, "successRate": 0.47},
        "rushingPlays": {"ppa": ppa / 2, "successRate": 0.41},
    }


def _advanced_team(school: str, *, ppa: float) -> dict:
    return {"season": 2026, "team": school, "offense": _side(ppa), "defense": _side(ppa / 2)}


def _fbs_teams() -> list[dict]:
    return [
        {
            "id": 1,
            "school": "Home U",
            "abbreviation": "HOM",
            "alternateNames": [],
            "conference": "Test",
            "classification": "fbs",
        },
        {
            "id": 2,
            "school": "Away U",
            "abbreviation": "AWY",
            "alternateNames": [],
            "conference": "Test",
            "classification": "fbs",
        },
    ]


def _seed_started_and_future_weeks(conn) -> None:
    """Week 11 has started by ``NOW``. Week 13 has not, so it must not be fetched."""
    store.upsert_team(
        conn, {"team_id": "cfbd:1", "school": "Home U", "alias": "HOM", "market": "Home U"}
    )
    store.upsert_team(
        conn, {"team_id": "cfbd:2", "school": "Away U", "alias": "AWY", "market": "Away U"}
    )
    for game_id, week, kickoff, football_date in (
        ("cfbd:w11", 11, WEEK_11_KICKOFF, "2026-11-07"),
        ("cfbd:w13", 13, "2026-11-28T23:00:00+00:00", "2026-11-28"),
    ):
        store.upsert_cfbd_game(
            conn,
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "season_type": "regular",
                "kickoff_utc": kickoff,
                "football_date": football_date,
                "neutral_site": 0,
                "conference_game": 0,
                "home_team_id": "cfbd:1",
                "away_team_id": "cfbd:2",
                "venue_name": None,
                "venue_id": None,
                "status": "scheduled",
                "home_points": None,
                "away_points": None,
                "completed": 0,
                "source": "cfbd",
                "kickoff_tbd": 0,
            },
        )


def _silence_other_cfbd_legs(monkeypatch) -> None:
    """Leave the advanced fetch on the real client. Replay serves its cache."""
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_lines.ingest_lines",
        lambda *_a, **_k: LinesIngestSummary(year=2026),
    )
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_rankings.ingest_rankings",
        lambda *_a, **_k: RankingsSummary(weeks=0, rows=0, unresolved=0),
    )
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_fundamentals.backfill_elo",
        lambda *_a, **_k: EloBackfillSummary(seasons=1, endpoints=0, ratings=0, filtered=0),
    )
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_boxes.ingest_completed_boxes",
        lambda *_a, **_k: BoxSummary(games=0, rows=0, unresolved=0),
    )
    monkeypatch.setattr(daily, "_run_refresh_scores", lambda *_a, **_k: None)


def _advanced_urls() -> tuple[str, str]:
    client = CFBDClient()
    teams = client._url("/teams/fbs", year=2026)
    advanced = client._url(
        "/stats/season/advanced",
        year=2026,
        excludeGarbageTime=True,
        startWeek=1,
        endWeek=11,
        classification="fbs",
    )
    return teams, advanced


def _run(conn, monkeypatch, *, advanced_payload: list):
    monkeypatch.setenv("CFBD_API_KEY", "test-key")
    _seed_started_and_future_weeks(conn)
    _silence_other_cfbd_legs(monkeypatch)
    teams_url, advanced_url = _advanced_urls()
    _record_replay(teams_url, _fbs_teams())
    _record_replay(advanced_url, advanced_payload)
    return daily.run_daily(
        conn,
        season=2026,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=NOW,
    ), advanced_url


def _season_count(conn) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM team_season_advanced WHERE season = 2026"
        ).fetchone()["n"]
    )


def test_daily_fetch_fills_an_empty_season_and_the_guard_passes(conn, capsys, monkeypatch):
    """A daily run that starts with zero current-season rows must store the
    replayed advanced payload and clear the guard.

    On master the job never calls ``fetch_advanced``, so this fails with
    ``season 2026 has 0 team_season_advanced rows``.
    """
    assert _season_count(conn) == 0
    report, advanced_url = _run(
        conn,
        monkeypatch,
        advanced_payload=[_advanced_team("Home U", ppa=0.2), _advanced_team("Away U", ppa=0.3)],
    )
    outcome = next(item for item in report.outcomes if item.name == "team_advanced")
    assert outcome.status == "ok", outcome.detail
    assert outcome.rows == 4
    assert _season_count(conn) == 4
    weeks = [
        row["week"]
        for row in conn.execute(
            "SELECT DISTINCT week FROM team_season_advanced WHERE season = 2026 ORDER BY week"
        )
    ]
    assert weeks == [11]
    kickoff = datetime.fromisoformat(WEEK_11_KICKOFF)
    stamps = {
        row["as_of_utc"]
        for row in conn.execute("SELECT as_of_utc FROM team_season_advanced WHERE season = 2026")
    }
    assert stamps == {(kickoff + timedelta(days=1)).isoformat()}
    endpoints = {
        row["endpoint"]
        for row in conn.execute(
            """SELECT endpoint FROM source_health
               WHERE endpoint LIKE 'stats/season/advanced:%'"""
        )
    }
    assert endpoints == {"stats/season/advanced:2026:week:11"}
    assert report.ok is True
    assert daily.daily_exit_code(report) == 0
    assert "::error::mandatory CFBD source team_advanced failed" not in capsys.readouterr().out

    _record_replay(
        advanced_url,
        [_advanced_team("Home U", ppa=0.8), _advanced_team("Away U", ppa=0.3)],
    )
    second = daily.run_daily(
        conn,
        season=2026,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=NOW,
    )
    second_guard = next(item for item in second.outcomes if item.name == "team_advanced")
    assert second_guard.status == "ok"
    assert _season_count(conn) == 4
    home_off = conn.execute(
        """SELECT ppa FROM team_season_advanced
           WHERE season = 2026 AND week = 11 AND team_id = 'cfbd:1' AND side = 'off'"""
    ).fetchone()
    assert home_off["ppa"] == 0.8


def test_guard_still_fails_when_the_advanced_fetch_returns_nothing(conn, capsys, monkeypatch):
    assert _season_count(conn) == 0
    report, _url = _run(conn, monkeypatch, advanced_payload=[])
    outcome = next(item for item in report.outcomes if item.name == "team_advanced")
    assert outcome.status == "failed"
    assert outcome.detail == "season 2026 has 0 team_season_advanced rows"
    assert _season_count(conn) == 0
    health = conn.execute(
        """SELECT ok, rows FROM source_health
           WHERE endpoint = 'stats/season/advanced:2026:week:11'"""
    ).fetchone()
    assert health is not None
    assert health["ok"] == 1
    assert health["rows"] == 0
    assert report.ok is False
    code = daily.daily_exit_code(report)
    assert code == 1
    assert (
        "::error::mandatory CFBD source team_advanced failed: "
        "season 2026 has 0 team_season_advanced rows"
    ) in capsys.readouterr().out
