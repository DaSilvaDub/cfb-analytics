"""Mandatory CFBD legs fail the daily job, including HTTP 429.

On 2026-10-31 the daily job logged FAIL for lines, rankings, Elo, game
scores, and team boxes (all HTTP 429) and OK for player passing, internal
ratings, internal Elo, weather, and the slate report. ``DailyReport.ok`` was
true because any success counted, so the step exited 0 and the partial store
was published.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

from cfb_analytics import cli, daily
from cfb_analytics.errors import SourceError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_boxes import BoxSummary
from cfb_analytics.ingest.cfbd_fundamentals import EloBackfillSummary
from cfb_analytics.ingest.cfbd_lines import LinesIngestSummary
from cfb_analytics.ingest.cfbd_rankings import RankingsSummary
from cfb_analytics.utils import FOOTBALL_TZ

RATE_LIMIT = "[cfbd] HTTP 429 for https://api.collegefootballdata.com/example"
NOW = datetime(2026, 10, 31, 12, tzinfo=FOOTBALL_TZ)


def _seed_slate_game(conn) -> None:
    store.upsert_team(
        conn, {"team_id": "cfbd:1", "school": "Home U", "alias": "H", "market": "Home U"}
    )
    store.upsert_team(
        conn, {"team_id": "cfbd:2", "school": "Away U", "alias": "A", "market": "Away U"}
    )
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "cfbd:1",
            "season": 2026,
            "week": 9,
            "season_type": "regular",
            "kickoff_utc": "2026-10-31T23:00:00+00:00",
            "football_date": "2026-10-31",
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
        },
    )


def _ok_lines(*_a, **_k):
    return LinesIngestSummary(year=2026, games_seen=1, games_matched=1, odds_rows=1)


def _ok_rankings(*_a, **_k):
    return RankingsSummary(weeks=1, rows=1, unresolved=0)


def _ok_elo(*_a, **_k):
    return EloBackfillSummary(seasons=1, endpoints=1, ratings=1, filtered=0)


def _ok_boxes(*_a, **_k):
    return BoxSummary(games=1, rows=2, unresolved=0)


def _rate_limit(*_a, **_k):
    raise SourceError(RATE_LIMIT)


class _GamesClient:
    def __init__(self, *args, **kwargs):
        self.fail = bool(kwargs.get("fail"))

    def fetch_games(self, *args, **kwargs):
        if self.fail:
            raise SourceError(RATE_LIMIT)
        return []


def _patch_legs(monkeypatch, *, failing: str | None) -> None:
    monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: True)
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_lines.ingest_lines",
        _rate_limit if failing == "cfbd" else _ok_lines,
    )
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_rankings.ingest_rankings",
        _rate_limit if failing == "rankings" else _ok_rankings,
    )
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_fundamentals.backfill_elo",
        _rate_limit if failing == "elo" else _ok_elo,
    )
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_boxes.ingest_completed_boxes",
        _rate_limit if failing == "team_boxes" else _ok_boxes,
    )

    class Client(_GamesClient):
        def __init__(self, *args, **kwargs):
            super().__init__(fail=failing == "game_scores")

    monkeypatch.setattr("cfb_analytics.sources.cfbd.CFBDClient", Client)


def _run(conn, monkeypatch, *, failing: str | None):
    _seed_slate_game(conn)
    _patch_legs(monkeypatch, failing=failing)
    return daily.run_daily(
        conn,
        season=2026,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        now=NOW,
    )


@pytest.mark.parametrize("leg", ["cfbd", "game_scores", "team_boxes", "elo"])
def test_http_429_on_each_mandatory_leg_exits_nonzero(conn, monkeypatch, capsys, leg):
    report = _run(conn, monkeypatch, failing=leg)
    code = daily.daily_exit_code(report)
    captured = capsys.readouterr()

    assert any(outcome.status == "ok" for outcome in report.outcomes)
    failed = next(outcome for outcome in report.outcomes if outcome.name == leg)
    assert failed.status == "failed"
    assert "429" in failed.detail
    assert report.ok is False
    assert code == 1
    assert f"::error::mandatory CFBD source {leg} failed" in captured.out
    assert "429" in captured.out


def test_oct31_mix_of_429s_and_successes_exits_nonzero(capsys):
    """The outcome mix from daily-ingest run 41195272529."""
    report = daily.DailyReport(started_utc="2026-10-31T11:03:53+00:00")
    report.outcomes = [
        daily.SourceOutcome("cfbd", "failed", "[cfbd] HTTP 429 for lines"),
        daily.SourceOutcome("rankings", "failed", "[cfbd] HTTP 429 for rankings"),
        daily.SourceOutcome("elo", "failed", "[cfbd] HTTP 429 for teams/fbs"),
        daily.SourceOutcome("game_scores", "failed", "[cfbd] HTTP 429 for games"),
        daily.SourceOutcome("player_passing", "ok", "already current"),
        daily.SourceOutcome("team_boxes", "failed", "[cfbd] HTTP 429 for games/teams"),
        daily.SourceOutcome("internal_ratings", "ok", "fit"),
        daily.SourceOutcome("internal_elo", "ok", "fit"),
        daily.SourceOutcome("weather", "ok", "13 observations"),
        daily.SourceOutcome("slate_report", "ok", "slate"),
    ]

    code = daily.daily_exit_code(report)
    captured = capsys.readouterr()

    assert report.ok is False
    assert code == 1
    for name in ("cfbd", "elo", "game_scores", "team_boxes"):
        assert f"::error::mandatory CFBD source {name} failed" in captured.out
    assert "mandatory CFBD source rankings failed" not in captured.out


def test_rankings_429_alone_does_not_fail_the_run(conn, monkeypatch, capsys):
    report = _run(conn, monkeypatch, failing="rankings")
    code = daily.daily_exit_code(report)
    captured = capsys.readouterr()

    rankings = next(outcome for outcome in report.outcomes if outcome.name == "rankings")
    assert rankings.status == "failed"
    assert "429" in rankings.detail
    assert report.ok is True
    assert code == 0
    assert "::error::" not in captured.out


def test_missing_cfbd_key_still_exits_zero(conn, monkeypatch, capsys):
    _seed_slate_game(conn)
    monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: False)
    report = daily.run_daily(
        conn,
        season=2026,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        now=NOW,
    )
    code = daily.daily_exit_code(report)
    captured = capsys.readouterr()

    assert report.ok is True
    assert code == 0
    assert "::error::" not in captured.out


def test_cli_daily_returns_nonzero_and_annotates_a_mandatory_429(conn, monkeypatch, capsys):
    report = daily.DailyReport(started_utc="2026-10-31T11:03:53+00:00")
    report.outcomes = [
        daily.SourceOutcome("elo", "failed", RATE_LIMIT),
        daily.SourceOutcome("internal_ratings", "ok", "fit"),
    ]

    @contextmanager
    def _open(*_a, **_k):
        yield conn

    monkeypatch.setattr("cfb_analytics.db.open_db", _open)
    monkeypatch.setattr("cfb_analytics.daily.run_daily", lambda *_a, **_k: report)
    args = cli.build_parser().parse_args(["daily", "--no-outlier", "--season", "2026"])

    code = cli._cmd_daily(args)
    captured = capsys.readouterr()

    assert code == 1
    assert "::error::mandatory CFBD source elo failed" in captured.out
    assert "429" in captured.out


def test_publish_step_runs_only_after_a_successful_daily_ingest():
    text = Path(__file__).resolve().parents[1].joinpath(
        ".github", "workflows", "daily-ingest.yml"
    ).read_text(encoding="utf-8")
    assert "id: daily" in text
    assert "steps.daily.outcome == 'success'" in text
    assert "REFUSING TO PUBLISH" in text
