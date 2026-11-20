"""Required skips, unresolved poll names, and OVER-board errors fail closed."""

from __future__ import annotations

import pytest

from cfb_analytics import daily
from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest.cfbd_rankings import RankingsSummary, ingest_rankings
from cfb_analytics.reporting.slate import build_slate_report


def test_skipped_required_stage_beside_a_success_is_not_green(capsys) -> None:
    report = daily.DailyReport(started_utc="2026-11-20T12:00:00+00:00")
    report.outcomes = [
        daily.SourceOutcome("cfbd", "ok", "lines"),
        daily.SourceOutcome("rankings", "skipped", "no CFBD_API_KEY configured"),
        daily.SourceOutcome("over_board", "ok", "board"),
        daily.SourceOutcome("slate_report", "ok", "report"),
    ]
    assert report.ok is False
    code = daily.daily_exit_code(report)
    assert code == 1
    assert "::error::required stage rankings skipped" in capsys.readouterr().out


def test_an_all_skipped_run_stays_green(capsys) -> None:
    report = daily.DailyReport(started_utc="2026-11-20T12:00:00+00:00")
    report.outcomes = [
        daily.SourceOutcome("cfbd", "skipped", "no key"),
        daily.SourceOutcome("rankings", "skipped", "no key"),
        daily.SourceOutcome("elo", "skipped", "no key"),
        daily.SourceOutcome("team_boxes", "skipped", "no key"),
    ]
    assert report.ok is True
    assert daily.daily_exit_code(report) == 0
    assert "::error::" not in capsys.readouterr().out


class _Rankings:
    def __init__(self) -> None:
        self.weeks = 1
        self.rows = 1
        self.unresolved = 52
        self.unresolved_names = tuple(f"School {i}" for i in range(52))


def test_unresolved_ranking_names_fail_the_stage_and_are_named(conn, monkeypatch) -> None:
    monkeypatch.setenv("CFBD_API_KEY", "test-key")
    monkeypatch.setattr(
        "cfb_analytics.ingest.cfbd_rankings.ingest_rankings",
        lambda *_a, **_k: _Rankings(),
    )
    report = daily.DailyReport(started_utc="2026-11-20T12:00:00+00:00")
    daily._run_cfbd_rankings(conn, report, 2026, now=None)
    outcome = report.outcomes[0]
    assert outcome.status == "failed"
    assert "52" in outcome.detail
    assert "School 0" in outcome.detail
    assert "School 51" in outcome.detail
    assert report.ok is False


def test_rankings_with_no_unresolved_names_stay_ok(conn, monkeypatch) -> None:
    monkeypatch.setenv("CFBD_API_KEY", "test-key")

    def _ok(*_a, **_k):
        return RankingsSummary(weeks=1, rows=25, unresolved=0)

    monkeypatch.setattr("cfb_analytics.ingest.cfbd_rankings.ingest_rankings", _ok)
    report = daily.DailyReport(started_utc="2026-11-20T12:00:00+00:00")
    daily._run_cfbd_rankings(conn, report, 2026)
    assert report.outcomes[0].status == "ok"


class _Client:
    def fetch_rankings(self, year, week=None, season_type="regular"):
        ranks = [
            {"rank": 1, "school": "Home U", "points": 1500, "firstPlaceVotes": 10},
            *[{"rank": i, "school": f"Unknown {i}", "points": 1} for i in range(2, 53)],
        ]
        return [
            {
                "season": year,
                "week": 10,
                "seasonType": season_type,
                "polls": [{"poll": "AP Top 25", "ranks": ranks}],
            }
        ]


def test_ingest_rankings_names_the_schools_it_could_not_resolve(conn) -> None:
    from tests.conftest import seed_canonical_team

    seed_canonical_team(conn, 1, "Home U", "HOM")
    summary = ingest_rankings(conn, _Client(), 2026, week=10)
    assert summary.rows == 1
    assert summary.unresolved == 51
    assert summary.unresolved_names[0] == "Unknown 2"
    assert "Unknown 52" in summary.unresolved_names
    assert "Home U" not in summary.unresolved_names


def test_slate_report_does_not_swallow_over_board_errors(conn, monkeypatch) -> None:
    def boom(*_a, **_k):
        raise SchemaError("over board exploded")

    monkeypatch.setattr("cfb_analytics.features.over_confidence.load_published_board", boom)
    with pytest.raises(SchemaError, match="over board exploded"):
        build_slate_report(conn, "2026-11-20")
