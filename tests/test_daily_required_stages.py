"""Rankings, the OVER board, and the slate report must fail the daily run.

Each stage catches its own error and records status ``failed``. A sibling leg
that succeeded used to leave ``DailyReport.ok`` true and ``daily_exit_code``
at 0.
"""

from __future__ import annotations

from cfb_analytics import daily
from cfb_analytics.errors import SourceError
from tests.test_daily_mandatory_sources import _run


def _assert_failed_stage(report: daily.DailyReport, capsys, name: str) -> None:
    code = daily.daily_exit_code(report)
    captured = capsys.readouterr()
    failed = next(outcome for outcome in report.outcomes if outcome.name == name)
    assert failed.status == "failed"
    assert any(outcome.status == "ok" for outcome in report.outcomes)
    assert report.ok is False
    assert code == 1
    assert f"::error::mandatory CFBD source {name} failed" in captured.out


def test_rankings_failure_fails_the_run(conn, monkeypatch, capsys):
    report = _run(conn, monkeypatch, failing="rankings")
    rankings = next(outcome for outcome in report.outcomes if outcome.name == "rankings")
    assert "429" in rankings.detail
    _assert_failed_stage(report, capsys, "rankings")


def test_over_board_failure_fails_the_run(conn, monkeypatch, capsys):
    def _boom(*_args, **_kwargs):
        raise SourceError("over board unavailable")

    monkeypatch.setattr(
        "cfb_analytics.features.over_confidence.build_over_confidence_board",
        _boom,
    )
    report = _run(conn, monkeypatch, failing=None)
    _assert_failed_stage(report, capsys, "over_board")


def test_slate_report_failure_fails_the_run(conn, monkeypatch, capsys):
    def _boom(*_args, **_kwargs):
        raise SourceError("slate report unavailable")

    monkeypatch.setattr(
        "cfb_analytics.reporting.slate.build_slate_report",
        _boom,
    )
    report = _run(conn, monkeypatch, failing=None)
    _assert_failed_stage(report, capsys, "slate_report")
