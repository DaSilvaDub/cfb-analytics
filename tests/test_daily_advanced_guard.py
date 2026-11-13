"""The daily job fails when the current season has no advanced-stat rows."""

from __future__ import annotations

from datetime import datetime

from cfb_analytics import daily
from cfb_analytics.utils import FOOTBALL_TZ


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
