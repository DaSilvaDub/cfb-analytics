"""Tests for the live backtest report."""

from __future__ import annotations

from cfb_analytics.backtest.live_report import (
    LiveBacktestReport,
    LiveDriveOutcomeSlice,
    LiveTotalsSlice,
    LiveWinProbSlice,
)


def _make_report(*, with_stress: bool = False) -> LiveBacktestReport:
    wp = LiveWinProbSlice(
        label="2014-2025 (excl. 2020)",
        n_predictions=100,
        brier=0.22,
        log_loss=0.65,
    )
    drive = LiveDriveOutcomeSlice(
        label="2014-2025 (excl. 2020)",
        n_predictions=200,
        multinomial_log_loss=1.4,
        per_class={
            "TD": {"precision": 0.3, "recall": 0.4, "true_count": 50.0, "pred_count": 60.0},
            "FG": {"precision": 0.2, "recall": 0.3, "true_count": 30.0, "pred_count": 40.0},
            "PUNT": {"precision": 0.4, "recall": 0.5, "true_count": 80.0, "pred_count": 70.0},
            "TURNOVER": {"precision": 0.2, "recall": 0.2, "true_count": 30.0, "pred_count": 25.0},
            "SAFETY": {"precision": 0.0, "recall": 0.0, "true_count": 10.0, "pred_count": 5.0},
        },
    )
    totals = LiveTotalsSlice(
        label="2014-2025 (excl. 2020)",
        n_predictions=400,
        mae=6.5,
        rmse=8.2,
        by_phase={
            "early": {"mae": 8.0, "rmse": 10.0, "n": 150.0},
            "mid": {"mae": 5.0, "rmse": 6.5, "n": 150.0},
            "late": {"mae": 3.0, "rmse": 4.0, "n": 100.0},
        },
    )
    stress_wp = None
    stress_drive = None
    stress_totals = None
    if with_stress:
        stress_wp = LiveWinProbSlice(
            label="2020 stress slice",
            n_predictions=20,
            brier=0.25,
            log_loss=0.70,
        )
        stress_drive = LiveDriveOutcomeSlice(
            label="2020 stress slice",
            n_predictions=40,
            multinomial_log_loss=1.5,
            per_class={},
        )
        stress_totals = LiveTotalsSlice(
            label="2020 stress slice",
            n_predictions=80,
            mae=7.0,
            rmse=9.0,
            by_phase={},
        )
    return LiveBacktestReport(
        games_replayed=50,
        games_skipped_no_pbp=5,
        win_prob=wp,
        win_prob_stress=stress_wp,
        drive_outcomes=drive,
        drive_outcomes_stress=stress_drive,
        totals=totals,
        totals_stress=stress_totals,
    )


def test_report_as_text():
    report = _make_report()
    text = report.as_text()
    assert "live micro-markets backtest" in text
    assert "games replayed" in text
    assert "50" in text
    assert "brier=0.2200" in text
    assert "multinomial_log_loss=1.4000" in text
    assert "mae=6.50" in text
    assert "uncalibrated_shadow" in text


def test_report_with_stress():
    report = _make_report(with_stress=True)
    text = report.as_text()
    assert "2020 stress slice" in text


def test_report_model_status():
    report = _make_report()
    assert report.model_status == "uncalibrated_shadow"


def test_report_no_stress_does_not_crash():
    report = _make_report(with_stress=False)
    text = report.as_text()
    assert "2020 stress slice" not in text


def test_report_structure():
    report = _make_report()
    assert report.win_prob.n_predictions == 100
    assert report.drive_outcomes.n_predictions == 200
    assert report.totals.n_predictions == 400
    assert report.games_replayed == 50
    assert report.games_skipped_no_pbp == 5
