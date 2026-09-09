"""Tests for live backtest metrics."""

from __future__ import annotations

import pytest

from cfb_analytics.backtest.live_harness import (
    DriveOutcomePrediction,
    TotalsPrediction,
    WinProbPrediction,
)
from cfb_analytics.backtest.live_metrics import (
    drive_outcome_accuracy,
    multinomial_log_loss,
    totals_by_game_phase,
    totals_mae,
    totals_rmse,
    win_prob_to_predictions,
)
from cfb_analytics.backtest.replay import CheckpointLabel
from cfb_analytics.features.live import DriveOutcome


def _make_wp(prob: float, won: bool) -> WinProbPrediction:
    return WinProbPrediction(
        game_id="g1",
        season=2024,
        week=1,
        checkpoint=CheckpointLabel(kind="drive_start", quarter=1, drive_number=1),
        predicted_home_win_prob=prob,
        actual_home_won=won,
        pregame_home_margin=0.0,
        seconds_remaining=3000,
    )


def _make_drive(
    td: float,
    fg: float,
    punt: float,
    to: float,
    safety: float,
    actual: str,
) -> DriveOutcomePrediction:
    return DriveOutcomePrediction(
        game_id="g1",
        season=2024,
        week=1,
        drive_number=1,
        start_yardline=75,
        predicted_td=td,
        predicted_fg=fg,
        predicted_punt=punt,
        predicted_turnover=to,
        predicted_safety=safety,
        actual_outcome=actual,
    )


def _make_totals(projected: float, actual: int, remaining: int) -> TotalsPrediction:
    return TotalsPrediction(
        game_id="g1",
        season=2024,
        week=1,
        checkpoint=CheckpointLabel(kind="drive_start", quarter=1),
        team_id="t1",
        projected_total=projected,
        actual_total=actual,
        current_score=0,
        remaining_seconds=remaining,
    )


def test_win_prob_conversion():
    preds = [_make_wp(0.7, True), _make_wp(0.3, False)]
    pairs = win_prob_to_predictions(preds)
    assert len(pairs) == 2
    assert pairs[0] == (0.7, True)
    assert pairs[1] == (0.3, False)


def test_multinomial_log_loss_perfect():
    """Perfect prediction gives low log loss."""
    preds = [_make_drive(0.99, 0.005, 0.002, 0.002, 0.001, DriveOutcome.TOUCHDOWN.value)]
    loss = multinomial_log_loss(preds)
    assert loss < 0.02  # Should be very close to 0


def test_multinomial_log_loss_wrong():
    """Wrong prediction gives high log loss."""
    preds = [_make_drive(0.01, 0.01, 0.01, 0.96, 0.01, DriveOutcome.TOUCHDOWN.value)]
    loss = multinomial_log_loss(preds)
    assert loss > 3.0  # -log(0.01) ~ 4.6


def test_multinomial_log_loss_empty():
    with pytest.raises(ValueError):
        multinomial_log_loss([])


def test_multinomial_log_loss_downs_mapped_to_turnover():
    """DOWNS outcome is mapped to TURNOVER class."""
    preds = [_make_drive(0.1, 0.1, 0.1, 0.6, 0.1, DriveOutcome.DOWNS.value)]
    loss = multinomial_log_loss(preds)
    # 0.6 predicted for TURNOVER, DOWNS maps to TURNOVER -> -log(0.6) ~ 0.51
    assert 0.4 < loss < 0.6


def test_drive_outcome_accuracy():
    preds = [
        _make_drive(0.5, 0.1, 0.2, 0.15, 0.05, DriveOutcome.TOUCHDOWN.value),
        _make_drive(0.1, 0.1, 0.6, 0.15, 0.05, DriveOutcome.PUNT.value),
        _make_drive(0.3, 0.1, 0.3, 0.25, 0.05, DriveOutcome.TOUCHDOWN.value),
    ]
    result = drive_outcome_accuracy(preds)
    assert DriveOutcome.TOUCHDOWN.value in result
    assert DriveOutcome.PUNT.value in result
    # Two TDs predicted (argmax), two actual
    assert result[DriveOutcome.TOUCHDOWN.value]["true_count"] == 2.0


def test_drive_outcome_accuracy_empty():
    assert drive_outcome_accuracy([]) == {}


def test_totals_mae():
    preds = [_make_totals(28.0, 21, 3000), _make_totals(14.0, 21, 3000)]
    mae = totals_mae(preds)
    assert mae == pytest.approx(7.0, abs=0.01)  # (7+7)/2


def test_totals_rmse():
    preds = [_make_totals(28.0, 21, 3000), _make_totals(14.0, 21, 3000)]
    rmse = totals_rmse(preds)
    assert rmse == pytest.approx(7.0, abs=0.01)  # sqrt((49+49)/2) = 7


def test_totals_empty():
    with pytest.raises(ValueError):
        totals_mae([])
    with pytest.raises(ValueError):
        totals_rmse([])


def test_totals_by_game_phase():
    preds = [
        _make_totals(28.0, 21, 3000),  # early
        _make_totals(21.0, 21, 1500),  # mid
        _make_totals(21.0, 21, 500),  # late
    ]
    result = totals_by_game_phase(preds)
    assert "early" in result
    assert "mid" in result
    assert "late" in result
    assert result["early"]["n"] == 1.0
    assert result["mid"]["n"] == 1.0
    assert result["late"]["n"] == 1.0
    assert result["mid"]["mae"] == 0.0  # Perfect prediction
    assert result["late"]["mae"] == 0.0
