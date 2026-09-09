"""Scoring metrics for the live micro-markets backtest.

Reuses ``metrics.py`` for binary win probability scoring (Brier, log loss,
buckets, reliability) and adds multi-class drive outcome metrics and
team totals error metrics.
"""

from __future__ import annotations

import math

from cfb_analytics.backtest.live_harness import (
    DriveOutcomePrediction,
    TotalsPrediction,
    WinProbPrediction,
)
from cfb_analytics.backtest.metrics import (
    Prediction,
)
from cfb_analytics.features.live import DriveOutcome

_LOGLOSS_EPS = 1e-12

# Canonical outcome classes for drive scoring
DRIVE_OUTCOME_CLASSES: tuple[str, ...] = (
    DriveOutcome.TOUCHDOWN.value,
    DriveOutcome.FIELD_GOAL.value,
    DriveOutcome.PUNT.value,
    DriveOutcome.TURNOVER.value,
    DriveOutcome.SAFETY.value,
)


def win_prob_to_predictions(
    preds: list[WinProbPrediction],
) -> list[Prediction]:
    """Convert live win probability predictions to (prob, outcome) pairs."""
    return [(p.predicted_home_win_prob, p.actual_home_won) for p in preds]


def multinomial_log_loss(predictions: list[DriveOutcomePrediction]) -> float:
    """Mean multinomial negative log-likelihood over drive outcome predictions.

    For each drive, the model predicted a 5-class probability distribution
    and one class actually occurred. Scoring follows standard multi-class
    log loss.
    """
    if not predictions:
        raise ValueError("no predictions to score")

    total = 0.0
    for p in predictions:
        probs = {
            DriveOutcome.TOUCHDOWN.value: p.predicted_td,
            DriveOutcome.FIELD_GOAL.value: p.predicted_fg,
            DriveOutcome.PUNT.value: p.predicted_punt,
            DriveOutcome.TURNOVER.value: p.predicted_turnover,
            DriveOutcome.SAFETY.value: p.predicted_safety,
        }
        # Map actual outcome to our canonical classes
        actual = p.actual_outcome
        if actual in (DriveOutcome.DOWNS.value,):
            actual = DriveOutcome.TURNOVER.value  # DOWNS -> TURNOVER class
        if actual in (DriveOutcome.END_HALF.value, DriveOutcome.END_GAME.value):
            actual = DriveOutcome.PUNT.value  # Time-expiring drives -> PUNT class

        prob_actual = probs.get(actual, 0.0)
        clipped = min(max(prob_actual, _LOGLOSS_EPS), 1.0 - _LOGLOSS_EPS)
        total += -math.log(clipped)

    return total / len(predictions)


def drive_outcome_accuracy(
    predictions: list[DriveOutcomePrediction],
) -> dict[str, dict[str, float]]:
    """Per-class precision and recall for drive outcome predictions.

    Returns a dict keyed by outcome class with precision, recall, and count.
    """
    if not predictions:
        return {}

    # Map DOWNS/END_HALF/END_GAME to canonical classes
    def _canonical(outcome: str) -> str:
        if outcome in (DriveOutcome.DOWNS.value,):
            return DriveOutcome.TURNOVER.value
        if outcome in (DriveOutcome.END_HALF.value, DriveOutcome.END_GAME.value):
            return DriveOutcome.PUNT.value
        return outcome

    true_counts: dict[str, int] = {c: 0 for c in DRIVE_OUTCOME_CLASSES}
    pred_counts: dict[str, int] = {c: 0 for c in DRIVE_OUTCOME_CLASSES}
    correct_counts: dict[str, int] = {c: 0 for c in DRIVE_OUTCOME_CLASSES}

    for p in predictions:
        probs = {
            DriveOutcome.TOUCHDOWN.value: p.predicted_td,
            DriveOutcome.FIELD_GOAL.value: p.predicted_fg,
            DriveOutcome.PUNT.value: p.predicted_punt,
            DriveOutcome.TURNOVER.value: p.predicted_turnover,
            DriveOutcome.SAFETY.value: p.predicted_safety,
        }
        predicted_class = max(probs, key=probs.get)  # type: ignore[arg-type]
        actual_class = _canonical(p.actual_outcome)

        if actual_class in true_counts:
            true_counts[actual_class] += 1
        if predicted_class in pred_counts:
            pred_counts[predicted_class] += 1
        if predicted_class == actual_class and predicted_class in correct_counts:
            correct_counts[predicted_class] += 1

    result: dict[str, dict[str, float]] = {}
    for cls in DRIVE_OUTCOME_CLASSES:
        precision = correct_counts[cls] / pred_counts[cls] if pred_counts[cls] > 0 else 0.0
        recall = correct_counts[cls] / true_counts[cls] if true_counts[cls] > 0 else 0.0
        result[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "true_count": float(true_counts[cls]),
            "pred_count": float(pred_counts[cls]),
        }
    return result


def totals_mae(predictions: list[TotalsPrediction]) -> float:
    """Mean absolute error of team total projections."""
    if not predictions:
        raise ValueError("no predictions to score")
    return sum(abs(p.projected_total - p.actual_total) for p in predictions) / len(predictions)


def totals_rmse(predictions: list[TotalsPrediction]) -> float:
    """Root mean squared error of team total projections."""
    if not predictions:
        raise ValueError("no predictions to score")
    mse = sum((p.projected_total - p.actual_total) ** 2 for p in predictions) / len(predictions)
    return math.sqrt(mse)


def totals_by_game_phase(
    predictions: list[TotalsPrediction],
) -> dict[str, dict[str, float]]:
    """MAE and RMSE bucketed by game phase (checkpoint kind).

    Phases: early (>2700s remaining), mid (900-2700s), late (<900s).
    """
    phases: dict[str, list[TotalsPrediction]] = {
        "early": [],
        "mid": [],
        "late": [],
    }
    for p in predictions:
        if p.remaining_seconds > 2700:
            phases["early"].append(p)
        elif p.remaining_seconds > 900:
            phases["mid"].append(p)
        else:
            phases["late"].append(p)

    result: dict[str, dict[str, float]] = {}
    for phase, preds in phases.items():
        if not preds:
            result[phase] = {"mae": 0.0, "rmse": 0.0, "n": 0.0}
        else:
            result[phase] = {
                "mae": round(totals_mae(preds), 2),
                "rmse": round(totals_rmse(preds), 2),
                "n": float(len(preds)),
            }
    return result
