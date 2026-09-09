"""Orchestrator for the live micro-markets backtest.

Ties together the walk-forward harness, metrics computation, and report
generation. Follows the same fit/stress split as the moneyline backtest
(2020 is the stress season).
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.backtest.live_harness import (
    DriveOutcomePrediction,
    TotalsPrediction,
    WinProbPrediction,
    run_live_walk_forward,
)
from cfb_analytics.backtest.live_metrics import (
    drive_outcome_accuracy,
    multinomial_log_loss,
    totals_by_game_phase,
    totals_mae,
    totals_rmse,
    win_prob_to_predictions,
)
from cfb_analytics.backtest.live_report import (
    LiveBacktestReport,
    LiveDriveOutcomeSlice,
    LiveTotalsSlice,
    LiveWinProbSlice,
)
from cfb_analytics.backtest.metrics import brier_score, log_loss

STRESS_SEASONS = frozenset({2020})
DEFAULT_LIVE_SEASONS = tuple(range(2014, 2026))


def _score_win_prob_slice(label: str, preds: list[WinProbPrediction]) -> LiveWinProbSlice | None:
    if not preds:
        return None
    pairs = win_prob_to_predictions(preds)
    return LiveWinProbSlice(
        label=label,
        n_predictions=len(pairs),
        brier=brier_score(pairs),
        log_loss=log_loss(pairs),
    )


def _score_drive_slice(
    label: str, preds: list[DriveOutcomePrediction]
) -> LiveDriveOutcomeSlice | None:
    if not preds:
        return None
    return LiveDriveOutcomeSlice(
        label=label,
        n_predictions=len(preds),
        multinomial_log_loss=multinomial_log_loss(preds),
        per_class=drive_outcome_accuracy(preds),
    )


def _score_totals_slice(label: str, preds: list[TotalsPrediction]) -> LiveTotalsSlice | None:
    if not preds:
        return None
    return LiveTotalsSlice(
        label=label,
        n_predictions=len(preds),
        mae=round(totals_mae(preds), 2),
        rmse=round(totals_rmse(preds), 2),
        by_phase=totals_by_game_phase(preds),
    )


def run_live_backtest(
    conn: sqlite3.Connection,
    seasons: tuple[int, ...] = DEFAULT_LIVE_SEASONS,
) -> LiveBacktestReport:
    """Run the full live micro-markets backtest.

    1. Walk-forward replay of all games in the given seasons.
    2. Split predictions into fit and stress (2020) slices.
    3. Compute win probability, drive outcome, and totals metrics.
    4. Return a structured report.
    """
    run = run_live_walk_forward(conn, list(seasons))

    # Split into fit and stress
    fit_wp = [p for p in run.win_prob_predictions if p.season not in STRESS_SEASONS]
    stress_wp = [p for p in run.win_prob_predictions if p.season in STRESS_SEASONS]

    fit_drive = [p for p in run.drive_outcome_predictions if p.season not in STRESS_SEASONS]
    stress_drive = [p for p in run.drive_outcome_predictions if p.season in STRESS_SEASONS]

    fit_totals = [p for p in run.totals_predictions if p.season not in STRESS_SEASONS]
    stress_totals = [p for p in run.totals_predictions if p.season in STRESS_SEASONS]

    fit_seasons = [s for s in seasons if s not in STRESS_SEASONS]
    fit_label = f"{min(fit_seasons)}-{max(fit_seasons)} (excl. 2020)" if fit_seasons else "fit"

    wp_fit = _score_win_prob_slice(fit_label, fit_wp)
    wp_stress = _score_win_prob_slice("2020 stress slice", stress_wp)

    drive_fit = _score_drive_slice(fit_label, fit_drive)
    drive_stress = _score_drive_slice("2020 stress slice", stress_drive)

    totals_fit = _score_totals_slice(fit_label, fit_totals)
    totals_stress = _score_totals_slice("2020 stress slice", stress_totals)

    # Provide defaults for empty slices
    empty_wp = LiveWinProbSlice(label=fit_label, n_predictions=0, brier=0.0, log_loss=0.0)
    empty_drive = LiveDriveOutcomeSlice(
        label=fit_label, n_predictions=0, multinomial_log_loss=0.0, per_class={}
    )
    empty_totals = LiveTotalsSlice(label=fit_label, n_predictions=0, mae=0.0, rmse=0.0, by_phase={})

    return LiveBacktestReport(
        games_replayed=run.games_replayed,
        games_skipped_no_pbp=run.games_skipped_no_pbp,
        win_prob=wp_fit or empty_wp,
        win_prob_stress=wp_stress,
        drive_outcomes=drive_fit or empty_drive,
        drive_outcomes_stress=drive_stress,
        totals=totals_fit or empty_totals,
        totals_stress=totals_stress,
    )
