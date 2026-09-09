"""Structured report for the live micro-markets backtest."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveWinProbSlice:
    label: str
    n_predictions: int
    brier: float
    log_loss: float


@dataclass(frozen=True)
class LiveDriveOutcomeSlice:
    label: str
    n_predictions: int
    multinomial_log_loss: float
    per_class: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LiveTotalsSlice:
    label: str
    n_predictions: int
    mae: float
    rmse: float
    by_phase: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LiveBacktestReport:
    games_replayed: int
    games_skipped_no_pbp: int
    win_prob: LiveWinProbSlice
    win_prob_stress: LiveWinProbSlice | None
    drive_outcomes: LiveDriveOutcomeSlice
    drive_outcomes_stress: LiveDriveOutcomeSlice | None
    totals: LiveTotalsSlice
    totals_stress: LiveTotalsSlice | None
    model_status: str = "uncalibrated_shadow"

    def as_text(self) -> str:
        lines = [
            "cfb-analytics live micro-markets backtest (walk-forward)",
            f"  model status                      : {self.model_status}",
            f"  games replayed                    : {self.games_replayed}",
            f"  games skipped (no PBP)            : {self.games_skipped_no_pbp}",
            "",
            "== win probability ==",
            _wp_text(self.win_prob),
        ]
        if self.win_prob_stress:
            lines += ["", _wp_text(self.win_prob_stress)]
        lines += [
            "",
            "== drive outcomes ==",
            _drive_text(self.drive_outcomes),
        ]
        if self.drive_outcomes_stress:
            lines += ["", _drive_text(self.drive_outcomes_stress)]
        lines += [
            "",
            "== team totals ==",
            _totals_text(self.totals),
        ]
        if self.totals_stress:
            lines += ["", _totals_text(self.totals_stress)]
        lines += [
            "",
            "  NOTE: This is an uncalibrated shadow model. Results are for research",
            "  calibration only, not actionable recommendations.",
        ]
        return "\n".join(lines)


def _wp_text(s: LiveWinProbSlice) -> str:
    return f"  [{s.label}] n={s.n_predictions}  brier={s.brier:.4f}  log_loss={s.log_loss:.4f}"


def _drive_text(s: LiveDriveOutcomeSlice) -> str:
    lines = [
        f"  [{s.label}] n={s.n_predictions}  multinomial_log_loss={s.multinomial_log_loss:.4f}",
    ]
    for cls, metrics in sorted(s.per_class.items()):
        lines.append(
            f"    {cls:>10}  precision={metrics['precision']:.3f}  "
            f"recall={metrics['recall']:.3f}  "
            f"n_true={int(metrics['true_count'])}  n_pred={int(metrics['pred_count'])}"
        )
    return "\n".join(lines)


def _totals_text(s: LiveTotalsSlice) -> str:
    lines = [
        f"  [{s.label}] n={s.n_predictions}  mae={s.mae:.2f}  rmse={s.rmse:.2f}",
    ]
    for phase, metrics in sorted(s.by_phase.items()):
        n = int(metrics.get("n", 0))
        if n > 0:
            lines.append(
                f"    {phase:>6}  mae={metrics['mae']:.2f}  rmse={metrics['rmse']:.2f}  n={n}"
            )
    return "\n".join(lines)
