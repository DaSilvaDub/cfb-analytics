"""Orchestrates the ridge walk-forward backtest into one reportable result.

This is the plan's section 8 moneyline backtest, with an honest limitation
stated up front: the plan requires beating three baselines out-of-sample
before promotion (vig-free market, SP+-only, Elo-only). None of the three
are computable leakage-safely against this store's *historical* seasons
right now --

* Market: ``odds_snapshots`` only holds the current season's live capture
  (daily ingest started 2026); there is no historical market to compare
  against 2014-2025 outcomes.
* SP+-only / Elo-only: ``team_ratings`` only has ``season_final`` snapshots
  for CFBD's SP+/SRS/Elo (see the backfill in ``ingest/cfbd_fundamentals``).
  Using a season-final rating to predict week 3 of that same season is
  exactly the classic leak ``AsOfReader`` exists to catch -- there is no
  *weekly* snapshot of these ratings stored to compare against instead.

So this backtest reports the ridge model's own calibration quality in
isolation. It is real, useful evidence (is the model's stated confidence
trustworthy at all?), but it is NOT a promotion decision by itself --
``config/promotion.json``'s baseline-beating gates stay unsatisfiable until
either historical odds or weekly SP+/Elo snapshots are backfilled. That is a
tracked gap, not an oversight.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from cfb_analytics.backtest.calibration import calibrate_sigma, margin_to_prob
from cfb_analytics.backtest.harness import GamePrediction, run_walk_forward
from cfb_analytics.backtest.metrics import (
    BucketResult,
    Prediction,
    brier_score,
    bucketed_win_rate,
    favorite_side,
    log_loss,
    reliability_curve,
)
from cfb_analytics.models.ridge import DEFAULT_MIN_GAMES, DEFAULT_RIDGE_LAMBDA

# Excluded from the sigma_0 fit (COVID-disrupted, partial/irregular schedules
# per plan section 8) but still predicted and reported, as a stress slice.
STRESS_SEASONS = frozenset({2020})

DEFAULT_SEASONS = tuple(range(2014, 2026))


@dataclass(frozen=True)
class SliceMetrics:
    label: str
    n_games: int
    brier: float
    log_loss: float
    confidence_buckets: list[BucketResult]
    reliability: list[BucketResult]


@dataclass(frozen=True)
class MoneylineBacktestReport:
    sigma_0: float
    n_games_calibrated: int
    seasons: SliceMetrics
    stress: SliceMetrics | None
    skipped_insufficient_history: int
    skipped_unrated_team: int

    def as_text(self) -> str:
        lines = [
            "cfb-analytics moneyline backtest (internal ridge, walk-forward)",
            f"  sigma_0 (fitted point-spread stdev)  : {self.sigma_0:.2f}",
            f"  games used to fit sigma_0            : {self.n_games_calibrated}",
            f"  skipped (insufficient season history): {self.skipped_insufficient_history}",
            f"  skipped (a team had no rating)       : {self.skipped_unrated_team}",
            "",
            _slice_text(self.seasons),
        ]
        if self.stress is not None:
            lines += ["", _slice_text(self.stress)]
        lines += [
            "",
            "  NOTE: no baseline comparison yet (market/SP+/Elo all lack a leakage-safe",
            "  historical series in this store -- see moneyline.py module docstring).",
            "  This is a calibration check, not a promotion decision.",
        ]
        return "\n".join(lines)


def _slice_text(metrics: SliceMetrics) -> str:
    lines = [
        f"  [{metrics.label}] n={metrics.n_games}  "
        f"brier={metrics.brier:.4f}  log_loss={metrics.log_loss:.4f}",
        "    confidence buckets (favored side only):",
    ]
    for bucket in metrics.confidence_buckets:
        lines.append(f"      {_bucket_line(bucket)}")
    lines.append("    reliability curve (raw home-win probability):")
    for bucket in metrics.reliability:
        lines.append(f"      {_bucket_line(bucket)}")
    return "\n".join(lines)


def _bucket_line(bucket: BucketResult) -> str:
    if bucket.n == 0:
        return f"{bucket.label:>10}  n=0"
    assert bucket.win_rate is not None and bucket.wilson_low is not None
    return (
        f"{bucket.label:>10}  n={bucket.n:<5} "
        f"win_rate={bucket.win_rate:.3f}  "
        f"95% CI=[{bucket.wilson_low:.3f}, {bucket.wilson_high:.3f}]"
    )


def _score_slice(label: str, predictions: list[Prediction]) -> SliceMetrics:
    favored = [favorite_side(p, won) for p, won in predictions]
    return SliceMetrics(
        label=label,
        n_games=len(predictions),
        brier=brier_score(predictions),
        log_loss=log_loss(predictions),
        confidence_buckets=bucketed_win_rate(favored),
        reliability=reliability_curve(predictions),
    )


def run_moneyline_backtest(
    conn: sqlite3.Connection,
    seasons: tuple[int, ...] = DEFAULT_SEASONS,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
) -> MoneylineBacktestReport:
    run = run_walk_forward(conn, list(seasons), ridge_lambda=ridge_lambda, min_games=min_games)

    fit_predictions = [p for p in run.predictions if p.season not in STRESS_SEASONS]
    stress_predictions = [p for p in run.predictions if p.season in STRESS_SEASONS]

    sigma_0 = calibrate_sigma([p.actual_margin - p.predicted_margin for p in fit_predictions])

    def to_probs(preds: list[GamePrediction]) -> list[Prediction]:
        return [(margin_to_prob(p.predicted_margin, sigma_0), p.home_won) for p in preds]

    fit_probs = to_probs(fit_predictions)
    stress_metrics = (
        _score_slice("2020 stress slice", to_probs(stress_predictions))
        if stress_predictions
        else None
    )
    fit_seasons = [s for s in seasons if s not in STRESS_SEASONS]

    return MoneylineBacktestReport(
        sigma_0=sigma_0,
        n_games_calibrated=len(fit_predictions),
        seasons=_score_slice(f"{min(fit_seasons)}-{max(fit_seasons)} (excl. 2020)", fit_probs),
        stress=stress_metrics,
        skipped_insufficient_history=run.skipped_insufficient_history,
        skipped_unrated_team=run.skipped_unrated_team,
    )
