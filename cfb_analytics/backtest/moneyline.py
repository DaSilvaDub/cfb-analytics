"""Orchestrates the ridge walk-forward backtest into one reportable result.

This is the plan's section 8 moneyline backtest. It reports the internal
ridge model's own calibration (Brier, log loss, confidence buckets,
reliability curve) alongside one of the plan's three required baselines --
Elo-only, using CFBD's own weekly Elo ratings (``backfill-elo``) run
through the standard logistic Elo formula (``backtest/elo_baseline.py``).
Elo is scored on exactly the same games ridge was, so the comparison is
apples to apples: a game the ridge model could not predict never enters
either model's metrics (see ``harness.py``'s module docstring).

The other two required baselines are still not computable leakage-safely
against this store's historical seasons:

* Market: ``odds_snapshots`` only holds the current season's live capture
  (daily ingest started 2026); there is no historical market to compare
  against 2014-2025 outcomes.
* SP+-only: ``team_ratings`` only has ``season_final`` SP+ snapshots (see
  ``ingest/cfbd_fundamentals``) -- CFBD's own ``/ratings/sp`` endpoint was
  live-tested and found to silently ignore its own ``week`` parameter for
  historical seasons, always returning the season-final number. There is no
  way to reconstruct a genuine weekly SP+ history from this API at all, so
  this gap (unlike Elo's, which was a fixable bug) is not closable by
  backfilling harder.

So this backtest is real, useful evidence -- is the model's stated
confidence trustworthy, and does it actually beat a simple Elo baseline --
but it is NOT a full promotion decision by itself: ``config/promotion.json``
also requires beating a market and an SP+ baseline, neither of which exist
yet. That is a tracked gap, not an oversight.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from cfb_analytics.backtest.calibration import calibrate_sigma, margin_to_prob
from cfb_analytics.backtest.elo_baseline import elo_win_probability
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
from cfb_analytics.models.shrinkage import DEFAULT_COEFFICIENTS, ShrinkageCoefficients

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
    elo_seasons: SliceMetrics | None
    elo_stress: SliceMetrics | None
    skipped_elo_unrated: int

    def as_text(self) -> str:
        lines = [
            "cfb-analytics moneyline backtest (internal ridge, walk-forward)",
            f"  sigma_0 (fitted point-spread stdev)  : {self.sigma_0:.2f}",
            f"  games used to fit sigma_0            : {self.n_games_calibrated}",
            f"  skipped (insufficient season history): {self.skipped_insufficient_history}",
            f"  skipped (a team had no rating)       : {self.skipped_unrated_team}",
            f"  skipped (no Elo baseline available)  : {self.skipped_elo_unrated}",
            "",
            "== internal ridge ==",
            _slice_text(self.seasons),
        ]
        if self.stress is not None:
            lines += ["", _slice_text(self.stress)]
        lines += ["", "== Elo-only baseline (plan section 8, baseline #3) =="]
        if self.elo_seasons is not None:
            lines.append(_slice_text(self.elo_seasons))
            lines.append(f"    {_comparison_line(self.seasons, self.elo_seasons)}")
        else:
            lines.append("  (no games had an Elo rating for both teams)")
        if self.elo_stress is not None:
            lines += ["", _slice_text(self.elo_stress)]
        lines += [
            "",
            "  NOTE: market and SP+-only baselines are still not computable (no leakage-safe",
            "  historical series in this store -- see moneyline.py module docstring). This is",
            "  a calibration and Elo-comparison check, not a full promotion decision.",
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


def _comparison_line(ridge: SliceMetrics, elo: SliceMetrics) -> str:
    """A factual log-loss comparison, not a promotion verdict.

    Only meaningful when both slices cover the same games, which is true
    for the main (non-stress) slice by construction (see harness.py) --
    NOT necessarily true if a caller ever compared across different season
    ranges, so this stays a private helper rather than a public API.
    """
    verb = "beats" if ridge.log_loss < elo.log_loss else "does not beat"
    return (
        f"ridge {verb} the Elo-only baseline on log loss "
        f"({ridge.log_loss:.4f} vs {elo.log_loss:.4f})"
    )


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


def _elo_probs(predictions: list[GamePrediction]) -> tuple[list[Prediction], int]:
    """Elo-only probabilities for the games where CFBD's backfill actually
    resolved a weekly Elo rating for both teams (FCS opponents are the
    usual reason it did not -- see ``backfill_elo``'s docstring)."""
    probs: list[Prediction] = []
    skipped = 0
    for prediction in predictions:
        if prediction.elo_home_rating is None or prediction.elo_away_rating is None:
            skipped += 1
            continue
        prob = elo_win_probability(
            prediction.elo_home_rating,
            prediction.elo_away_rating,
            neutral_site=prediction.neutral_site,
        )
        probs.append((prob, prediction.home_won))
    return probs, skipped


def run_moneyline_backtest(
    conn: sqlite3.Connection,
    seasons: tuple[int, ...] = DEFAULT_SEASONS,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
    apply_shrinkage: bool = True,
    coeffs: ShrinkageCoefficients = DEFAULT_COEFFICIENTS,
) -> MoneylineBacktestReport:
    run = run_walk_forward(
        conn, list(seasons), ridge_lambda=ridge_lambda, min_games=min_games,
        apply_shrinkage=apply_shrinkage, coeffs=coeffs,
    )

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

    elo_fit_probs, elo_fit_skipped = _elo_probs(fit_predictions)
    elo_stress_probs, elo_stress_skipped = _elo_probs(stress_predictions)
    elo_seasons_metrics = (
        _score_slice("Elo, same seasons/games as ridge", elo_fit_probs) if elo_fit_probs else None
    )
    elo_stress_metrics = (
        _score_slice("Elo, 2020 stress slice", elo_stress_probs) if elo_stress_probs else None
    )

    return MoneylineBacktestReport(
        sigma_0=sigma_0,
        n_games_calibrated=len(fit_predictions),
        seasons=_score_slice(f"{min(fit_seasons)}-{max(fit_seasons)} (excl. 2020)", fit_probs),
        stress=stress_metrics,
        skipped_insufficient_history=run.skipped_insufficient_history,
        skipped_unrated_team=run.skipped_unrated_team,
        elo_seasons=elo_seasons_metrics,
        elo_stress=elo_stress_metrics,
        skipped_elo_unrated=elo_fit_skipped + elo_stress_skipped,
    )
