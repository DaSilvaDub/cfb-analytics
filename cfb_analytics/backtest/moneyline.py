"""Orchestrates the ridge walk-forward backtest into one reportable result.

This is the plan's section 8 moneyline backtest. It reports the internal
ridge model's own calibration (Brier, log loss, confidence buckets,
reliability curve) alongside two comparisons:

* One of the plan's three required baselines -- Elo-only, using CFBD's own
  weekly Elo ratings (``backfill-elo``) run through the standard logistic
  Elo formula (``backtest/elo_baseline.py``).
* The plan's OWN internal Elo model (section 6.2; ``models/elo.py`` /
  ``features/elo_internal.py``) -- a genuinely different model from ridge
  (sequential rating updates, not a batch regression), reported here as a
  second comparison point, not one of the plan's three required baselines.

Both are scored on exactly the same games ridge was, so every comparison is
apples to apples: a game the ridge model could not predict never enters any
model's metrics (see ``harness.py``'s module docstring).

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

It also reports ``P_logit`` alone (``features/ensemble.py``'s IRLS logistic
regression on talent/returning-production/home-field/rest/advanced-stat-net
diffs -- see that module's docstring for the full feature list and what is
still a tracked gap), the same way it reports internal Elo alone: a genuine
comparison point, not one of the plan's three required baselines.

Finally, it reports the plan's own three-model ENSEMBLE (section 6.4):
``P_ridge`` (``Phi(M/sigma)``), ``P_elo`` (the internal Elo model, NOT the
CFBD baseline -- the baseline is a comparison point, never an ensemble
input), and ``P_logit``. The pooling weights ``v`` are fit once, by grid
search minimizing log loss on the non-stress fit predictions
(``backtest/ensemble_fit.py``), then applied to both the fit and stress
slices -- the same fit-then-apply shape ``sigma_0`` and the shrinkage/lambda
constants already use elsewhere in this module and ``models/``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from cfb_analytics.backtest.calibration import calibrate_sigma, margin_to_prob
from cfb_analytics.backtest.elo_baseline import elo_win_probability
from cfb_analytics.backtest.ensemble_fit import fit_ensemble_weights
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
from cfb_analytics.models.elo import DEFAULT_K as DEFAULT_ELO_K
from cfb_analytics.models.elo import HFA_ELO_POINTS
from cfb_analytics.models.ensemble import pool_probabilities
from cfb_analytics.models.logistic import DEFAULT_L2_LAMBDA as DEFAULT_LOGIT_L2_LAMBDA
from cfb_analytics.models.ridge import DEFAULT_MIN_GAMES, DEFAULT_RIDGE_LAMBDA
from cfb_analytics.models.shrinkage import DEFAULT_COEFFICIENTS, ShrinkageCoefficients

# Plan section 6.4's three ensemble members. Order matters only for display.
ENSEMBLE_MEMBER_NAMES = ("ridge", "internal_elo", "logit")

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
    internal_elo_seasons: SliceMetrics | None
    internal_elo_stress: SliceMetrics | None
    skipped_internal_elo_unrated: int
    logit_seasons: SliceMetrics | None
    logit_stress: SliceMetrics | None
    skipped_logit_unrated: int
    ensemble_weights: dict[str, float] | None
    ensemble_seasons: SliceMetrics | None
    ensemble_stress: SliceMetrics | None

    def as_text(self) -> str:
        lines = [
            "cfb-analytics moneyline backtest (internal ridge, walk-forward)",
            f"  sigma_0 (fitted point-spread stdev)     : {self.sigma_0:.2f}",
            f"  games used to fit sigma_0               : {self.n_games_calibrated}",
            f"  skipped (insufficient season history)   : {self.skipped_insufficient_history}",
            f"  skipped (a team had no rating)          : {self.skipped_unrated_team}",
            f"  skipped (no Elo baseline available)     : {self.skipped_elo_unrated}",
            f"  skipped (no internal Elo rating)        : {self.skipped_internal_elo_unrated}",
            f"  skipped (no P_logit prediction)         : {self.skipped_logit_unrated}",
            "",
            "== internal ridge ==",
            _slice_text(self.seasons),
        ]
        if self.stress is not None:
            lines += ["", _slice_text(self.stress)]
        lines += ["", "== Elo-only baseline (plan section 8, baseline #3) =="]
        if self.elo_seasons is not None:
            lines.append(_slice_text(self.elo_seasons))
            lines.append(
                f"    {_comparison_line('ridge', self.seasons, 'Elo-only', self.elo_seasons)}"
            )
        else:
            lines.append("  (no games had an Elo rating for both teams)")
        if self.elo_stress is not None:
            lines += ["", _slice_text(self.elo_stress)]
        lines += ["", "== internal Elo (plan section 6.2) =="]
        if self.internal_elo_seasons is not None:
            lines.append(_slice_text(self.internal_elo_seasons))
            comparison = _comparison_line(
                "ridge", self.seasons, "internal Elo", self.internal_elo_seasons
            )
            lines.append(f"    {comparison}")
        else:
            lines.append("  (no games had an internal Elo rating for both teams)")
        if self.internal_elo_stress is not None:
            lines += ["", _slice_text(self.internal_elo_stress)]
        lines += ["", "== P_logit (plan section 6.4) =="]
        if self.logit_seasons is not None:
            lines.append(_slice_text(self.logit_seasons))
            comparison = _comparison_line(
                "ridge", self.seasons, "P_logit", self.logit_seasons
            )
            lines.append(f"    {comparison}")
        else:
            lines.append("  (no games had a P_logit prediction available)")
        if self.logit_stress is not None:
            lines += ["", _slice_text(self.logit_stress)]
        lines += ["", "== three-model ensemble (plan section 6.4) =="]
        if self.ensemble_weights is not None:
            weight_text = ", ".join(
                f"{name}={weight:.2f}" for name, weight in self.ensemble_weights.items()
            )
            lines.append(f"  fitted weights: {weight_text}")
        if self.ensemble_seasons is not None:
            lines.append(_slice_text(self.ensemble_seasons))
            comparison = _comparison_line(
                "ensemble", self.ensemble_seasons, "ridge", self.seasons
            )
            lines.append(f"    {comparison}")
        else:
            lines.append("  (no games had any ensemble member's prediction available)")
        if self.ensemble_stress is not None:
            lines += ["", _slice_text(self.ensemble_stress)]
        lines += [
            "",
            "  NOTE: market and SP+-only baselines are still not computable (no leakage-safe",
            "  historical series in this store -- see moneyline.py module docstring). This is",
            "  a calibration and model-comparison check, not a full promotion decision.",
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


def _comparison_line(
    left_label: str, left: SliceMetrics, right_label: str, right: SliceMetrics
) -> str:
    """A factual log-loss comparison, not a promotion verdict.

    Only meaningful when both slices cover the same games, which is true
    for the main (non-stress) slice by construction (see harness.py) --
    NOT necessarily true if a caller ever compared across different season
    ranges, so this stays a private helper rather than a public API.
    """
    verb = "beats" if left.log_loss < right.log_loss else "does not beat"
    return (
        f"{left_label} {verb} {right_label} on log loss "
        f"({left.log_loss:.4f} vs {right.log_loss:.4f})"
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


def _internal_elo_probs(predictions: list[GamePrediction]) -> tuple[list[Prediction], int]:
    """Internal-Elo probabilities for the games where the model actually
    produced one (already computed by harness.py -- no margin-to-prob
    conversion needed here, Elo's own output is already a probability)."""
    probs: list[Prediction] = []
    skipped = 0
    for prediction in predictions:
        if prediction.internal_elo_win_prob is None:
            skipped += 1
            continue
        probs.append((prediction.internal_elo_win_prob, prediction.home_won))
    return probs, skipped


def _logit_probs(predictions: list[GamePrediction]) -> tuple[list[Prediction], int]:
    """P_logit-alone probabilities for the games where it actually produced
    one (already computed by harness.py -- see min_n's own insufficient-data
    status on ``models/logistic.py``'s ``LogisticFit``)."""
    probs: list[Prediction] = []
    skipped = 0
    for prediction in predictions:
        if prediction.logit_win_prob is None:
            skipped += 1
            continue
        probs.append((prediction.logit_win_prob, prediction.home_won))
    return probs, skipped


def _member_probs(prediction: GamePrediction, sigma_0: float) -> dict[str, float]:
    """Every ensemble member's probability for one game, omitting whichever
    are unavailable (internal Elo's own insufficient-history weeks, P_logit
    before it has cleared its own min_n) rather than fabricating one --
    ``pool_probabilities`` renormalizes among whatever is actually present.
    Ridge is always present: once a ``GamePrediction`` exists at all, its
    margin (and so ``P_ridge``) is always defined.
    """
    probs = {"ridge": margin_to_prob(prediction.predicted_margin, sigma_0)}
    if prediction.internal_elo_win_prob is not None:
        probs["internal_elo"] = prediction.internal_elo_win_prob
    if prediction.logit_win_prob is not None:
        probs["logit"] = prediction.logit_win_prob
    return probs


def _ensemble_predictions(
    predictions: list[GamePrediction], sigma_0: float, weights: dict[str, float]
) -> list[Prediction]:
    result: list[Prediction] = []
    for prediction in predictions:
        member_probs = _member_probs(prediction, sigma_0)
        available = {name: w for name, w in weights.items() if name in member_probs}
        if not available or sum(available.values()) <= 0:
            continue
        result.append((pool_probabilities(member_probs, available), prediction.home_won))
    return result


def run_moneyline_backtest(
    conn: sqlite3.Connection,
    seasons: tuple[int, ...] = DEFAULT_SEASONS,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
    apply_shrinkage: bool = True,
    coeffs: ShrinkageCoefficients = DEFAULT_COEFFICIENTS,
    elo_k: float = DEFAULT_ELO_K,
    elo_hfa: float = HFA_ELO_POINTS,
    logit_l2_lambda: float = DEFAULT_LOGIT_L2_LAMBDA,
    logit_min_n: int = 30,
    ensemble_grid_step: float = 0.05,
) -> MoneylineBacktestReport:
    run = run_walk_forward(
        conn, list(seasons), ridge_lambda=ridge_lambda, min_games=min_games,
        apply_shrinkage=apply_shrinkage, coeffs=coeffs, elo_k=elo_k, elo_hfa=elo_hfa,
        logit_l2_lambda=logit_l2_lambda, logit_min_n=logit_min_n,
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

    internal_elo_fit_probs, internal_elo_fit_skipped = _internal_elo_probs(fit_predictions)
    internal_elo_stress_probs, internal_elo_stress_skipped = _internal_elo_probs(
        stress_predictions
    )
    internal_elo_seasons_metrics = (
        _score_slice("internal Elo, same seasons/games as ridge", internal_elo_fit_probs)
        if internal_elo_fit_probs
        else None
    )
    internal_elo_stress_metrics = (
        _score_slice("internal Elo, 2020 stress slice", internal_elo_stress_probs)
        if internal_elo_stress_probs
        else None
    )

    logit_fit_probs, logit_fit_skipped = _logit_probs(fit_predictions)
    logit_stress_probs, logit_stress_skipped = _logit_probs(stress_predictions)
    logit_seasons_metrics = (
        _score_slice("P_logit, same seasons/games as ridge", logit_fit_probs)
        if logit_fit_probs
        else None
    )
    logit_stress_metrics = (
        _score_slice("P_logit, 2020 stress slice", logit_stress_probs)
        if logit_stress_probs
        else None
    )

    ensemble_fit_member_probs = [_member_probs(p, sigma_0) for p in fit_predictions]
    ensemble_fit_outcomes = [p.home_won for p in fit_predictions]
    ensemble_weights = (
        fit_ensemble_weights(
            ensemble_fit_member_probs, ensemble_fit_outcomes, ENSEMBLE_MEMBER_NAMES,
            grid_step=ensemble_grid_step,
        )
        if ensemble_fit_member_probs
        else None
    )
    ensemble_seasons_metrics = None
    ensemble_stress_metrics = None
    if ensemble_weights is not None:
        ensemble_fit_probs = _ensemble_predictions(fit_predictions, sigma_0, ensemble_weights)
        if ensemble_fit_probs:
            ensemble_seasons_metrics = _score_slice(
                "ensemble, same seasons/games as ridge", ensemble_fit_probs
            )
        ensemble_stress_probs = _ensemble_predictions(
            stress_predictions, sigma_0, ensemble_weights
        )
        if ensemble_stress_probs:
            ensemble_stress_metrics = _score_slice(
                "ensemble, 2020 stress slice", ensemble_stress_probs
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
        internal_elo_seasons=internal_elo_seasons_metrics,
        internal_elo_stress=internal_elo_stress_metrics,
        skipped_internal_elo_unrated=internal_elo_fit_skipped + internal_elo_stress_skipped,
        logit_seasons=logit_seasons_metrics,
        logit_stress=logit_stress_metrics,
        skipped_logit_unrated=logit_fit_skipped + logit_stress_skipped,
        ensemble_weights=ensemble_weights,
        ensemble_seasons=ensemble_seasons_metrics,
        ensemble_stress=ensemble_stress_metrics,
    )
