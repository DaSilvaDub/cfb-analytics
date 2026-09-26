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

Market close consensus becomes available once
``ingest/cfbd_lines_historical`` has dual-stamped a season into
``odds_snapshots`` (``source='cfbd_historical'``) and ``build_market_for_slate``
has rebuilt ``market_consensus``. When close ML consensus exists for the
same games the walk-forward scores, this module reports a market Brier /
log-loss slice **and** a same-game-set compare (ensemble / ridge / Elo vs
market on the identical overlap) plus model-vs-close median CLV for
``--promote`` evidence; when coverage is thin or absent it prints an
explicit ``market: N/A (coverage)`` rather than silently skipping. Live
``source='cfbd'`` rows are never mixed into historical baseline scoring.

SP+-only remains not computable leakage-safely: ``team_ratings`` only has
``season_final`` SP+ snapshots (see ``ingest/cfbd_fundamentals``) -- CFBD's
own ``/ratings/sp`` endpoint was live-tested and found to silently ignore
its own ``week`` parameter for historical seasons. That gap is not closable
by backfilling harder.

So this backtest is real, useful evidence -- is the model's stated
confidence trustworthy, and does it beat Elo / (when present) market --
but it is NOT a full promotion decision by itself until market covers
``min_seasons_backtested`` and SP+ is resolved. Promotion status stays
``shadow`` until then.

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

Ensemble probabilities then pass through a walk-forward probability
calibrator (``backtest/prob_calibrate.py``: default Platt on logit(p),
season-blocked). ``--promote`` scores gap + beat_market on the calibrated
ensemble; raw ensemble metrics and median CLV stay on the uncalibrated blend.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from statistics import median

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
from cfb_analytics.backtest.prob_calibrate import (
    Fold,
    Method,
    TimedProb,
    walk_forward_calibrate,
)
from cfb_analytics.backtest.market_blend_fit import (
    BlendRow,
    score_fixed_weight,
    walk_forward_market_blend,
)
from cfb_analytics.models.market_blend import (
    DEFAULT_MARKET_WEIGHT_FLOOR,
    market_weight_floor_from_settings,
)
from cfb_analytics.features.open_market_prior import (
    OPEN_MARKET_BLEND_MAX_WEEK,
    OPEN_MARKET_BLEND_WEIGHT,
    OPEN_MARKET_PRIOR_MAX_WEEK,
    OPEN_MARKET_PRIOR_WEIGHT,
    load_open_home_spreads,
    open_prior_weight_for_week,
    open_spread_to_home_prob,
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

# Early-season ridge confidence deflation (spike 2026-09-26, weeks 1–4 path).
# Global sigma_0 is fit on full-season residuals; early-week ridge margins are
# noisier (tiny per-team n after the week-1 blackout clears at min_games=30),
# so Phi(M/sigma) is overconfident. Inflating sigma for weeks 1–4 on the ridge
# member only cut weeks-1–4 ridge logloss ~0.67→0.57 and ensemble gap vs
# market ~0.0538→0.0501 on the 2023–2025 same-game set without moving weeks
# 5+. See docs/scorecards/moneyline_scorecard_2023_2025_early_season.md.
EARLY_SEASON_MAX_WEEK = 4
EARLY_SEASON_RIDGE_SIGMA_SCALE = 1.5

# Opening-line market prior (spike 2026-09-26). Uses CFBD spreadOpen @
# kickoff-7d only — never close. Weeks ≤ OPEN_MARKET_PRIOR_MAX_WEEK replace
# ensemble raw P(home) with open-implied Phi(-spread/sigma) (weight=1.0).
# Weeks prior_max < week ≤ OPEN_MARKET_BLEND_MAX_WEEK soft-blend with
# OPEN_MARKET_BLEND_WEIGHT < 1 (default 0 until W3–4 spike ships a winner).
# Week-1 games skipped by the ridge min_games blackout are unlocked as
# open-only synthetic predictions so promote overlap covers week 1 without
# bare Elo/ridge priors (those lost to market by ~0.13 LL). See
# docs/scorecards/moneyline_scorecard_2023_2025_market_prior_early.md and
# docs/scorecards/moneyline_scorecard_2023_2025_open_blend_w34.md.
# Note: open≈close, so early rows partially borrow market info vs the close
# baseline; beat_market still fails full-season and promotion stays shadow.


@dataclass(frozen=True)
class SliceMetrics:
    label: str
    n_games: int
    brier: float
    log_loss: float
    confidence_buckets: list[BucketResult]
    reliability: list[BucketResult]


@dataclass(frozen=True)
class SameGameSetEvidence:
    """Apples-to-apples market overlap metrics for promotion gates.

    Every model metric here is scored on the **identical** game set: games
    where the walk-forward produced a prediction **and** vig-free close
    P(home) exists in ``market_consensus``. Comparing full-n ridge logloss
    to market logloss on a covered subset is dishonest; this struct exists
    so ``require_oos_logloss_beat_market`` cannot make that mistake.

    ``beat_market`` uses the **ensemble** as the promotion candidate
    (plan section 6.4 / ``promotion.json`` evidence), not ridge alone.

    Median CLV is **model-vs-close** probability CLV on the *raw* ensemble
    (open ML is typically N/A from CFBD). See ``CLV_FORMULA``. Walk-forward
    probability calibration is applied to ``ensemble`` for gap + beat_market
    only; ``ensemble_raw`` and CLV keep the uncalibrated blend.
    """

    CLV_FORMULA = (
        "model-vs-close probability CLV on the side the model favors: "
        "if p_model>=0.5 (favors home): clv = p_close - p_model; "
        "else (favors away): clv = p_model - p_close "
        "(equivalently (1-p_close)-(1-p_model)). "
        "Units: probability points (0.01 = 1pp); bps = 10000 * points. "
        "Positive => close assigned more probability to the model's "
        "favored side than the model did (classic 'better number than close'). "
        "Open-ML book CLV is not used: CFBD historical open ML is typically N/A."
    )

    n_full: int
    n_overlap: int
    n_skipped: int
    seasons_covered: tuple[int, ...]
    market_min_books: int
    promotion_candidate: str
    market: SliceMetrics | None
    ensemble: SliceMetrics | None
    ensemble_raw: SliceMetrics | None
    ridge: SliceMetrics | None
    elo: SliceMetrics | None
    beat_market: bool | None
    median_clv: float | None
    median_clv_bps: float | None
    median_clv_non_negative: bool | None
    calibration_method: str | None = None
    clv_formula: str = CLV_FORMULA
    # Market blend (settings.blend.market_weight_floor) — evidence only.
    # beat_market stays on pure calibrated ensemble (skill to lower the floor).
    market_weight_floor: float | None = None
    blend_weight_by_season: dict[int, float] | None = None
    blend_fitted: SliceMetrics | None = None  # walk-forward w in [floor, 1]
    blend_at_floor: SliceMetrics | None = None  # fixed w=floor
    blend_pure_model: SliceMetrics | None = None  # w=0 bracket (ignores floor)
    blend_pure_market: SliceMetrics | None = None  # w=1 bracket
    blend_non_worse_than_market: bool | None = None
    blend_gate_interpretation: str = (
        "pure_model_must_beat_market_before_lowering_floor"
    )


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
    ensemble_seasons_raw: SliceMetrics | None = None
    ensemble_calibration_method: str | None = None
    market_seasons: SliceMetrics | None = None
    market_note: str | None = None
    skipped_market_unpriced: int = 0
    same_game_set: SameGameSetEvidence | None = None

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
        if self.ensemble_calibration_method is not None:
            lines.append(
                f"  walk-forward calibrator          : {self.ensemble_calibration_method}"
            )
        if self.ensemble_seasons is not None:
            lines.append(_slice_text(self.ensemble_seasons))
            comparison = _comparison_line(
                "ensemble", self.ensemble_seasons, "ridge", self.seasons
            )
            lines.append(f"    {comparison}")
        else:
            lines.append("  (no games had any ensemble member's prediction available)")
        if self.ensemble_seasons_raw is not None:
            lines.append(_slice_text(self.ensemble_seasons_raw))
        if self.ensemble_stress is not None:
            lines += ["", _slice_text(self.ensemble_stress)]
        lines += ["", "== market close baseline (plan section 8, baseline #1) =="]
        if self.market_seasons is not None:
            lines.append(_slice_text(self.market_seasons))
            lines.append(
                f"    skipped (no close ML consensus): {self.skipped_market_unpriced}"
            )
            lines.append(
                "    NOTE: headline market n is the covered subset; see same-game-set"
            )
            lines.append(
                "    section below for honest ensemble/ridge/Elo vs market logloss."
            )
        else:
            note = self.market_note or "market: N/A (coverage)"
            lines.append(f"  {note}")
        lines += ["", "== same-game-set market compare (promotion evidence) =="]
        if self.same_game_set is not None:
            lines.extend(_same_game_set_text(self.same_game_set))
        else:
            lines.append("  same-game-set: N/A (no market overlap)")
        lines += [
            "",
            "  NOTE: SP+-only baseline is still not computable (CFBD /ratings/sp has no",
            "  weekly historical series -- see moneyline.py module docstring). Market is",
            "  scored when cfbd_historical close consensus exists; otherwise the report",
            "  states market: N/A (coverage) explicitly. This is not a full promotion decision.",
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


def ridge_sigma_for_week(sigma_0: float, week: int) -> float:
    """Effective sigma for converting a ridge margin in ``week``.

    Weeks 1–4 use ``EARLY_SEASON_RIDGE_SIGMA_SCALE``; later weeks use
    ``sigma_0`` unchanged. Leakage-safe (week index is known at prediction
    time; no future outcomes).
    """
    if sigma_0 <= 0:
        raise ValueError(f"sigma_0 must be positive, got {sigma_0!r}")
    if week <= EARLY_SEASON_MAX_WEEK:
        return sigma_0 * EARLY_SEASON_RIDGE_SIGMA_SCALE
    return sigma_0


def _ridge_prob(prediction: GamePrediction, sigma_0: float) -> float:
    return margin_to_prob(
        prediction.predicted_margin,
        ridge_sigma_for_week(sigma_0, prediction.week),
    )


def _member_probs(prediction: GamePrediction, sigma_0: float) -> dict[str, float]:
    """Every ensemble member's probability for one game, omitting whichever
    are unavailable (internal Elo's own insufficient-history weeks, P_logit
    before it has cleared its own min_n) rather than fabricating one --
    ``pool_probabilities`` renormalizes among whatever is actually present.
    Ridge is always present: once a ``GamePrediction`` exists at all, its
    margin (and so ``P_ridge``) is always defined. Early-season ridge uses
    an inflated sigma (see ``ridge_sigma_for_week``).
    """
    probs = {"ridge": _ridge_prob(prediction, sigma_0)}
    if prediction.internal_elo_win_prob is not None:
        probs["internal_elo"] = prediction.internal_elo_win_prob
    if prediction.logit_win_prob is not None:
        probs["logit"] = prediction.logit_win_prob
    return probs


def _blend_open_prior(p_model: float, p_open: float, weight: float) -> float:
    """Log-odds blend of model toward open-implied market prior."""
    w = max(0.0, min(1.0, float(weight)))
    if w <= 0.0:
        return p_model
    if w >= 1.0:
        return p_open
    return pool_probabilities(
        {"model": p_model, "open_mkt": p_open},
        {"model": 1.0 - w, "open_mkt": w},
    )


def _ensemble_raw_prob(
    prediction: GamePrediction,
    sigma_0: float,
    weights: dict[str, float],
    open_spreads: dict[str, float] | None = None,
    *,
    open_prior_max_week: int = OPEN_MARKET_PRIOR_MAX_WEEK,
    open_prior_weight: float = OPEN_MARKET_PRIOR_WEIGHT,
    open_blend_max_week: int = OPEN_MARKET_BLEND_MAX_WEEK,
    open_blend_weight: float = OPEN_MARKET_BLEND_WEIGHT,
) -> float | None:
    """Ensemble P(home), with optional early-season open-spread prior.

    Weeks ≤ ``open_prior_max_week`` use ``open_prior_weight`` (production:
    replace). Weeks through ``open_blend_max_week`` use ``open_blend_weight``
    (soft blend, typically weeks 3–4). See ``open_prior_weight_for_week``.
    """
    member_probs = _member_probs(prediction, sigma_0)
    available = {name: w for name, w in weights.items() if name in member_probs}
    if not available or sum(available.values()) <= 0:
        return None
    p = pool_probabilities(member_probs, available)
    if not open_spreads:
        return p
    w_open = open_prior_weight_for_week(
        prediction.week,
        prior_max_week=open_prior_max_week,
        prior_weight=open_prior_weight,
        blend_max_week=open_blend_max_week,
        blend_weight=open_blend_weight,
    )
    if w_open > 0:
        spread = open_spreads.get(prediction.game_id)
        if spread is not None:
            p_open = open_spread_to_home_prob(spread, sigma_0)
            p = _blend_open_prior(p, p_open, w_open)
    return p


def _week1_open_prior_predictions(
    conn: sqlite3.Connection,
    seasons: list[int] | tuple[int, ...],
    *,
    sigma_0: float,
    existing_ids: set[str],
) -> list[GamePrediction]:
    """Synthesize week-1 predictions from open spread when ridge blacked out.

    ``predicted_margin = -open_spread`` and ``internal_elo_win_prob`` =
    open-implied P(home) so the ensemble (and open-prior replace) both land
    on the opening line. Logit left absent (insufficient history).
    """
    if not seasons:
        return []
    placeholders = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"""
        SELECT game_id, season, week, home_team_id, away_team_id,
               neutral_site, home_points, away_points
        FROM games
        WHERE season IN ({placeholders})
          AND season_type = 'regular'
          AND week = 1
          AND completed = 1
          AND home_points IS NOT NULL AND away_points IS NOT NULL
        """,
        tuple(seasons),
    ).fetchall()
    candidates = []
    for row in rows:
        gid = str(row["game_id"] if isinstance(row, sqlite3.Row) else row[0])
        if gid in existing_ids:
            continue
        candidates.append(row)
    if not candidates:
        return []
    ids = [
        str(r["game_id"] if isinstance(r, sqlite3.Row) else r[0]) for r in candidates
    ]
    spreads = load_open_home_spreads(conn, ids)
    out: list[GamePrediction] = []
    for row in candidates:
        if isinstance(row, sqlite3.Row):
            gid = str(row["game_id"])
            season = int(row["season"])
            week = int(row["week"])
            home_id = str(row["home_team_id"])
            away_id = str(row["away_team_id"])
            neutral = bool(row["neutral_site"])
            actual = float(row["home_points"]) - float(row["away_points"])
        else:
            gid = str(row[0])
            season = int(row[1])
            week = int(row[2])
            home_id = str(row[3])
            away_id = str(row[4])
            neutral = bool(row[5])
            actual = float(row[6]) - float(row[7])
        spread = spreads.get(gid)
        if spread is None:
            continue
        margin = -float(spread)
        p_open = open_spread_to_home_prob(spread, sigma_0)
        out.append(
            GamePrediction(
                game_id=gid,
                season=season,
                week=week,
                home_team_id=home_id,
                away_team_id=away_id,
                neutral_site=neutral,
                predicted_margin=margin,
                actual_margin=actual,
                elo_home_rating=None,
                elo_away_rating=None,
                internal_elo_win_prob=p_open,
                logit_win_prob=None,
            )
        )
    return out


def _ensemble_predictions(
    predictions: list[GamePrediction],
    sigma_0: float,
    weights: dict[str, float],
    open_spreads: dict[str, float] | None = None,
) -> list[Prediction]:
    result: list[Prediction] = []
    for prediction in predictions:
        p = _ensemble_raw_prob(prediction, sigma_0, weights, open_spreads)
        if p is None:
            continue
        result.append((p, prediction.home_won))
    return result


def _ensemble_timed_probs(
    predictions: list[GamePrediction],
    sigma_0: float,
    weights: dict[str, float],
    open_spreads: dict[str, float] | None = None,
) -> list[TimedProb]:
    """Raw ensemble P(home) with (season, week) for walk-forward calibration."""
    rows: list[TimedProb] = []
    for prediction in predictions:
        p = _ensemble_raw_prob(prediction, sigma_0, weights, open_spreads)
        if p is None:
            continue
        rows.append(
            TimedProb(
                season=prediction.season,
                week=prediction.week,
                p=p,
                won=prediction.home_won,
                key=prediction.game_id,
            )
        )
    return rows


def _calibrate_timed(
    rows: list[TimedProb],
    *,
    method: Method = "platt",
    min_fit_n: int = 200,
    fold: Fold = "season",
) -> tuple[list[Prediction], list[Prediction], dict[str, float]]:
    """Walk-forward calibrate; return (calibrated, raw) Prediction lists + p_cal by key."""
    if not rows:
        return [], [], {}
    wf = walk_forward_calibrate(rows, method=method, min_fit_n=min_fit_n, fold=fold)
    calibrated: list[Prediction] = []
    raw: list[Prediction] = []
    by_key: dict[str, float] = {}
    for row, (raw_p, cal_p, _cal) in zip(rows, wf, strict=True):
        calibrated.append((cal_p, row.won))
        raw.append((raw_p, row.won))
        if row.key:
            by_key[row.key] = cal_p
    return calibrated, raw, by_key



def _same_game_set_text(sgs: SameGameSetEvidence) -> list[str]:
    lines = [
        f"  n_full (walk-forward fit games) : {sgs.n_full}",
        f"  n_overlap (model ∩ market close): {sgs.n_overlap}",
        f"  skipped (no close ML consensus) : {sgs.n_skipped}",
        f"  seasons with >=1 overlap game   : "
        f"{', '.join(str(s) for s in sgs.seasons_covered) or '(none)'}",
        f"  market min_books                : {sgs.market_min_books}",
        f"  promotion candidate             : {sgs.promotion_candidate}",
    ]
    if sgs.calibration_method is not None:
        lines.append(f"  ensemble calibrator              : {sgs.calibration_method}")
    for label, slice_ in (
        ("market", sgs.market),
        ("ensemble (calibrated)", sgs.ensemble),
        ("ensemble (raw)", sgs.ensemble_raw),
        ("ridge", sgs.ridge),
        ("Elo-only", sgs.elo),
    ):
        if slice_ is None:
            lines.append(f"  [{label}] n=0 (unavailable on overlap)")
            continue
        lines.append(
            f"  [{label}] n={slice_.n_games}  "
            f"brier={slice_.brier:.4f}  log_loss={slice_.log_loss:.4f}"
        )
    if (
        sgs.ensemble is not None
        and sgs.market is not None
        and sgs.beat_market is not None
    ):
        verb = "beats" if sgs.beat_market else "does not beat"
        lines.append(
            f"    ensemble {verb} market on log loss "
            f"({sgs.ensemble.log_loss:.4f} vs {sgs.market.log_loss:.4f}) "
            f"[same n={sgs.n_overlap}]"
        )
    if sgs.median_clv is not None:
        sign_ok = "yes" if sgs.median_clv_non_negative else "no"
        lines.append(
            f"  median CLV (model-vs-close)     : "
            f"{sgs.median_clv:+.4f} prob-pts "
            f"({sgs.median_clv_bps:+.1f} bps); "
            f"non_negative={sign_ok}"
        )
        lines.append(f"  CLV formula: {sgs.clv_formula}")
    else:
        lines.append("  median CLV: N/A (no overlap)")

    lines.append("")
    lines.append("  -- market blend (settings.blend.market_weight_floor) --")
    if sgs.market_weight_floor is not None:
        lines.append(f"  market_weight_floor            : {sgs.market_weight_floor:.2f}")
    if sgs.blend_weight_by_season:
        wtxt = ", ".join(
            f"{season}={w:.2f}" for season, w in sorted(sgs.blend_weight_by_season.items())
        )
        lines.append(f"  walk-forward w by season       : {wtxt}")
    for label, slice_ in (
        ("blend walk-forward (w>=floor)", sgs.blend_fitted),
        ("blend fixed w=floor", sgs.blend_at_floor),
        ("pure model w=0 (bracket)", sgs.blend_pure_model),
        ("pure market w=1 (bracket)", sgs.blend_pure_market),
    ):
        if slice_ is None:
            continue
        lines.append(
            f"  [{label}] n={slice_.n_games}  "
            f"brier={slice_.brier:.4f}  log_loss={slice_.log_loss:.4f}"
        )
    if sgs.blend_non_worse_than_market is not None and sgs.market is not None:
        verb = "non-worse than" if sgs.blend_non_worse_than_market else "WORSE than"
        bf = sgs.blend_fitted
        if bf is not None:
            lines.append(
                f"    blend walk-forward {verb} market on log loss "
                f"({bf.log_loss:.4f} vs {sgs.market.log_loss:.4f})"
            )
    lines.append(
        f"  blend gate interpretation       : {sgs.blend_gate_interpretation}"
    )
    lines.append(
        "  NOTE: beat_market gate stays on pure calibrated ensemble "
        "(skill to justify lowering floor). High w makes blend≈market; "
        "redefining beat_market on the blend would game the gate."
    )
    return lines


def model_vs_close_clv(p_model: float, p_close: float) -> float:
    """Model-vs-close probability CLV on the side the model favors.

    Treats the model's fair P as the bet price and the vig-free close P as
    the closing price. Open-ML book CLV is not used (CFBD historical open
    ML is typically N/A).

    Formula (home-win probabilities in; return in probability points):

    * If ``p_model >= 0.5`` (model favors home): ``clv = p_close - p_model``
    * Else (model favors away): ``clv = p_model - p_close``
      (equivalently ``(1 - p_close) - (1 - p_model)``)

    Positive CLV means the closing market assigned more probability to the
    model's favored side than the model did — classic "got a better number
    than close." Units: probability points (0.01 = 1 percentage point);
    multiply by 10_000 for bps of probability.
    """
    if p_model >= 0.5:
        return p_close - p_model
    return p_model - p_close


def median_model_vs_close_clv(
    model_and_close: list[tuple[float, float]],
) -> float | None:
    """Median of ``model_vs_close_clv`` over ``(p_model, p_close)`` pairs."""
    if not model_and_close:
        return None
    return float(median(model_vs_close_clv(p_m, p_c) for p_m, p_c in model_and_close))



def _build_same_game_set(
    *,
    fit_predictions: list[GamePrediction],
    market_home: dict[str, float],
    sigma_0: float,
    ensemble_weights: dict[str, float] | None,
    market_min_books: int,
    ensemble_calibrated_by_game: dict[str, float] | None = None,
    calibration_method: str | None = None,
    market_weight_floor: float = DEFAULT_MARKET_WEIGHT_FLOOR,
    open_spreads: dict[str, float] | None = None,
) -> SameGameSetEvidence | None:
    """Score market / ensemble / ridge / Elo on the identical overlap set.

    ``ensemble`` metrics / beat_market / median CLV use the *calibrated*
    ensemble when ``ensemble_calibrated_by_game`` is provided; raw ensemble
    is still scored into ``ensemble_raw`` for the report. Promotion gap and
    beat_market therefore see the walk-forward calibrator, never a fit on
    the evaluation slice.

    Returns None only when there are no fit predictions. An empty overlap
    still returns a struct with n_overlap=0 so coverage counts are never
    silent (n_full / n_overlap / skipped always printed by ``as_text``).
    """
    n_full = len(fit_predictions)
    if n_full == 0:
        return None

    overlap_preds = [p for p in fit_predictions if p.game_id in market_home]
    n_overlap = len(overlap_preds)
    n_skipped = n_full - n_overlap
    seasons_covered = tuple(sorted({p.season for p in overlap_preds}))

    if n_overlap == 0:
        return SameGameSetEvidence(
            n_full=n_full,
            n_overlap=0,
            n_skipped=n_skipped,
            seasons_covered=(),
            market_min_books=market_min_books,
            promotion_candidate="ensemble",
            market=None,
            ensemble=None,
            ensemble_raw=None,
            ridge=None,
            elo=None,
            beat_market=None,
            median_clv=None,
            median_clv_bps=None,
            median_clv_non_negative=None,
            calibration_method=calibration_method,
            market_weight_floor=float(market_weight_floor),
        )

    market_probs: list[Prediction] = [
        (market_home[p.game_id], p.home_won) for p in overlap_preds
    ]
    ridge_probs: list[Prediction] = [
        (_ridge_prob(p, sigma_0), p.home_won) for p in overlap_preds
    ]
    elo_probs, _ = _elo_probs(overlap_preds)

    # Ensemble + CLV pairs in one pass so beat_market and median CLV share
    # the exact same (p_model, p_close, outcome) rows. Prefer calibrated P.
    ensemble_cal_probs: list[Prediction] = []
    ensemble_raw_probs: list[Prediction] = []
    paired_market_for_ensemble: list[Prediction] = []
    model_close_pairs: list[tuple[float, float]] = []
    cal_map = ensemble_calibrated_by_game or {}
    if ensemble_weights is not None:
        for prediction in overlap_preds:
            p_raw = _ensemble_raw_prob(
                prediction, sigma_0, ensemble_weights, open_spreads
            )
            if p_raw is None:
                continue
            p_ens = cal_map.get(prediction.game_id, p_raw)
            p_mkt = market_home[prediction.game_id]
            ensemble_cal_probs.append((p_ens, prediction.home_won))
            ensemble_raw_probs.append((p_raw, prediction.home_won))
            paired_market_for_ensemble.append((p_mkt, prediction.home_won))
            # CLV stays on raw ensemble: calibrator is for gap + beat_market
            # only (promotion.json). Softening probs toward 0.5 systematically
            # flips model-vs-close CLV sign without reflecting a real edge change.
            model_close_pairs.append((p_raw, p_mkt))

    market_slice = _score_slice("market close (same-game set)", market_probs)
    ridge_slice = _score_slice("ridge (same-game set)", ridge_probs)
    elo_slice = (
        _score_slice("Elo-only (same-game set)", elo_probs) if elo_probs else None
    )
    ensemble_slice = (
        _score_slice("ensemble calibrated (same-game set)", ensemble_cal_probs)
        if ensemble_cal_probs
        else None
    )
    ensemble_raw_slice = (
        _score_slice("ensemble raw (same-game set)", ensemble_raw_probs)
        if ensemble_raw_probs
        else None
    )

    beat_market: bool | None = None
    if ensemble_cal_probs and paired_market_for_ensemble:
        beat_market = log_loss(ensemble_cal_probs) < log_loss(paired_market_for_ensemble)

    med_clv = median_model_vs_close_clv(model_close_pairs)
    med_bps = (med_clv * 10_000.0) if med_clv is not None else None
    med_nonneg = (med_clv >= 0.0) if med_clv is not None else None

    # Market blend evidence: walk-forward w in [floor, 1] on calibrated ensemble.
    # Pure-model (w=0) and pure-market (w=1) brackets ignore the floor clamp so
    # the report shows the full range; fitted / at-floor respect the floor.
    blend_fitted_slice = None
    blend_at_floor_slice = None
    blend_pure_model_slice = None
    blend_pure_market_slice = None
    blend_w_by_season: dict[int, float] | None = None
    blend_non_worse: bool | None = None
    floor = float(market_weight_floor)
    if ensemble_cal_probs and paired_market_for_ensemble:
        blend_rows: list[BlendRow] = []
        # Rebuild in overlap order matching ensemble_cal_probs construction.
        if ensemble_weights is not None:
            for prediction in overlap_preds:
                p_raw = _ensemble_raw_prob(
                    prediction, sigma_0, ensemble_weights, open_spreads
                )
                if p_raw is None:
                    continue
                p_ens = cal_map.get(prediction.game_id, p_raw)
                blend_rows.append(
                    BlendRow(
                        season=prediction.season,
                        p_market=market_home[prediction.game_id],
                        p_model=p_ens,
                        won=prediction.home_won,
                        key=prediction.game_id,
                    )
                )
        if blend_rows:
            wf = walk_forward_market_blend(blend_rows, floor=floor)
            blend_w_by_season = dict(wf.weight_by_season)
            blend_fitted_slice = _score_slice(
                f"market blend walk-forward (floor={floor:.2f})", wf.blended
            )
            blend_at_floor_slice = _score_slice(
                f"market blend fixed w={floor:.2f}",
                score_fixed_weight(blend_rows, floor, floor=floor),
            )
            # Brackets: pass floor=0 so w=0 is truly pure model.
            blend_pure_model_slice = _score_slice(
                "pure model w=0 (bracket)",
                score_fixed_weight(blend_rows, 0.0, floor=0.0),
            )
            blend_pure_market_slice = _score_slice(
                "pure market w=1 (bracket)",
                score_fixed_weight(blend_rows, 1.0, floor=floor),
            )
            blend_non_worse = blend_fitted_slice.log_loss <= market_slice.log_loss

    return SameGameSetEvidence(
        n_full=n_full,
        n_overlap=n_overlap,
        n_skipped=n_skipped,
        seasons_covered=seasons_covered,
        market_min_books=market_min_books,
        promotion_candidate="ensemble",
        market=market_slice,
        ensemble=ensemble_slice,
        ensemble_raw=ensemble_raw_slice,
        ridge=ridge_slice,
        elo=elo_slice,
        beat_market=beat_market,
        median_clv=med_clv,
        median_clv_bps=med_bps,
        median_clv_non_negative=med_nonneg,
        calibration_method=calibration_method,
        market_weight_floor=floor,
        blend_weight_by_season=blend_w_by_season,
        blend_fitted=blend_fitted_slice,
        blend_at_floor=blend_at_floor_slice,
        blend_pure_model=blend_pure_model_slice,
        blend_pure_market=blend_pure_market_slice,
        blend_non_worse_than_market=blend_non_worse,
    )


def _load_market_home_probs(
    conn: sqlite3.Connection,
    game_ids: list[str],
    *,
    min_books: int,
) -> dict[str, float]:
    """Latest pre-kickoff ML HOME vig-free prob per game from market_consensus.

    Prefers shin, falls back to multiplicative. Rows with n_books below
    ``min_books`` are excluded (thin markets stay out of the promotion
    compare unless the caller lowered the floor via
    ``historical_cfbd_min_books``).
    """
    if not game_ids:
        return {}
    placeholders = ",".join("?" * len(game_ids))
    rows = conn.execute(
        f"""SELECT c.game_id, c.prob_shin, c.prob_multiplicative, c.n_books, c.as_of_utc
            FROM market_consensus c
            WHERE c.market = 'ML' AND c.side = 'HOME'
              AND c.game_id IN ({placeholders})
              AND c.n_books >= ?
            ORDER BY c.game_id, c.as_of_utc DESC""",
        (*game_ids, int(min_books)),
    ).fetchall()
    out: dict[str, float] = {}
    for row in rows:
        gid = str(row["game_id"])
        if gid in out:
            continue  # already took the latest as_of
        prob = row["prob_shin"] if row["prob_shin"] is not None else row["prob_multiplicative"]
        if prob is None:
            continue
        out[gid] = float(prob)
    return out


def _market_probs(
    predictions: list[GamePrediction],
    home_probs: dict[str, float],
) -> tuple[list[Prediction], int]:
    """Market close probabilities on the same game set as ridge."""
    probs: list[Prediction] = []
    skipped = 0
    for prediction in predictions:
        prob = home_probs.get(prediction.game_id)
        if prob is None:
            skipped += 1
            continue
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
    elo_k: float = DEFAULT_ELO_K,
    elo_hfa: float = HFA_ELO_POINTS,
    logit_l2_lambda: float = DEFAULT_LOGIT_L2_LAMBDA,
    logit_min_n: int = 30,
    ensemble_grid_step: float = 0.05,
    historical_cfbd_min_books: int | None = None,
    ensemble_calibrator: Method = "platt",
    ensemble_calibrator_min_fit_n: int = 200,
    ensemble_calibrator_fold: Fold = "season",
    use_open_market_prior: bool = True,
    unlock_week1_open_prior: bool = True,
) -> MoneylineBacktestReport:
    run = run_walk_forward(
        conn, list(seasons), ridge_lambda=ridge_lambda, min_games=min_games,
        apply_shrinkage=apply_shrinkage, coeffs=coeffs, elo_k=elo_k, elo_hfa=elo_hfa,
        logit_l2_lambda=logit_l2_lambda, logit_min_n=logit_min_n,
    )

    fit_predictions = [p for p in run.predictions if p.season not in STRESS_SEASONS]
    stress_predictions = [p for p in run.predictions if p.season in STRESS_SEASONS]

    sigma_0 = calibrate_sigma([p.actual_margin - p.predicted_margin for p in fit_predictions])

    # Week-1 open-prior unlock (ridge blackout at min_games). Appended after
    # sigma fit so residuals stay ridge-only; open-only rows do not re-fit sigma.
    if use_open_market_prior and unlock_week1_open_prior:
        existing = {p.game_id for p in run.predictions}
        unlocked = _week1_open_prior_predictions(
            conn, seasons, sigma_0=sigma_0, existing_ids=existing,
        )
        fit_predictions = fit_predictions + [
            p for p in unlocked if p.season not in STRESS_SEASONS
        ]
        stress_predictions = stress_predictions + [
            p for p in unlocked if p.season in STRESS_SEASONS
        ]

    def to_probs(preds: list[GamePrediction]) -> list[Prediction]:
        return [(_ridge_prob(p, sigma_0), p.home_won) for p in preds]

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

    open_spreads: dict[str, float] | None = None
    if use_open_market_prior:
        open_ids = [p.game_id for p in fit_predictions] + [
            p.game_id for p in stress_predictions
        ]
        open_spreads = load_open_home_spreads(conn, open_ids)

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
    ensemble_seasons_raw_metrics = None
    ensemble_stress_metrics = None
    ensemble_cal_by_game: dict[str, float] = {}
    cal_method_label: str | None = None
    if ensemble_weights is not None:
        timed = _ensemble_timed_probs(
            fit_predictions, sigma_0, ensemble_weights, open_spreads
        )
        if timed:
            cal_preds, raw_preds, ensemble_cal_by_game = _calibrate_timed(
                timed,
                method=ensemble_calibrator,
                min_fit_n=ensemble_calibrator_min_fit_n,
                fold=ensemble_calibrator_fold,
            )
            cal_method_label = f"{ensemble_calibrator}/{ensemble_calibrator_fold}"
            ensemble_seasons_metrics = _score_slice(
                f"ensemble calibrated ({cal_method_label}), same seasons/games as ridge",
                cal_preds,
            )
            ensemble_seasons_raw_metrics = _score_slice(
                "ensemble raw (pre-calibrator), same seasons/games as ridge",
                raw_preds,
            )
        ensemble_stress_probs = _ensemble_predictions(
            stress_predictions, sigma_0, ensemble_weights, open_spreads
        )
        if ensemble_stress_probs:
            # Stress slice: fit calibrator on all non-stress games, apply once.
            if timed and ensemble_calibrator != "identity":
                from cfb_analytics.backtest.prob_calibrate import fit_calibrator
                stress_cal = fit_calibrator(
                    [r.p for r in timed],
                    [r.won for r in timed],
                    method=ensemble_calibrator,
                )
                ensemble_stress_probs = [
                    (stress_cal.apply(p), won) for p, won in ensemble_stress_probs
                ]
            ensemble_stress_metrics = _score_slice(
                f"ensemble calibrated ({cal_method_label}), 2020 stress slice",
                ensemble_stress_probs,
            )

    from cfb_analytics import config as _config

    market_settings = _config.settings().get("market", {})
    if historical_cfbd_min_books is not None:
        market_min_books = int(historical_cfbd_min_books)
    elif market_settings.get("historical_cfbd_min_books") is not None:
        # Baseline-only override documented in settings; never used for live CORE.
        market_min_books = int(market_settings["historical_cfbd_min_books"])
    else:
        market_min_books = int(market_settings.get("min_books_for_consensus", 3))

    market_home = _load_market_home_probs(
        conn,
        [p.game_id for p in fit_predictions],
        min_books=market_min_books,
    )
    market_fit_probs, market_fit_skipped = _market_probs(fit_predictions, market_home)
    # Require meaningful overlap with the ridge game set; otherwise say so.
    coverage = (len(market_fit_probs) / len(fit_predictions)) if fit_predictions else 0.0
    market_seasons_metrics = None
    market_note = None
    if market_fit_probs and coverage >= 0.5:
        market_seasons_metrics = _score_slice(
            "market close, covered subset (see same-game-set for honest compare)",
            market_fit_probs,
        )
    else:
        market_note = (
            f"market: N/A (coverage) — "
            f"{len(market_fit_probs)}/{len(fit_predictions)} games had close ML "
            f"consensus with n_books>={market_min_books}"
        )

    blend_settings = _config.settings()
    market_weight_floor = market_weight_floor_from_settings(blend_settings)

    same_game_set = _build_same_game_set(
        fit_predictions=fit_predictions,
        market_home=market_home,
        sigma_0=sigma_0,
        ensemble_weights=ensemble_weights,
        market_min_books=market_min_books,
        ensemble_calibrated_by_game=ensemble_cal_by_game or None,
        calibration_method=cal_method_label,
        market_weight_floor=market_weight_floor,
        open_spreads=open_spreads,
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
        ensemble_seasons_raw=ensemble_seasons_raw_metrics,
        ensemble_stress=ensemble_stress_metrics,
        ensemble_calibration_method=cal_method_label,
        market_seasons=market_seasons_metrics,
        market_note=market_note,
        skipped_market_unpriced=market_fit_skipped,
        same_game_set=same_game_set,
    )
