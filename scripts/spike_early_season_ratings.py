#!/usr/bin/env python3
"""Walk-forward spike: weeks 1–4 early-season ratings path (2023–2025).

Measures candidates against same-game-set market close. Stdlib-friendly;
keeps raw transcripts under artifacts/ (gitignored). Scorecard lives under
docs/scorecards/ once a candidate is accepted or rejected.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfb_analytics.backtest.calibration import calibrate_sigma, margin_to_prob
from cfb_analytics.backtest.harness import GamePrediction, run_walk_forward
from cfb_analytics.backtest.metrics import brier_score, log_loss
from cfb_analytics.backtest.moneyline import (
    _load_market_home_probs,
    _member_probs,
)
from cfb_analytics.models.ensemble import pool_probabilities
from cfb_analytics.models.shrinkage import ShrinkageCoefficients, DEFAULT_COEFFICIENTS

SEASONS = (2023, 2024, 2025)
WEIGHTS = {"ridge": 0.15, "internal_elo": 0.80, "logit": 0.05}
EARLY_WEEKS = {1, 2, 3, 4}


@dataclass
class Slice:
    n: int
    log_loss: float
    brier: float


def _score(pairs: list[tuple[float, bool]]) -> Slice | None:
    if not pairs:
        return None
    return Slice(n=len(pairs), log_loss=log_loss(pairs), brier=brier_score(pairs))


def _ens_prob(pred: GamePrediction, sigma: float, weights: dict[str, float],
              *, sigma_scale: float = 1.0, ridge_scale: float = 1.0) -> float | None:
    # Optional early-season levers applied at score time (no leakage).
    scaled_sigma = sigma * sigma_scale
    member = _member_probs(pred, scaled_sigma)
    w = dict(weights)
    if ridge_scale != 1.0 and "ridge" in w:
        w["ridge"] = w["ridge"] * ridge_scale
    available = {k: v for k, v in w.items() if v > 0 and k in member}
    if not available or sum(available.values()) <= 0:
        return None
    return pool_probabilities(member, available)


def evaluate(
    label: str,
    predictions: list[GamePrediction],
    market: dict[str, float],
    *,
    sigma: float | None = None,
    weights: dict[str, float] = WEIGHTS,
    early_sigma_scale: float = 1.0,
    early_ridge_scale: float = 1.0,
) -> dict:
    if sigma is None:
        residuals = [p.actual_margin - p.predicted_margin for p in predictions]
        sigma = calibrate_sigma(residuals) if len(residuals) >= 2 else 16.5

    ens_all: list[tuple[float, bool]] = []
    mkt_all: list[tuple[float, bool]] = []
    ridge_all: list[tuple[float, bool]] = []
    elo_all: list[tuple[float, bool]] = []

    ens_early: list[tuple[float, bool]] = []
    mkt_early: list[tuple[float, bool]] = []
    ridge_early: list[tuple[float, bool]] = []
    elo_early: list[tuple[float, bool]] = []

    ens_late: list[tuple[float, bool]] = []
    mkt_late: list[tuple[float, bool]] = []

    week1_n = 0
    by_week_gap: dict[str, list[float]] = defaultdict(list)

    for p in predictions:
        p_mkt = market.get(p.game_id)
        if p_mkt is None:
            continue
        early = p.week in EARLY_WEEKS
        sigma_scale = early_sigma_scale if early else 1.0
        ridge_scale = early_ridge_scale if early else 1.0
        p_ens = _ens_prob(
            p, sigma, weights, sigma_scale=sigma_scale, ridge_scale=ridge_scale
        )
        if p_ens is None:
            continue
        p_ridge = margin_to_prob(p.predicted_margin, sigma * sigma_scale)
        p_elo = p.internal_elo_win_prob

        ens_all.append((p_ens, p.home_won))
        mkt_all.append((p_mkt, p.home_won))
        ridge_all.append((p_ridge, p.home_won))
        if p_elo is not None:
            elo_all.append((p_elo, p.home_won))

        if early:
            ens_early.append((p_ens, p.home_won))
            mkt_early.append((p_mkt, p.home_won))
            ridge_early.append((p_ridge, p.home_won))
            if p_elo is not None:
                elo_early.append((p_elo, p.home_won))
            if p.week == 1:
                week1_n += 1
        else:
            ens_late.append((p_ens, p.home_won))
            mkt_late.append((p_mkt, p.home_won))

    def pack(ens, mkt, ridge=None, elo=None):
        e, m = _score(ens), _score(mkt)
        out = {
            "n": e.n if e else 0,
            "ens_ll": e.log_loss if e else None,
            "ens_brier": e.brier if e else None,
            "mkt_ll": m.log_loss if m else None,
            "mkt_brier": m.brier if m else None,
            "gap": (e.log_loss - m.log_loss) if e and m else None,
        }
        if ridge is not None:
            r = _score(ridge)
            out["ridge_ll"] = r.log_loss if r else None
        if elo is not None:
            el = _score(elo)
            out["elo_ll"] = el.log_loss if el else None
        return out

    return {
        "label": label,
        "sigma": sigma,
        "n_predictions": len(predictions),
        "week1_overlap": week1_n,
        "full": pack(ens_all, mkt_all, ridge_all, elo_all),
        "weeks_1_4": pack(ens_early, mkt_early, ridge_early, elo_early),
        "weeks_5_plus": pack(ens_late, mkt_late),
        "beat_market_full": (
            (_score(ens_all).log_loss < _score(mkt_all).log_loss)
            if ens_all and mkt_all
            else None
        ),
    }


def run_variant(conn, label, **wf_kwargs) -> tuple[str, list[GamePrediction], float]:
    t0 = time.time()
    run = run_walk_forward(conn, list(SEASONS), **wf_kwargs)
    elapsed = time.time() - t0
    print(f"  [{label}] preds={len(run.predictions)} "
          f"skip_hist={run.skipped_insufficient_history} "
          f"skip_unrated={run.skipped_unrated_team} "
          f"elapsed={elapsed:.1f}s", flush=True)
    return label, run.predictions, elapsed


def main() -> int:
    db = ROOT / "data" / "cfb.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    results = []
    # Baseline (current production path; skip logit for speed — weights renorm)
    print("== baseline (skip_logit) ==", flush=True)
    _, base_preds, _ = run_variant(
        conn, "baseline", skip_logit=True, allow_prior_only=False,
    )
    # Collect market for union of all game ids we'll need
    all_ids = {p.game_id for p in base_preds}

    print("== prior_only (week-1 unlock) ==", flush=True)
    _, prior_preds, _ = run_variant(
        conn, "prior_only", skip_logit=True, allow_prior_only=True,
    )
    all_ids |= {p.game_id for p in prior_preds}

    print("== stronger_k=12 ==", flush=True)
    _, k12_preds, _ = run_variant(
        conn, "k12", skip_logit=True, allow_prior_only=False,
        coeffs=ShrinkageCoefficients(
            a=DEFAULT_COEFFICIENTS.a, b=DEFAULT_COEFFICIENTS.b,
            c=DEFAULT_COEFFICIENTS.c, k=12.0,
        ),
    )
    all_ids |= {p.game_id for p in k12_preds}

    print("== prior_only + k=12 ==", flush=True)
    _, prior_k12_preds, _ = run_variant(
        conn, "prior_only_k12", skip_logit=True, allow_prior_only=True,
        coeffs=ShrinkageCoefficients(
            a=DEFAULT_COEFFICIENTS.a, b=DEFAULT_COEFFICIENTS.b,
            c=DEFAULT_COEFFICIENTS.c, k=12.0,
        ),
    )
    all_ids |= {p.game_id for p in prior_k12_preds}

    print("== stronger_k=20 ==", flush=True)
    _, k20_preds, _ = run_variant(
        conn, "k20", skip_logit=True, allow_prior_only=False,
        coeffs=ShrinkageCoefficients(
            a=DEFAULT_COEFFICIENTS.a, b=DEFAULT_COEFFICIENTS.b,
            c=DEFAULT_COEFFICIENTS.c, k=20.0,
        ),
    )
    all_ids |= {p.game_id for p in k20_preds}

    market = _load_market_home_probs(conn, sorted(all_ids), min_books=1)
    print(f"market coverage on union: {len(market)}/{len(all_ids)}", flush=True)

    # Baseline sigma shared for post-hoc score-time levers
    base_sigma = calibrate_sigma(
        [p.actual_margin - p.predicted_margin for p in base_preds]
    )

    candidates = [
        # Re-fit ratings variants: calibrate sigma on that variant's residuals.
        ("baseline", base_preds, {"sigma": None}),
        ("prior_only", prior_preds, {"sigma": None}),
        ("shrink_k=12", k12_preds, {"sigma": None}),
        ("shrink_k=20", k20_preds, {"sigma": None}),
        ("prior_only+k=12", prior_k12_preds, {"sigma": None}),
        # Score-time levers on baseline residuals/sigma (no re-fit).
        ("early_sigma_x1.5", base_preds, {"sigma": base_sigma, "early_sigma_scale": 1.5}),
        ("early_sigma_x2.0", base_preds, {"sigma": base_sigma, "early_sigma_scale": 2.0}),
        ("early_ridge_w_x0.25", base_preds, {"sigma": base_sigma, "early_ridge_scale": 0.25}),
        ("early_ridge_w_x0", base_preds, {"sigma": base_sigma, "early_ridge_scale": 0.0}),
        ("prior_only+early_sigma_x1.5", prior_preds, {"sigma": None, "early_sigma_scale": 1.5}),
    ]

    # Fixed weights without logit (renorm among ridge+elo)
    w_no_logit = {"ridge": 0.15 / 0.95, "internal_elo": 0.80 / 0.95}

    for label, preds, kwargs in candidates:
        row = evaluate(label, preds, market, weights=w_no_logit, **kwargs)
        results.append(row)
        w14 = row["weeks_1_4"]
        full = row["full"]
        print(
            f"{label:28s} W1-4 n={w14['n']:4d} ens={w14['ens_ll']} mkt={w14['mkt_ll']} "
            f"gap={w14['gap']} | full ens={full['ens_ll']} gap={full['gap']} "
            f"week1={row['week1_overlap']}",
            flush=True,
        )

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "spike_early_season_ratings_2023_2025.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
