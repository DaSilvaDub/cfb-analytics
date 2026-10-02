#!/usr/bin/env python3
"""Segment residual ens-vs-mkt logloss gap on the shipped 2023–2025 pipeline.

Production levers (early σ ×1.5 W≤4, W1–2 open replace + W1 unlock,
W3–4 soft open blend w=0.75). Close ML is scoring/segmentation only —
never a same-game feature. Stdlib-friendly; writes artifacts JSON.
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
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfb_analytics.backtest.calibration import calibrate_sigma
from cfb_analytics.backtest.harness import GamePrediction, run_walk_forward
from cfb_analytics.backtest.metrics import brier_score, log_loss
from cfb_analytics.backtest.moneyline import (
    _ensemble_raw_prob,
    _load_market_home_probs,
    _week1_open_prior_predictions,
)
from cfb_analytics.features.open_market_prior import (
    OPEN_MARKET_BLEND_MAX_WEEK,
    OPEN_MARKET_BLEND_WEIGHT,
    OPEN_MARKET_PRIOR_MAX_WEEK,
    OPEN_MARKET_PRIOR_WEIGHT,
    load_open_home_spreads,
)

SEASONS = (2023, 2024, 2025)
# Production mix without logit for speed (matches prior spikes).
WEIGHTS = {"ridge": 0.15 / 0.95, "internal_elo": 0.80 / 0.95}
PICKEM_BAND = 0.02


@dataclass
class Slice:
    n: int
    log_loss: float
    brier: float


def _score(pairs: list[tuple[float, bool]]) -> Slice | None:
    if not pairs:
        return None
    return Slice(n=len(pairs), log_loss=log_loss(pairs), brier=brier_score(pairs))


def _fav_side(p_mkt: float) -> str:
    if abs(p_mkt - 0.5) < PICKEM_BAND:
        return "pickem"
    return "away_fav" if p_mkt < 0.5 else "home_fav"


def _mkt_edge_bucket(p_mkt: float) -> str:
    """Strength of favorite from close ML (|p-0.5|)."""
    e = abs(p_mkt - 0.5)
    if e < 0.05:
        return "edge_0_5pp"
    if e < 0.10:
        return "edge_5_10pp"
    if e < 0.15:
        return "edge_10_15pp"
    if e < 0.25:
        return "edge_15_25pp"
    return "edge_25pp_plus"


def _open_spread_bucket(spread: float | None) -> str:
    """Home spread (negative => home favored). None => no_open."""
    if spread is None:
        return "spread_no_open"
    s = abs(float(spread))
    if s < 3:
        return "spread_abs_0_3"
    if s < 7:
        return "spread_abs_3_7"
    if s < 14:
        return "spread_abs_7_14"
    return "spread_abs_14_plus"


def _week_bucket(week: int) -> str:
    if week <= 2:
        return "weeks_1_2"
    if week <= 4:
        return "weeks_3_4"
    if week <= 8:
        return "weeks_5_8"
    if week <= 12:
        return "weeks_9_12"
    return "weeks_13_plus"


def _load_game_meta(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT g.game_id, g.neutral_site, g.conference_game,
               th.conference AS home_conf, ta.conference AS away_conf
        FROM games g
        LEFT JOIN teams th ON th.team_id = g.home_team_id
        LEFT JOIN teams ta ON ta.team_id = g.away_team_id
        """
    ).fetchall()
    out: dict[str, dict] = {}
    for gid, neut, conf_game, hc, ac in rows:
        out[str(gid)] = {
            "neutral": bool(neut),
            "conference_game": bool(conf_game),
            "home_conf": hc,
            "away_conf": ac,
        }
    return out


def evaluate(
    predictions: list[GamePrediction],
    close_mkt: dict[str, float],
    open_spreads: dict[str, float],
    game_meta: dict[str, dict],
    *,
    sigma: float,
) -> dict:
    buckets: dict[str, dict[str, list]] = defaultdict(
        lambda: {"ens": [], "mkt": [], "bias": [], "abs_gap_contrib": []}
    )

    for p in predictions:
        p_mkt = close_mkt.get(p.game_id)
        if p_mkt is None:
            continue
        p_ens = _ensemble_raw_prob(
            p,
            sigma,
            WEIGHTS,
            open_spreads,
            open_prior_max_week=OPEN_MARKET_PRIOR_MAX_WEEK,
            open_prior_weight=OPEN_MARKET_PRIOR_WEIGHT,
            open_blend_max_week=OPEN_MARKET_BLEND_MAX_WEEK,
            open_blend_weight=OPEN_MARKET_BLEND_WEIGHT,
        )
        if p_ens is None:
            continue
        y = p.home_won
        meta = game_meta.get(p.game_id, {})
        open_sp = open_spreads.get(p.game_id)

        keys = [
            "all",
            _fav_side(p_mkt),
            _week_bucket(p.week),
            f"week_{p.week}",
            f"season_{p.season}",
            _mkt_edge_bucket(p_mkt),
            _open_spread_bucket(open_sp),
            "neutral" if meta.get("neutral") else "non_neutral",
            "conf_game" if meta.get("conference_game") else "non_conf_game",
        ]
        # W5+ aggregate (binding skill slice)
        if p.week >= 5:
            keys.append("weeks_5_plus")
        if p.week <= 4:
            keys.append("weeks_1_4")
        # Cross: fav × late
        if p.week >= 5:
            keys.append(f"w5plus_{_fav_side(p_mkt)}")
            keys.append(f"w5plus_{_mkt_edge_bucket(p_mkt)}")

        for key in keys:
            buckets[key]["ens"].append((p_ens, y))
            buckets[key]["mkt"].append((p_mkt, y))
            buckets[key]["bias"].append(p_ens - (1.0 if y else 0.0))

    def pack(b: dict) -> dict | None:
        e, m = _score(b["ens"]), _score(b["mkt"])
        if e is None or m is None:
            return None
        gap = e.log_loss - m.log_loss
        # Approximate gap contribution: n/N_all * gap (filled later)
        return {
            "n": e.n,
            "ens_ll": round(e.log_loss, 6),
            "mkt_ll": round(m.log_loss, 6),
            "gap": round(gap, 6),
            "ens_bias": round(mean(b["bias"]), 6) if b["bias"] else None,
            "brier_ens": round(e.brier, 6),
            "brier_mkt": round(m.brier, 6),
        }

    packed = {k: pack(v) for k, v in buckets.items()}
    packed = {k: v for k, v in packed.items() if v is not None}
    n_all = packed.get("all", {}).get("n") or 1
    for k, v in packed.items():
        # share of total absolute positive gap mass (only positive gaps)
        v["gap_mass"] = round(v["n"] / n_all * max(v["gap"], 0.0), 6)
        v["n_share"] = round(v["n"] / n_all, 4)
    return packed


def main() -> int:
    t0 = time.time()
    db = ROOT / "data" / "cfb.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    print("running walk-forward...", flush=True)
    result = run_walk_forward(conn, list(SEASONS), skip_logit=True)
    preds = list(result.predictions)
    print(
        f"  base preds={len(preds)} skip_hist={result.skipped_insufficient_history} "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )

    residuals = [p.actual_margin - p.predicted_margin for p in preds]
    sigma = calibrate_sigma(residuals) if len(residuals) >= 2 else 16.5

    # Week-1 unlock (production)
    existing = {p.game_id for p in preds}
    unlocked = _week1_open_prior_predictions(
        conn, SEASONS, sigma_0=sigma, existing_ids=existing
    )
    preds = preds + unlocked
    print(f"  week1 unlocked={len(unlocked)} total_preds={len(preds)}", flush=True)

    all_ids = [p.game_id for p in preds]
    close_mkt = _load_market_home_probs(conn, all_ids, min_books=1)
    open_spreads = load_open_home_spreads(conn, all_ids)
    game_meta = _load_game_meta(conn)
    conn.close()

    packed = evaluate(preds, close_mkt, open_spreads, game_meta, sigma=sigma)

    # Rank positive-gap slices (exclude 'all' and tiny n)
    ranked = sorted(
        (
            (k, v)
            for k, v in packed.items()
            if k != "all" and v["n"] >= 80 and v["gap"] > 0
        ),
        key=lambda kv: kv[1]["gap_mass"],
        reverse=True,
    )
    ranked_by_gap = sorted(
        (
            (k, v)
            for k, v in packed.items()
            if k != "all" and v["n"] >= 80 and v["gap"] > 0
        ),
        key=lambda kv: kv[1]["gap"],
        reverse=True,
    )

    payload = {
        "generated_et": time.strftime("%Y-%m-%d %H:%M ET", time.localtime()),
        "seasons": list(SEASONS),
        "sigma": sigma,
        "n_preds": len(preds),
        "n_week1_unlocked": len(unlocked),
        "n_overlap": packed.get("all", {}).get("n"),
        "full": packed.get("all"),
        "open_prior_max_week": OPEN_MARKET_PRIOR_MAX_WEEK,
        "open_prior_weight": OPEN_MARKET_PRIOR_WEIGHT,
        "open_blend_max_week": OPEN_MARKET_BLEND_MAX_WEEK,
        "open_blend_weight": OPEN_MARKET_BLEND_WEIGHT,
        "weights": WEIGHTS,
        "note": (
            "skip_logit for speed (ridge/elo renorm); production promote uses "
            "full ensemble + platt/season. Gaps are raw ens vs close ML."
        ),
        "slices": packed,
        "top_by_gap_mass": [
            {"slice": k, **v} for k, v in ranked[:25]
        ],
        "top_by_gap": [
            {"slice": k, **v} for k, v in ranked_by_gap[:25]
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = ROOT / "artifacts" / "segment_residual_gap_2023_2025.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)
    print(f"full: {payload['full']}", flush=True)
    print("top by gap_mass:", flush=True)
    for row in payload["top_by_gap_mass"][:12]:
        print(
            f"  {row['slice']:28s} n={row['n']:4d} gap={row['gap']:+.4f} "
            f"mass={row['gap_mass']:.4f} bias={row['ens_bias']}",
            flush=True,
        )
    print("top by gap:", flush=True)
    for row in payload["top_by_gap"][:12]:
        print(
            f"  {row['slice']:28s} n={row['n']:4d} gap={row['gap']:+.4f} "
            f"mass={row['gap_mass']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
