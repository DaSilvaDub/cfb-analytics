#!/usr/bin/env python3
"""Walk-forward spike: Elo HFA / away-favorite skill gap (2023–2025).

Leakage rules:
- HFA candidates are either global constants or season-blocked picks using
  *prior seasons only* (no same-season outcomes in the HFA choice).
- Market close ML is used only for segmentation + scoring baseline — never as
  a same-game feature. Open prior (weeks ≤2) matches production.

Stdlib-friendly; raw JSON under artifacts/; scorecard under docs/scorecards/.
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
    _ridge_prob,
)
from cfb_analytics.features.open_market_prior import (
    OPEN_MARKET_PRIOR_MAX_WEEK,
    OPEN_MARKET_PRIOR_WEIGHT,
    load_open_home_spreads,
)
from cfb_analytics.models.elo import HFA_ELO_POINTS, expected_score

SEASONS = (2023, 2024, 2025)
WEIGHTS = {"ridge": 0.15 / 0.95, "internal_elo": 0.80 / 0.95}
# Plan default 60 ≈ 2.6 pts. Ridge mid-season HFA ~5.4 pts ≈ 125 Elo under
# the plan's 60↔2.6 conversion — motivates probing the upper end.
HFA_GRID = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0)
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


def evaluate(
    label: str,
    predictions: list[GamePrediction],
    market: dict[str, float],
    *,
    sigma: float | None = None,
    open_spreads: dict[str, float] | None = None,
    weights: dict[str, float] = WEIGHTS,
) -> dict:
    if sigma is None:
        residuals = [p.actual_margin - p.predicted_margin for p in predictions]
        sigma = calibrate_sigma(residuals) if len(residuals) >= 2 else 16.5

    buckets: dict[str, dict[str, list]] = defaultdict(
        lambda: {
            "ens": [], "mkt": [], "elo": [], "ridge": [],
            "ens_bias": [], "elo_bias": [],
        }
    )

    for p in predictions:
        p_mkt = market.get(p.game_id)
        if p_mkt is None:
            continue
        p_ens = _ensemble_raw_prob(
            p,
            sigma,
            weights,
            open_spreads,
            open_prior_max_week=OPEN_MARKET_PRIOR_MAX_WEEK,
            open_prior_weight=OPEN_MARKET_PRIOR_WEIGHT,
        )
        if p_ens is None:
            continue
        side = _fav_side(p_mkt)
        y = p.home_won
        keys = [side, "all", f"season_{p.season}"]
        if p.neutral_site:
            keys.append("neutral")
        for key in keys:
            buckets[key]["ens"].append((p_ens, y))
            buckets[key]["mkt"].append((p_mkt, y))
            buckets[key]["ens_bias"].append(p_ens - (1.0 if y else 0.0))
            buckets[key]["ridge"].append((_ridge_prob(p, sigma), y))
            if p.internal_elo_win_prob is not None:
                buckets[key]["elo"].append((p.internal_elo_win_prob, y))
                buckets[key]["elo_bias"].append(
                    p.internal_elo_win_prob - (1.0 if y else 0.0)
                )

    def pack(b: dict) -> dict:
        e, m = _score(b["ens"]), _score(b["mkt"])
        el, r = _score(b["elo"]), _score(b["ridge"])
        return {
            "n": e.n if e else 0,
            "ens_ll": e.log_loss if e else None,
            "ens_brier": e.brier if e else None,
            "mkt_ll": m.log_loss if m else None,
            "gap": (e.log_loss - m.log_loss) if e and m else None,
            "elo_ll": el.log_loss if el else None,
            "ridge_ll": r.log_loss if r else None,
            "ens_bias": mean(b["ens_bias"]) if b["ens_bias"] else None,
            "elo_bias": mean(b["elo_bias"]) if b["elo_bias"] else None,
        }

    full = pack(buckets["all"])
    by_season = {
        str(s): pack(buckets[f"season_{s}"])
        for s in SEASONS
        if buckets[f"season_{s}"]["ens"]
    }
    return {
        "label": label,
        "sigma": sigma,
        "n_predictions": len(predictions),
        "full": full,
        "away_fav": pack(buckets["away_fav"]),
        "home_fav": pack(buckets["home_fav"]),
        "pickem": pack(buckets["pickem"]),
        "neutral": pack(buckets["neutral"]),
        "by_season": by_season,
        "beat_market_full": (
            full["ens_ll"] < full["mkt_ll"]
            if full["ens_ll"] is not None and full["mkt_ll"] is not None
            else None
        ),
    }


def run_hfa(conn, elo_hfa: float, seasons=SEASONS) -> list[GamePrediction]:
    t0 = time.time()
    run = run_walk_forward(
        conn, list(seasons), skip_logit=True, elo_hfa=elo_hfa,
    )
    print(
        f"  [hfa={elo_hfa:g}] preds={len(run.predictions)} "
        f"skip_hist={run.skipped_insufficient_history} "
        f"skip_unrated={run.skipped_unrated_team} "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )
    return run.predictions


def build_wf_preds_from_grid(
    pred_cache: dict[float, list[GamePrediction]],
    grid_rows: list[dict],
) -> tuple[dict[int, float], list[GamePrediction]]:
    """Season-blocked HFA: pick argmin prior-season Elo LL from the constant grid.

    2023 has no prior in-window → default plan HFA. No re-fit required.
    """
    # elo_ll[hfa][season] from already-scored grid rows
    elo_ll: dict[float, dict[str, float]] = {}
    for row in grid_rows:
        hfa = float(row["elo_hfa"])
        elo_ll[hfa] = {}
        for s, pack in row["by_season"].items():
            if pack.get("elo_ll") is not None:
                elo_ll[hfa][s] = pack["elo_ll"]

    chosen: dict[int, float] = {}
    for season in SEASONS:
        priors = [str(s) for s in SEASONS if s < season]
        if not priors:
            chosen[season] = HFA_ELO_POINTS
            print(f"  season {season}: no prior → HFA={HFA_ELO_POINTS:g}", flush=True)
            continue
        best_hfa, best_ll = HFA_ELO_POINTS, math.inf
        for hfa in HFA_GRID:
            vals = [elo_ll[hfa][s] for s in priors if s in elo_ll.get(hfa, {})]
            if not vals:
                continue
            ll = sum(vals) / len(vals)
            if ll < best_ll:
                best_ll, best_hfa = ll, hfa
        chosen[season] = best_hfa
        print(
            f"  season {season}: prior-fit HFA={best_hfa:g} "
            f"(mean prior Elo LL={best_ll:.4f} over {priors})",
            flush=True,
        )

    # Stitch predictions: for each season, take that season's games from the
    # chosen HFA's full run (same as re-running that HFA on that season alone).
    by_hfa_season: dict[tuple[float, int], list[GamePrediction]] = defaultdict(list)
    for hfa, preds in pred_cache.items():
        for p in preds:
            by_hfa_season[(hfa, p.season)].append(p)

    out: list[GamePrediction] = []
    for season, hfa in chosen.items():
        out.extend(by_hfa_season[(hfa, season)])
    return chosen, out


def print_row(row: dict) -> None:
    af, hf, full = row["away_fav"], row["home_fav"], row["full"]
    af_gap = af["gap"] if af["gap"] is not None else float("nan")
    hf_gap = hf["gap"] if hf["gap"] is not None else float("nan")
    full_gap = full["gap"] if full["gap"] is not None else float("nan")
    print(
        f"{row['label']:22s} "
        f"away n={af['n']:4d} gap={af_gap:+.4f} ens={af['ens_ll']} "
        f"| home n={hf['n']:4d} gap={hf_gap:+.4f} "
        f"| full gap={full_gap:+.4f} ens={full['ens_ll']} "
        f"elo={full['elo_ll']} beat={row['beat_market_full']}",
        flush=True,
    )


def main() -> int:
    db = ROOT / "data" / "cfb.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    results: list[dict] = []
    pred_cache: dict[float, list[GamePrediction]] = {}

    print("== constant HFA grid ==", flush=True)
    for hfa in HFA_GRID:
        pred_cache[hfa] = run_hfa(conn, hfa)

    all_ids: set[str] = set()
    for preds in pred_cache.values():
        all_ids |= {p.game_id for p in preds}

    market = _load_market_home_probs(conn, sorted(all_ids), min_books=1)
    open_spreads = load_open_home_spreads(conn, sorted(all_ids))
    print(
        f"market coverage={len(market)}/{len(all_ids)} "
        f"open_spreads={len(open_spreads)}",
        flush=True,
    )

    equal_team_p = {
        f"hfa_{h:g}": expected_score(1500, 1500, hfa=h) for h in HFA_GRID
    }

    grid_rows: list[dict] = []
    for hfa in HFA_GRID:
        row = evaluate(
            f"elo_hfa={hfa:g}",
            pred_cache[hfa],
            market,
            open_spreads=open_spreads,
        )
        row["elo_hfa"] = hfa
        row["equal_team_p_home"] = equal_team_p[f"hfa_{hfa:g}"]
        results.append(row)
        grid_rows.append(row)
        print_row(row)

    print("== season-blocked WF HFA (from grid, prior seasons only) ==", flush=True)
    chosen, wf_preds = build_wf_preds_from_grid(pred_cache, grid_rows)
    wf_row = evaluate(
        "wf_prior_hfa",
        wf_preds,
        market,
        open_spreads=open_spreads,
    )
    wf_row["elo_hfa_by_season"] = chosen
    results.append(wf_row)
    print_row(wf_row)

    base = next(r for r in results if r.get("elo_hfa") == HFA_ELO_POINTS)
    diagnosis = {
        "plan_hfa": HFA_ELO_POINTS,
        "equal_team_p_home_at_60": expected_score(1500, 1500, hfa=HFA_ELO_POINTS),
        "baseline_away_fav_gap": base["away_fav"]["gap"],
        "baseline_home_fav_gap": base["home_fav"]["gap"],
        "baseline_full_gap": base["full"]["gap"],
        "baseline_away_ens_bias": base["away_fav"]["ens_bias"],
        "baseline_away_elo_bias": base["away_fav"]["elo_bias"],
        "baseline_home_ens_bias": base["home_fav"]["ens_bias"],
        "baseline_home_elo_bias": base["home_fav"]["elo_bias"],
        "baseline_neutral": base["neutral"],
        "ridge_mid_2024_hfa_pts_probe": 5.37,
        "note": (
            "Positive bias = model too high on P(home). "
            "Away-fav: if elo_bias << 0, Elo understates HFA (too bullish on away)."
        ),
    }

    out = {
        "diagnosis": diagnosis,
        "wf_hfa_by_season": chosen,
        "equal_team_p_home": equal_team_p,
        "candidates": results,
    }
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "spike_hfa_away_fav_2023_2025.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
