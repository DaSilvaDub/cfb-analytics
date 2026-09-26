#!/usr/bin/env python3
"""Walk-forward spike: weeks 3–4 soft open-market blend (weight < 1).

Keeps production weeks 1–2 full open replace (weight=1.0) + week-1 unlock.
Candidates soft-blend open-implied Phi(-spread/sigma) into ensemble for
weeks 3–4 only (and an optional week-4-only arm). Never uses close as a
same-game prior; close ML remains the scoring baseline.

Stdlib-friendly; raw JSON under artifacts/; scorecard under docs/scorecards/.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
    OPEN_MARKET_PRIOR_MAX_WEEK,
    OPEN_MARKET_PRIOR_WEIGHT,
    load_open_home_spreads,
)

SEASONS = (2023, 2024, 2025)
# Production mix without logit for speed (renorm ridge/elo) — matches prior spikes.
WEIGHTS = {"ridge": 0.15 / 0.95, "internal_elo": 0.80 / 0.95}
# Soft-blend candidates for weeks 3–4 (w < 1). 0.0 = baseline (no W3–4 blend).
BLEND_WEIGHTS = (0.0, 0.25, 0.50, 0.75)
# Optional: soft blend on week 4 only (same weights).
WEEK4_ONLY_WEIGHTS = (0.25, 0.50, 0.75)


@dataclass
class Slice:
    n: int
    log_loss: float
    brier: float


def _score(pairs: list[tuple[float, bool]]) -> Slice | None:
    if not pairs:
        return None
    return Slice(n=len(pairs), log_loss=log_loss(pairs), brier=brier_score(pairs))


def evaluate(
    label: str,
    predictions: list[GamePrediction],
    close_mkt: dict[str, float],
    open_spreads: dict[str, float],
    *,
    sigma: float,
    weights: dict[str, float] = WEIGHTS,
    open_blend_weight: float = 0.0,
    open_blend_max_week: int = 4,
    open_prior_max_week: int = OPEN_MARKET_PRIOR_MAX_WEEK,
    open_prior_weight: float = OPEN_MARKET_PRIOR_WEIGHT,
) -> dict:
    ens_all: list[tuple[float, bool]] = []
    mkt_all: list[tuple[float, bool]] = []
    ens_w12: list[tuple[float, bool]] = []
    mkt_w12: list[tuple[float, bool]] = []
    ens_w34: list[tuple[float, bool]] = []
    mkt_w34: list[tuple[float, bool]] = []
    ens_w14: list[tuple[float, bool]] = []
    mkt_w14: list[tuple[float, bool]] = []
    ens_late: list[tuple[float, bool]] = []
    mkt_late: list[tuple[float, bool]] = []

    for p in predictions:
        p_mkt = close_mkt.get(p.game_id)
        if p_mkt is None:
            continue
        p_ens = _ensemble_raw_prob(
            p,
            sigma,
            weights,
            open_spreads,
            open_prior_max_week=open_prior_max_week,
            open_prior_weight=open_prior_weight,
            open_blend_max_week=open_blend_max_week,
            open_blend_weight=open_blend_weight,
        )
        if p_ens is None:
            continue
        pair = (p_ens, p.home_won)
        mpair = (p_mkt, p.home_won)
        ens_all.append(pair)
        mkt_all.append(mpair)
        if p.week <= 2:
            ens_w12.append(pair)
            mkt_w12.append(mpair)
        if 3 <= p.week <= 4:
            ens_w34.append(pair)
            mkt_w34.append(mpair)
        if p.week <= 4:
            ens_w14.append(pair)
            mkt_w14.append(mpair)
        if p.week >= 5:
            ens_late.append(pair)
            mkt_late.append(mpair)

    def pack(ens, mkt):
        e, m = _score(ens), _score(mkt)
        return {
            "n": e.n if e else 0,
            "ens_ll": e.log_loss if e else None,
            "ens_brier": e.brier if e else None,
            "mkt_ll": m.log_loss if m else None,
            "mkt_brier": m.brier if m else None,
            "gap": (e.log_loss - m.log_loss) if e and m else None,
        }

    full = pack(ens_all, mkt_all)
    return {
        "label": label,
        "open_blend_weight": open_blend_weight,
        "open_blend_max_week": open_blend_max_week,
        "sigma": sigma,
        "weeks_1_2": pack(ens_w12, mkt_w12),
        "weeks_3_4": pack(ens_w34, mkt_w34),
        "weeks_1_4": pack(ens_w14, mkt_w14),
        "weeks_5_plus": pack(ens_late, mkt_late),
        "full": full,
        "beat_market_full": (
            full["ens_ll"] is not None
            and full["mkt_ll"] is not None
            and full["ens_ll"] < full["mkt_ll"]
        ),
        # Ship heuristic: W3–4 gap improves vs baseline, W5+ gap not worse
        # by more than 1e-4, full gap improves or holds. Filled by main().
        "ship": None,
    }


def _fmt_gap(g: float | None) -> str:
    if g is None:
        return "n/a"
    return f"{g:+.4f}"


def main() -> int:
    db = ROOT / "data" / "cfb.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    print("== walk-forward (skip_logit, allow_prior_only=False) ==", flush=True)
    t0 = time.time()
    base = run_walk_forward(conn, list(SEASONS), skip_logit=True, allow_prior_only=False)
    print(
        f"  preds={len(base.predictions)} skip_hist={base.skipped_insufficient_history} "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )

    residuals = [p.actual_margin - p.predicted_margin for p in base.predictions]
    sigma = calibrate_sigma(residuals) if len(residuals) >= 2 else 16.5

    # Week-1 unlock (production path) — appended after sigma fit.
    unlocked = _week1_open_prior_predictions(
        conn,
        SEASONS,
        sigma_0=sigma,
        existing_ids={p.game_id for p in base.predictions},
    )
    predictions = list(base.predictions) + unlocked
    print(
        f"sigma_0={sigma:.3f} week1_unlocked={len(unlocked)} total_preds={len(predictions)}",
        flush=True,
    )

    all_ids = [p.game_id for p in predictions]
    close_mkt = _load_market_home_probs(conn, all_ids, min_books=1)
    open_spreads = load_open_home_spreads(conn, all_ids)
    print(
        f"close ML coverage={len(close_mkt)}/{len(all_ids)} "
        f"open spread coverage={len(open_spreads)}/{len(all_ids)}",
        flush=True,
    )

    candidates: list[tuple[str, float, int]] = []
    for w in BLEND_WEIGHTS:
        label = "baseline_w12_replace_no_w34" if w == 0.0 else f"w34_blend_{w:.2f}"
        candidates.append((label, w, 4))
    for w in WEEK4_ONLY_WEIGHTS:
        # Soft blend only on week 4: set prior_max=2, blend_max=4 but apply
        # weight only when week==4 by using blend_max=4 with a custom path:
        # open_prior_weight_for_week applies blend for weeks 3..blend_max.
        # For week-4-only we pass blend_max_week=4 and fake prior_max=3 so
        # week 3 stays model-only while week 4 soft-blends. W1–2 still use
        # production prior via open_prior_max_week=2.
        # Actually: if prior_max=2 and blend_max=4, weeks 3–4 get blend.
        # For week-4-only: prior_max=3 (with prior_weight=0 for week 3? No —
        # that would break W1–2 if prior_max becomes 3 with weight 1).
        #
        # Clean approach: prior_max=2 (W1–2 replace), blend_max=4,
        # but we need week 3 = 0 and week 4 = w. The helper doesn't support
        # a min week. Use blend_max_week=4 with a sentinel: pass
        # open_blend_max_week=4 and open_prior_max_week=3 with
        # open_prior_weight=1.0 would REPLACE week 3 — bad.
        #
        # Instead: evaluate week-4-only via open_blend_max_week=4 and a
        # special open_blend_min by calling with blend_weight=w but
        # overriding via a second path: set prior_max=2, and use
        # blend_max=4 with weight w only after zeroing week-3 in a
        # custom evaluate branch.
        #
        # Simplest: pass open_blend_max_week=4 and open_blend_weight=w,
        # AND open_prior_max_week=2, but add open_blend_min_week support.
        # For this spike without expanding API further, emulate week-4-only
        # by setting blend_max_week=4 and using a patched weight schedule
        # via open_prior_max_week=2 + temporarily treating week 3 as
        # "later" by setting blend_max_week=4 and prior covering 1-3 with
        # weight 1 only for <=2... helper already does that.
        #
        # Week-4-only emulation: call evaluate with blend_max_week=4 and
        # blend_weight=w, but zero out week-3 blends by setting
        # open_blend_max_week=4 and using a custom label evaluated with
        # open_blend_weight applied only when week==4 — handled below
        # with open_blend_max_week=4 and a hack: prior_max_week=2,
        # blend_max_week=4, and we pass blend_weight=0 for the shared
        # path then... just evaluate with blend_max_week=4 after
        # temporarily mapping week-3 games' open spreads away.
        candidates.append((f"w4_only_blend_{w:.2f}", w, 4))  # marked; see below

    results = []
    baseline_w34_gap = None
    baseline_late_gap = None
    baseline_full_gap = None

    for label, w, blend_max in candidates:
        kwargs: dict = {
            "open_blend_weight": w,
            "open_blend_max_week": blend_max,
        }
        # Week-4-only: drop open spreads for week-3 games so blend can't fire.
        spreads = open_spreads
        if label.startswith("w4_only_"):
            week3_ids = {p.game_id for p in predictions if p.week == 3}
            spreads = {
                gid: sp
                for gid, sp in open_spreads.items()
                if gid not in week3_ids
            }
            # W1–2 still need their opens — week3_ids only removes week 3.
            # Re-add nothing; W1–2 / W4 opens remain.

        row = evaluate(
            label,
            predictions,
            close_mkt,
            spreads,
            sigma=sigma,
            **kwargs,
        )
        results.append(row)
        w34, w14, late, full = (
            row["weeks_3_4"],
            row["weeks_1_4"],
            row["weeks_5_plus"],
            row["full"],
        )
        if w == 0.0 and label.startswith("baseline"):
            baseline_w34_gap = w34["gap"]
            baseline_late_gap = late["gap"]
            baseline_full_gap = full["gap"]
        print(
            f"{label:32s} W3-4 n={w34['n']:4d} ens={w34['ens_ll']} "
            f"mkt={w34['mkt_ll']} gap={_fmt_gap(w34['gap'])} | "
            f"W1-4 gap={_fmt_gap(w14['gap'])} | W5+ gap={_fmt_gap(late['gap'])} | "
            f"full gap={_fmt_gap(full['gap'])}",
            flush=True,
        )

    # Ship decision vs baseline (no W3–4 blend).
    eps = 1e-4
    for row in results:
        if row["label"].startswith("baseline"):
            row["ship"] = False
            row["ship_reason"] = "baseline"
            continue
        w34_gap = row["weeks_3_4"]["gap"]
        late_gap = row["weeks_5_plus"]["gap"]
        full_gap = row["full"]["gap"]
        w34_ok = (
            w34_gap is not None
            and baseline_w34_gap is not None
            and w34_gap < baseline_w34_gap - eps
        )
        late_ok = (
            late_gap is not None
            and baseline_late_gap is not None
            and late_gap <= baseline_late_gap + eps
        )
        full_ok = (
            full_gap is not None
            and baseline_full_gap is not None
            and full_gap <= baseline_full_gap + eps
        )
        # Clear win: W3–4 improves, late not tanked, full not worse.
        ship = bool(w34_ok and late_ok and full_ok)
        row["ship"] = ship
        row["ship_reason"] = (
            f"w34_ok={w34_ok} late_ok={late_ok} full_ok={full_ok}"
        )

    # Pick best shippable by full gap, then W3–4 gap.
    shippable = [r for r in results if r["ship"]]
    winner = None
    if shippable:
        winner = min(
            shippable,
            key=lambda r: (
                r["full"]["gap"] if r["full"]["gap"] is not None else 9.0,
                r["weeks_3_4"]["gap"] if r["weeks_3_4"]["gap"] is not None else 9.0,
            ),
        )

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "spike_open_blend_w34_2023_2025.json"
    payload = {
        "sigma": sigma,
        "n_base_preds": len(base.predictions),
        "n_week1_unlocked": len(unlocked),
        "n_close_mkt": len(close_mkt),
        "n_open_spreads": len(open_spreads),
        "baseline_w34_gap": baseline_w34_gap,
        "baseline_late_gap": baseline_late_gap,
        "baseline_full_gap": baseline_full_gap,
        "winner": winner["label"] if winner else None,
        "winner_blend_weight": winner["open_blend_weight"] if winner else None,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)
    if winner:
        print(
            f"WINNER: {winner['label']} w={winner['open_blend_weight']} "
            f"W3-4 gap={_fmt_gap(winner['weeks_3_4']['gap'])} "
            f"full gap={_fmt_gap(winner['full']['gap'])}",
            flush=True,
        )
    else:
        print("NO CLEAR WINNER — leave OPEN_MARKET_BLEND_WEIGHT=0.0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
