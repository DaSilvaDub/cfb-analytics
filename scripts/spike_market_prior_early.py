#!/usr/bin/env python3
"""Walk-forward spike: opening-line / as-of<kickoff market prior for weeks 1–2.

Leakage rule: use CFBD historical *open* spread only (synthetic stamp
kickoff-7d from spreadOpen; also line_movement.open_line). Never close ML
or close spread for the same game's prediction. Close ML remains the
scoring baseline via market_consensus.

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
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfb_analytics.backtest.calibration import calibrate_sigma, margin_to_prob
from cfb_analytics.backtest.harness import GamePrediction, run_walk_forward
from cfb_analytics.backtest.metrics import brier_score, log_loss
from cfb_analytics.backtest.moneyline import (
    EARLY_SEASON_RIDGE_SIGMA_SCALE,
    _load_market_home_probs,
    _member_probs,
)
from cfb_analytics.models.ensemble import pool_probabilities
from cfb_analytics.features.open_market_prior import (
    OPEN_MARKET_PRIOR_MAX_WEEK as PROD_MAX_WEEK,
    OPEN_MARKET_PRIOR_WEIGHT as PROD_WEIGHT,
)

SEASONS = (2023, 2024, 2025)
# Production defaults (shipped): weeks≤2 replace weight=1.0
assert PROD_MAX_WEEK == 2 and PROD_WEIGHT == 1.0
# Production mix without logit for speed (renorm ridge/elo).
WEIGHTS = {"ridge": 0.15 / 0.95, "internal_elo": 0.80 / 0.95}
OPEN_LOOKBACK_DAYS_LO = 6.0
OPEN_LOOKBACK_DAYS_HI = 8.0


@dataclass
class Slice:
    n: int
    log_loss: float
    brier: float


def _score(pairs: list[tuple[float, bool]]) -> Slice | None:
    if not pairs:
        return None
    return Slice(n=len(pairs), log_loss=log_loss(pairs), brier=brier_score(pairs))


def load_open_home_spreads(conn: sqlite3.Connection, game_ids: list[str]) -> dict[str, float]:
    """Median HOME open spread across books at the synthetic open stamp.

    Open stamp is kickoff-7d (see ingest/cfbd_lines_historical.py). Rows are
    source=cfbd_historical SPREAD HOME with captured ≈ kickoff-7d.
    Fallback: line_movement.open_line (CFBD spreadOpen) when present.
    """
    if not game_ids:
        return {}
    out: dict[str, float] = {}
    # Chunk to keep SQL sane
    for i in range(0, len(game_ids), 400):
        chunk = game_ids[i : i + 400]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT o.game_id, o.line
            FROM odds_snapshots o
            JOIN games g ON g.game_id = o.game_id
            WHERE o.game_id IN ({ph})
              AND o.source = 'cfbd_historical'
              AND o.market = 'SPREAD' AND o.side = 'HOME'
              AND o.line IS NOT NULL
              AND (julianday(g.kickoff_utc) - julianday(o.captured_utc))
                    BETWEEN ? AND ?
            """,
            (*chunk, OPEN_LOOKBACK_DAYS_LO, OPEN_LOOKBACK_DAYS_HI),
        ).fetchall()
        by_game: dict[str, list[float]] = defaultdict(list)
        for gid, line in rows:
            by_game[str(gid)].append(float(line))
        for gid, lines in by_game.items():
            out[gid] = float(median(lines))

        missing = [g for g in chunk if g not in out]
        if missing:
            ph2 = ",".join("?" * len(missing))
            lm = conn.execute(
                f"""
                SELECT game_id, open_line FROM line_movement
                WHERE market = 'SPREAD' AND side = 'HOME'
                  AND open_line IS NOT NULL
                  AND game_id IN ({ph2})
                """,
                missing,
            ).fetchall()
            by_lm: dict[str, list[float]] = defaultdict(list)
            for gid, line in lm:
                by_lm[str(gid)].append(float(line))
            for gid, lines in by_lm.items():
                if gid not in out:
                    out[gid] = float(median(lines))
    return out


def open_spread_to_home_prob(spread: float, sigma: float) -> float:
    """HOME spread → P(home win). Favored home at -14 => margin +14."""
    return margin_to_prob(-float(spread), sigma)


def _ens_prob(
    pred: GamePrediction,
    sigma: float,
    weights: dict[str, float],
    *,
    early_sigma: bool = True,
) -> float | None:
    week_sigma = sigma
    if early_sigma and pred.week <= 4:
        week_sigma = sigma * EARLY_SEASON_RIDGE_SIGMA_SCALE
    member = _member_probs(pred, week_sigma)
    available = {k: v for k, v in weights.items() if v > 0 and k in member}
    if not available or sum(available.values()) <= 0:
        return None
    return pool_probabilities(member, available)


def _logit_blend(p_model: float, p_mkt: float, w_mkt: float) -> float:
    """Blend in log-odds: w_mkt weight on market, rest on model."""
    w_mkt = max(0.0, min(1.0, w_mkt))
    if w_mkt <= 0:
        return p_model
    if w_mkt >= 1:
        return p_mkt
    # pool_probabilities style
    return pool_probabilities(
        {"model": p_model, "open_mkt": p_mkt},
        {"model": 1.0 - w_mkt, "open_mkt": w_mkt},
    )


def _margin_shrink(model_margin: float, open_spread: float, alpha: float) -> float:
    open_margin = -float(open_spread)
    a = max(0.0, min(1.0, alpha))
    return (1.0 - a) * model_margin + a * open_margin


@dataclass
class Week1Synth:
    """Synthetic week-1 prediction from open market only (no ridge/Elo)."""

    game_id: str
    season: int
    week: int
    home_won: bool
    open_spread: float


def load_week1_open_games(
    conn: sqlite3.Connection, seasons: tuple[int, ...]
) -> list[Week1Synth]:
    """Completed week-1 games with an open HOME spread (for unlock experiments)."""
    rows = conn.execute(
        f"""
        SELECT g.game_id, g.season, g.week, g.home_points, g.away_points
        FROM games g
        WHERE g.season IN ({",".join("?" * len(seasons))})
          AND g.season_type = 'regular'
          AND g.week = 1
          AND g.completed = 1
          AND g.home_points IS NOT NULL AND g.away_points IS NOT NULL
        """,
        seasons,
    ).fetchall()
    ids = [str(r[0]) for r in rows]
    opens = load_open_home_spreads(conn, ids)
    out: list[Week1Synth] = []
    for r in rows:
        gid = str(r[0])
        if gid not in opens:
            continue
        out.append(
            Week1Synth(
                game_id=gid,
                season=int(r[1]),
                week=int(r[2]),
                home_won=int(r[3]) > int(r[4]),
                open_spread=opens[gid],
            )
        )
    return out


def evaluate(
    label: str,
    predictions: list[GamePrediction],
    close_mkt: dict[str, float],
    open_spreads: dict[str, float],
    *,
    sigma: float,
    weights: dict[str, float] = WEIGHTS,
    # Score-time levers (weeks 1–2 unless noted):
    open_blend_w: float = 0.0,  # log-odds blend weight on open prob
    open_margin_alpha: float = 0.0,  # shrink predicted_margin toward -open
    open_replace: bool = False,  # replace ens with open when available
    week_max: int = 2,
    week1_synths: list[Week1Synth] | None = None,
    matched_ids: set[str] | None = None,
) -> dict:
    ens_all: list[tuple[float, bool]] = []
    mkt_all: list[tuple[float, bool]] = []
    open_all: list[tuple[float, bool]] = []

    ens_w12: list[tuple[float, bool]] = []
    mkt_w12: list[tuple[float, bool]] = []
    open_w12: list[tuple[float, bool]] = []

    ens_w14: list[tuple[float, bool]] = []
    mkt_w14: list[tuple[float, bool]] = []

    ens_late: list[tuple[float, bool]] = []
    mkt_late: list[tuple[float, bool]] = []

    seen: set[str] = set()

    def apply_open(pred: GamePrediction, p_ens: float) -> float | None:
        sp = open_spreads.get(pred.game_id)
        if sp is None:
            return p_ens
        p_open = open_spread_to_home_prob(sp, sigma)
        early = pred.week <= week_max
        if not early:
            return p_ens
        if open_replace:
            return p_open
        if open_margin_alpha > 0:
            m = _margin_shrink(pred.predicted_margin, sp, open_margin_alpha)
            # Rebuild ens with shrunk ridge margin; keep elo member.
            week_sigma = (
                sigma * EARLY_SEASON_RIDGE_SIGMA_SCALE if pred.week <= 4 else sigma
            )
            # Temporary: swap margin via member rebuild
            from dataclasses import replace

            adj = replace(pred, predicted_margin=m)
            p_adj = _ens_prob(adj, sigma, weights)
            if p_adj is None:
                return p_ens
            if open_blend_w > 0:
                return _logit_blend(p_adj, p_open, open_blend_w)
            return p_adj
        if open_blend_w > 0:
            return _logit_blend(p_ens, p_open, open_blend_w)
        return p_ens

    for p in predictions:
        if matched_ids is not None and p.game_id not in matched_ids:
            continue
        p_mkt = close_mkt.get(p.game_id)
        if p_mkt is None:
            continue
        p_ens = _ens_prob(p, sigma, weights)
        if p_ens is None:
            continue
        p_final = apply_open(p, p_ens)
        if p_final is None:
            continue
        seen.add(p.game_id)
        pair = (p_final, p.home_won)
        mpair = (p_mkt, p.home_won)
        ens_all.append(pair)
        mkt_all.append(mpair)
        sp = open_spreads.get(p.game_id)
        if sp is not None:
            open_all.append((open_spread_to_home_prob(sp, sigma), p.home_won))

        if p.week <= 2:
            ens_w12.append(pair)
            mkt_w12.append(mpair)
            if sp is not None:
                open_w12.append((open_spread_to_home_prob(sp, sigma), p.home_won))
        if p.week <= 4:
            ens_w14.append(pair)
            mkt_w14.append(mpair)
        if p.week >= 5:
            ens_late.append(pair)
            mkt_late.append(mpair)

    # Optional week-1 unlock from open-only synths (not in walk-forward).
    week1_added = 0
    if week1_synths:
        for s in week1_synths:
            if s.game_id in seen:
                continue
            if matched_ids is not None and s.game_id not in matched_ids:
                continue
            p_mkt = close_mkt.get(s.game_id)
            if p_mkt is None:
                continue
            p_open = open_spread_to_home_prob(s.open_spread, sigma)
            pair = (p_open, s.home_won)
            mpair = (p_mkt, s.home_won)
            ens_all.append(pair)
            mkt_all.append(mpair)
            open_all.append(pair)
            ens_w12.append(pair)
            mkt_w12.append(mpair)
            open_w12.append(pair)
            ens_w14.append(pair)
            mkt_w14.append(mpair)
            week1_added += 1

    def pack(ens, mkt, open_p=None):
        e, m = _score(ens), _score(mkt)
        out = {
            "n": e.n if e else 0,
            "ens_ll": e.log_loss if e else None,
            "ens_brier": e.brier if e else None,
            "mkt_ll": m.log_loss if m else None,
            "mkt_brier": m.brier if m else None,
            "gap": (e.log_loss - m.log_loss) if e and m else None,
        }
        if open_p is not None:
            o = _score(open_p)
            out["open_ll"] = o.log_loss if o else None
            out["open_n"] = o.n if o else 0
        return out

    return {
        "label": label,
        "sigma": sigma,
        "week1_added": week1_added,
        "full": pack(ens_all, mkt_all, open_all),
        "weeks_1_2": pack(ens_w12, mkt_w12, open_w12),
        "weeks_1_4": pack(ens_w14, mkt_w14),
        "weeks_5_plus": pack(ens_late, mkt_late),
        "beat_market_full": (
            (_score(ens_all).log_loss < _score(mkt_all).log_loss)
            if ens_all and mkt_all
            else None
        ),
    }


def main() -> int:
    db = ROOT / "data" / "cfb.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    print("== baseline walk-forward (skip_logit) ==", flush=True)
    t0 = time.time()
    base = run_walk_forward(conn, list(SEASONS), skip_logit=True, allow_prior_only=False)
    print(
        f"  preds={len(base.predictions)} skip_hist={base.skipped_insufficient_history} "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )

    print("== prior_only walk-forward ==", flush=True)
    t0 = time.time()
    prior = run_walk_forward(conn, list(SEASONS), skip_logit=True, allow_prior_only=True)
    print(
        f"  preds={len(prior.predictions)} skip_hist={prior.skipped_insufficient_history} "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )

    all_ids = {p.game_id for p in base.predictions} | {p.game_id for p in prior.predictions}
    week1_synths = load_week1_open_games(conn, SEASONS)
    all_ids |= {s.game_id for s in week1_synths}

    close_mkt = _load_market_home_probs(conn, sorted(all_ids), min_books=1)
    open_spreads = load_open_home_spreads(conn, sorted(all_ids))
    print(
        f"close ML coverage={len(close_mkt)}/{len(all_ids)} "
        f"open spread coverage={len(open_spreads)}/{len(all_ids)} "
        f"week1_open_synths={len(week1_synths)}",
        flush=True,
    )

    residuals = [p.actual_margin - p.predicted_margin for p in base.predictions]
    sigma = calibrate_sigma(residuals) if len(residuals) >= 2 else 16.5
    print(f"sigma_0={sigma:.3f}", flush=True)

    # Matched baseline game set (no week-1 unlock) for fair compare
    base_ids = {
        p.game_id
        for p in base.predictions
        if p.game_id in close_mkt
    }

    candidates: list[tuple[str, list[GamePrediction], dict]] = [
        ("baseline_early_sigma", base.predictions, {}),
        ("open_alone_w12_replace", base.predictions, {"open_replace": True}),
        ("open_blend_w12_0.25", base.predictions, {"open_blend_w": 0.25}),
        ("open_blend_w12_0.50", base.predictions, {"open_blend_w": 0.50}),
        ("open_blend_w12_0.75", base.predictions, {"open_blend_w": 0.75}),
        ("open_blend_w12_1.00", base.predictions, {"open_blend_w": 1.0}),
        ("open_margin_a0.50_w12", base.predictions, {"open_margin_alpha": 0.50}),
        ("open_margin_a0.75_w12", base.predictions, {"open_margin_alpha": 0.75}),
        ("open_margin_a1.00_w12", base.predictions, {"open_margin_alpha": 1.0}),
        (
            "open_margin_a0.75+blend0.25",
            base.predictions,
            {"open_margin_alpha": 0.75, "open_blend_w": 0.25},
        ),
        # Week-1 unlock: open-only for missing week-1 + blend on W2
        (
            "w1_open_unlock+blend0.75",
            base.predictions,
            {"open_blend_w": 0.75, "week1_synths": week1_synths},
        ),
        (
            "w1_open_unlock+replace",
            base.predictions,
            {"open_replace": True, "week1_synths": week1_synths},
        ),
        # prior_only bare (known fail) vs prior_only replaced by open on W1-2
        ("prior_only_bare", prior.predictions, {}),
        (
            "prior_only+open_replace_w12",
            prior.predictions,
            {"open_replace": True},
        ),
        (
            "prior_only+open_blend0.75",
            prior.predictions,
            {"open_blend_w": 0.75},
        ),
    ]

    results = []
    for label, preds, kwargs in candidates:
        # For unlock variants, do not restrict to base_ids (expanded set).
        use_matched = "week1_synths" not in kwargs and not label.startswith("prior_only")
        row = evaluate(
            label,
            preds,
            close_mkt,
            open_spreads,
            sigma=sigma,
            matched_ids=base_ids if use_matched else None,
            **kwargs,
        )
        results.append(row)
        w12, w14, full = row["weeks_1_2"], row["weeks_1_4"], row["full"]
        print(
            f"{label:32s} W1-2 n={w12['n']:4d} ens={w12['ens_ll']} mkt={w12['mkt_ll']} "
            f"gap={w12['gap']} | W1-4 gap={w14['gap']} | full gap={full['gap']} "
            f"w1+={row['week1_added']}",
            flush=True,
        )

    # Open-alone diagnostic on baseline overlap weeks 1-2 / 1-4 / full
    diag = evaluate(
        "diag_open_vs_close_on_base",
        base.predictions,
        close_mkt,
        open_spreads,
        sigma=sigma,
        open_replace=True,
        week_max=99,  # replace all weeks with open where available
        matched_ids=base_ids,
    )
    results.append(diag)
    print(
        f"{'diag_open_all_weeks':32s} full open_ll={diag['full'].get('open_ll')} "
        f"ens(open)={diag['full']['ens_ll']} mkt={diag['full']['mkt_ll']} "
        f"gap={diag['full']['gap']}",
        flush=True,
    )

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "spike_market_prior_early_2023_2025.json"
    payload = {
        "sigma": sigma,
        "n_base_preds": len(base.predictions),
        "n_prior_preds": len(prior.predictions),
        "n_close_mkt": len(close_mkt),
        "n_open_spreads": len(open_spreads),
        "n_week1_synths": len(week1_synths),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
