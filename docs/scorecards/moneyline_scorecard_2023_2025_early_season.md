# Moneyline scorecard — weeks 1–4 early-season ratings spike (2023–2025)

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~03:10 EDT** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| Spike script | `scripts/spike_early_season_ratings.py` (raw JSON under `artifacts/`, gitignored) |
| Same-game set | CFBD historical close ML, `min_books=1`, overlap with walk-forward preds |
| Promotion | still **shadow** — this spike does **not** flip `beat_market` |

## Root cause (weeks 1–4 gap)

From Track B: weeks 1–4 ensemble gap vs market ≈ **+0.0505** (ridge LL ~0.67 vs market ~0.49).

**1. Week-1 blackout (structural).** `backtest/harness.py` refits ridge once per week and **skips the entire week** when `RidgeRatings.status != 'active'`. Ridge (and internal Elo) both require `min_games=30` **this-season** games before becoming active (`models/ridge.py`, `models/elo.py`). Week 1 has `history_before=0` → ~290 games / 3 seasons skipped (`skipped_insufficient_history`). Market prices week 1; the model is absent from the promote overlap for those games.

**2. Priors exist but never produce week-1 predictions.** Elo has a full FBS preseason seed (`features/elo_internal.py` / `preseason_rating`); ridge has shrinkage prior (`models/shrinkage.py` `k=4`, `features/team_ratings.py` blend of prev-season final + talent + returning PPA). Both are gated behind the in-season `min_games` floor, so week 1 never uses them.

**3. Weeks 2–4 ridge overconfidence.** Once ~30+ week-1 results unlock the fit, per-team `n≈1–3` with shrinkage `k=4` still puts substantial weight on a noisy in-season solve. Global `sigma_0` (~19.7 pts, fit on full-season residuals) converts those margins via `Phi(M/sigma)` as if mid-season noise — ridge weeks 1–4 LL **0.6689** on the matched overlap.

Key files: `backtest/harness.py`, `features/team_ratings.py`, `models/shrinkage.py`, `features/preseason.py`, `features/elo_internal.py`, `models/elo.py` (`min_games`), `backtest/moneyline.py` (margin→prob).

Already-failed weight spikes (do not re-ship): drop logit, WF ensemble weights, drop-ridge early, elo-only early — see `moneyline_scorecard_2023_2025_market_blend.md`.

## Spike table (2023–2025 walk-forward, same-game market)

Ensemble weights held at production mix **without logit** for speed (`ridge/elo` renorm of 0.15/0.80). Baseline matched the promote overlap shape (n=2103 full / n=462 weeks 1–4). Directional results rechecked; early-sigma ships into the full (with-logit) scoring path.

| Candidate | W1–4 ens LL | W1–4 mkt LL | W1–4 gap | Full ens LL | Full gap | Ship? |
|---|---:|---:|---:|---:|---:|---|
| **baseline** | 0.5437 | 0.4899 | **+0.0538** | 0.5558 | +0.0271 | — |
| early_sigma ×1.35 | 0.5400 | 0.4899 | +0.0502 | 0.5550 | +0.0263 | near-tie |
| **early_sigma ×1.5** | **0.5400** | 0.4899 | **+0.0501** | **0.5550** | **+0.0263** | **Y (small)** |
| early_sigma ×2.0 | 0.5416 | 0.4899 | +0.0518 | 0.5553 | +0.0267 | N (past elbow) |
| early_ridge_w ×0.25 | 0.5409 | 0.4899 | +0.0510 | 0.5552 | +0.0265 | N (≤ sigma) |
| early_ridge_w ×0 | 0.5549 | 0.5019* | +0.0529 | 0.5583 | +0.0265 | N (known fail) |
| shrink `k=12` | 0.5442 | 0.4899 | +0.0543 | 0.5591 | +0.0305 | N (hurts late) |
| shrink `k=20` | 0.5441 | 0.4899 | +0.0543 | 0.5603 | +0.0316 | N |
| prior_only (unlock wk1) | 0.5468† | 0.4738† | **+0.0730** | 0.5559 | +0.0345 | **N** |
| prior_only + k=12 | 0.5449† | 0.4738† | +0.0710 | 0.5582 | +0.0367 | N |
| prior_only + sigma×1.5 | 0.5396† | 0.4738† | +0.0658 | 0.5539 | +0.0325 | N |

\* Drop-ridge early changes the overlap slightly (games with only ridge member).  
† Expanded overlap includes **week 1** (n_W1–4=625, week1 overlap=163). On the **matched** baseline game set, prior_only ≈ baseline (W1–4 gap +0.0523).

### Week-1 under prior_only (honest)

| Slice | n | ens LL | mkt LL | gap |
|---|---:|---:|---:|---:|
| Week 1 only (prior-only unlock) | 163 | 0.5601 | **0.4284** | **+0.1316** |
| Ridge week 1 | 163 | 0.6836 | 0.4284 | +0.255 |

Unlocking week 1 without a market-competitive prior **widens** the early-season hole. Prev-season + talent + returning PPA is not close to closing-line skill on week 1.

### Ridge member alone (matched W1–4)

| Candidate | ridge LL |
|---|---:|
| baseline | 0.6689 |
| early_sigma ×1.5 | **0.5731** |
| shrink k=12 | 0.6533 |

Sigma inflation is mostly a **calibration** fix for early ridge, not a new information source — which is why ensemble movement is small (ridge weight ≈ 0.15).

## What shipped

**`EARLY_SEASON_RIDGE_SIGMA_SCALE = 1.5` for weeks ≤ 4** in `backtest/moneyline.py` (`ridge_sigma_for_week` / `_ridge_prob`), applied to ridge-alone metrics and the ridge ensemble member. Weeks 5+ unchanged. Leakage-safe (week index only).

**Opt-in infrastructure (default off, not enabled in promote):**
- `fit_elo(..., allow_prior_only=True)` — seed-only active book below `min_games`
- `features/team_ratings.prior_only_ratings` — synthesize ridge book from shrinkage prior
- `run_walk_forward(..., allow_prior_only=True)` — unlock week 1 when priors exist

Left **off** in production backtest: week-1 priors lose to market by ~13 LL points.

## Promote gates (expected; not re-run full logit promote in this spike)

| Gate | Status |
|---|---|
| Pure-model OOS logloss beat market | **still fail** (full gap ~0.026, not closed) |
| Calibration gap ≤ 0.03 | unchanged / still fail (~0.037) |
| Median CLV ≥ 0 | unchanged (model-side) |
| `market_weight_floor` | unchanged (0.75); still require pure model beat before lowering |

## Recommended next action

1. **Do not chase more shrinkage-k / ensemble-weight early tweaks** — measured, failed or ≤ noise.
2. **Next early-season lever:** week-1 / week-2 **opening-line prior** (or a market-derived preseason power rating) as an ensemble member or ridge seed — must use **open** (or as-of &lt; kickoff) quotes only; close is leakage. Local `odds_snapshots` / `line_movement` may already carry open vs current.
3. After that (or in parallel on a separate track): **away-favorite / HFA** spike (#2 skill gap).
4. Stop if open-line prior also fails OOS — then early-season may be an irreducible market-info gap until QB / roster features land.

## Files touched

- `cfb_analytics/backtest/moneyline.py` — early ridge sigma scale (shipped)
- `cfb_analytics/models/elo.py` — `allow_prior_only` (opt-in)
- `cfb_analytics/features/elo_internal.py` — pass-through
- `cfb_analytics/features/team_ratings.py` — `prior_only_ratings`
- `cfb_analytics/backtest/harness.py` — prior-only week unlock + `skip_logit` (spike aid)
- `tests/test_early_season_sigma.py`, `tests/test_elo.py`
- `scripts/spike_early_season_ratings.py`
