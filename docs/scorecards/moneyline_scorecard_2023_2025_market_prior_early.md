# Moneyline scorecard — opening-line / as-of&lt;kickoff market prior (weeks 1–2, 2023–2025)

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~03:21 ET** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| Spike script | `scripts/spike_market_prior_early.py` (raw JSON under `artifacts/`) |
| Same-game set | CFBD historical **close** ML (`min_books=1`) vs model; prior uses **open** spread only |
| Promotion | still **shadow** — does **not** flip `beat_market` |

## Map: how open vs close join today

CFBD historical `/lines` has opening + closing *prices* but no capture clock. `cfb_analytics/ingest/cfbd_lines_historical.py` dual-stamps into `odds_snapshots` with `source=cfbd_historical`:

| Role | Synthetic stamp | Fields written |
|---|---|---|
| **open** | `kickoff − 7d` | `spreadOpen` / `overUnderOpen` only — **no moneyline open** in CFBD schema |
| **close** | `kickoff − 60s` | `spread` / `overUnder` / both moneylines |

Both stamps are strictly before kickoff (`AsOfReader` safe). **Close is still leakage for a same-game prior** when the promote baseline is close ML (open≈close → tautology).

`line_movement.open_line` / `current_line` carry CFBD open→close on the close stamp. `market_consensus` is rebuilt by `build_market_for_slate` / `AsOfReader` and therefore prefers the close stamp for scoring. Promote overlap join: `backtest/moneyline.py` `_load_market_home_probs` (latest pre-kickoff ML HOME).

**Safe prior feature:** median HOME open spread at the open stamp (`features/open_market_prior.py` → `load_open_home_spreads`), converted with `Phi(−spread / sigma_0)`. Never close spread/ML for the same game’s prediction.

Injection points considered: ridge/Elo margin shrink toward −open (failed), ensemble soft-blend / replace in W1–2 (won), week-1 unlock via open-only synth (won vs bare `prior_only`).

## Root design choice

**Weeks ≤ 2:** when an open HOME spread exists, **replace** ensemble raw P(home) with open-implied P(home) (`OPEN_MARKET_PRIOR_WEIGHT=1.0`).

**Week 1 unlock:** ridge still blacks out week 1 (`min_games=30`). Append open-only synthetic `GamePrediction`s (`predicted_margin=−open_spread`, `internal_elo_win_prob=open-implied`) so promote overlap covers week 1 without bare Elo/ridge priors (those lost to market by ~0.13 LL — prior scorecard).

Caveat: open≈close, so W1–2 rows partially borrow market info vs the close baseline. Full-season `beat_market` still fails; floor stays 0.75; shadow.

## Spike table (2023–2025 walk-forward)

Baseline = current path with early ridge sigma ×1.5, skip_logit weights renormed. Matched close-ML overlap unless noted (unlock expands n). `sigma_0=19.695`.

| Candidate | W1–2 ens LL | W1–2 mkt LL | W1–2 gap | W1–4 gap | Full ens LL | Full gap | Ship? |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_early_sigma` | 0.5857 | 0.5327 | **0.0530** | 0.0529 | 0.5556 | 0.0269 | — |
| `open_blend_w12_0.50` | 0.5576 | 0.5327 | **0.0249** | 0.0443 | 0.5537 | 0.0250 | near |
| `open_blend_w12_0.75` | 0.5489 | 0.5327 | **0.0162** | 0.0416 | 0.5531 | 0.0244 | near |
| `open_alone_w12_replace` | 0.5438 | 0.5327 | **0.0111** | 0.0400 | 0.5527 | 0.0241 | **Y** |
| `open_margin_a0.75_w12` | 0.5875 | 0.5327 | **0.0548** | 0.0535 | 0.5557 | 0.0270 | N |
| `open_margin_a1.00_w12` | 0.5899 | 0.5327 | **0.0572** | 0.0542 | 0.5559 | 0.0272 | N |
| `w1_open_unlock+replace` †n expands (+167 wk1) | 0.4893 | 0.4729 | **0.0164** | 0.0350 | 0.5447 | 0.0238 | **Y** (with replace) |
| `prior_only_bare` †expanded prior_only set | 0.5600 | 0.4770 | **0.0830** | 0.0676 | 0.5543 | 0.0328 | N (known) |
| `prior_only+open_replace_w12` †prior_only set | 0.4921 | 0.4770 | **0.0151** | 0.0345 | 0.5451 | 0.0237 | Y (equiv unlock) |
| `diag_open_vs_close_on_base` | 0.5438 | 0.5327 | **0.0111** | 0.0141 | 0.5429 | 0.0142 | diag |

### Notes

- Baseline W1–2 is almost entirely **week 2** (week 1 blackout; `week1_overlap=0`). Open-replace cuts W1–2 gap 0.0530 → 0.0111 and full gap 0.0269 → 0.0241 without hurting weeks 5+.
- Week-1 unlock + replace: W1–2 n=309, gap=0.0164, full gap=0.0238.
- **Margin shrink toward −open failed** (gaps worse than baseline) — market skill is in calibrated win probs, not in forcing open through the ridge/Elo pool.
- Open-alone all weeks vs close (diagnostic): ens LL=0.5429, mkt LL=0.5287, gap=0.0142 — open trails close slightly (expected CLV).
- Extending replace through week 4 further cuts W1–4 / full gap in a side check; scoped ship stays **weeks ≤ 2** per plan (broader replace ≈ market blend).

## What shipped

- `cfb_analytics/features/open_market_prior.py` — open spread load + Phi map; `OPEN_MARKET_PRIOR_MAX_WEEK=2`, `OPEN_MARKET_PRIOR_WEIGHT=1.0`
- `cfb_analytics/backtest/moneyline.py` — ensemble raw path applies open replace for weeks ≤ 2; week-1 open-only unlock into fit predictions (default on; `use_open_market_prior` / `unlock_week1_open_prior`)
- `tests/test_open_market_prior.py`, `scripts/spike_market_prior_early.py`

## Promote gates

| Gate | Status |
|---|---|
| Pure-model OOS logloss beat market | **still fail** (full gap ~0.024 even with open W1–2) |
| Calibration gap ≤ 0.03 | unchanged / still fail (~0.037 prior evidence) |
| Median CLV ≥ 0 | unchanged expectation (model-side; W1–2 open rows ≈ 0 CLV vs close) |
| `market_weight_floor` | unchanged (0.75); evidence-only |
| Status | **shadow** |

## Recommended next action

1. **Away-favorite / HFA skill-gap spike** (next lever after market prior).
2. Optional follow-up: week 3–4 open blend weight &lt; 1.0 (not full replace) if early gap remains after HFA.
3. Do not chase more shrinkage-k / weight-only early tweaks (already failed).
4. Do not claim `beat_market` from open-prior rows alone — they are market-seeded.

