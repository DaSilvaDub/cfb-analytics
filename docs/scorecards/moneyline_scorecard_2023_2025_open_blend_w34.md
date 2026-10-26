# Moneyline scorecard — weeks 3–4 soft open-market blend (2023–2025)

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~04:00 ET** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| Spike script | `scripts/spike_open_blend_w34.py` (raw JSON under `artifacts/`) |
| Same-game set | CFBD historical **close** ML (`min_books=1`) vs model; prior uses **open** spread only |
| Promotion | still **shadow** — does **not** flip `beat_market` |

## Design

**Weeks 1–2 (unchanged):** full open replace (`OPEN_MARKET_PRIOR_WEIGHT=1.0`) + week-1 open unlock when ridge blacked out.

**Weeks 3–4 (this spike):** soft blend `P = pool(model, open_mkt; w)` with **w &lt; 1** via `OPEN_MARKET_BLEND_WEIGHT`. Never full-replace weeks 3–4. Never use close as same-game prior.

Injection: `features/open_market_prior.py` → `open_prior_weight_for_week` → `backtest/moneyline.py` `_ensemble_raw_prob`.

## Spike table (2023–2025 walk-forward)

Baseline = shipped W1–2 replace + early ridge sigma ×1.5 + week-1 unlock; **no** W3–4 blend. `sigma_0=19.695`. Matched close-ML overlap (n_full≈2270 with unlock).

| Candidate | W3–4 ens LL | W3–4 mkt LL | W3–4 gap | W1–4 gap | W5+ gap | Full ens LL | Full gap | Ship? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_w12_replace_no_w34` | 0.5180 | 0.4708 | **+0.0472** | +0.0320 | +0.0196 | 0.5439 | **+0.0230** | — |
| `w34_blend_0.25` | 0.5050 | 0.4708 | **+0.0342** | +0.0254 | +0.0196 | 0.5420 | **+0.0212** | Y |
| `w34_blend_0.50` | 0.4953 | 0.4708 | **+0.0244** | +0.0205 | +0.0196 | 0.5407 | **+0.0198** | Y |
| `w34_blend_0.75` | 0.4889 | 0.4708 | **+0.0181** | +0.0172 | +0.0196 | 0.5398 | **+0.0189** | **Y (winner)** |
| `w4_only_blend_0.25` | 0.5116 | 0.4708 | **+0.0408** | +0.0288 | +0.0196 | 0.5430 | **+0.0221** | near |
| `w4_only_blend_0.50` | 0.5070 | 0.4708 | **+0.0361** | +0.0264 | +0.0196 | 0.5423 | **+0.0215** | near |
| `w4_only_blend_0.75` | 0.5042 | 0.4708 | **+0.0334** | +0.0250 | +0.0196 | 0.5419 | **+0.0211** | near |

W1–2 gap stayed **+0.0164** (n=309) on every arm — W3–4 knob does not regress the shipped replace.

### Notes

- Monotone improvement in W3–4 / W1–4 / full as `w` rises 0 → 0.75; W5+ gap **unchanged** (+0.0196) because blend does not fire after week 4.
- Week-4-only arms help less than weeks 3–4 (most of the remaining early gap is in week 3).
- Full-season ens still trails close market (gap +0.0189 at winner) — `beat_market=false`.
- Open≈close caveat still applies on blended rows; do not claim pure-model skill from open-seeded early weeks alone.

## What shipped

- `OPEN_MARKET_BLEND_MAX_WEEK=4`, `OPEN_MARKET_BLEND_WEIGHT=0.75`
- `open_prior_weight_for_week()` + `_ensemble_raw_prob(..., open_blend_*)`
- `scripts/spike_open_blend_w34.py`, tests in `tests/test_open_market_prior.py`

## Promote gates

| Gate | Status |
|---|---|
| Pure-model OOS logloss beat market | **still fail** (full gap ~0.0189 even with W3–4 blend 0.75) |
| Calibration gap ≤ 0.03 | unchanged / still fail (~0.037 prior evidence) |
| Median CLV ≥ 0 | unchanged expectation |
| `market_weight_floor` | unchanged (0.75); evidence-only |
| Status | **shadow** |

## Recommended next action

1. **Pause CFB skill spikes** for a beat — early-season open levers (W1–2 replace + W3–4 soft 0.75 + early ridge sigma) are largely exhausted; remaining full gap is weeks 5+ model skill + calibration.
2. Write a **promote-blockers one-pager** (beat_market / cal gap / floor discipline) before any more knobs.
3. Do **not** full-replace weeks 3–4, reopen Elo HFA, or hand-flip promotion.
4. Only reopen a skill lever if new data (e.g. true as-of open ML, not synthetic kickoff−7d) lands.
