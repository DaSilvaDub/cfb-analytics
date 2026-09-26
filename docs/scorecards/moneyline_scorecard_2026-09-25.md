# Moneyline scorecard — 2026-09-25

| Field | Value |
|---|---|
| Generated | **2026-09-25 21:50 EDT** |
| Code (`master`) | `3ccbf6a96e72e8dfd12518a699c0ec36bd4caba6` (PR #14 gzip data-branch publish merged) |
| Data branch tip | `752ffee7e54de9f6207b11dfa7d8efa91743ac41` |
| Data snapshot | `2026-09-25T15:45:39Z` → **11:45:39 EDT** (`data: snapshot …`) |
| DB | `data/cfb.sqlite3` restored via `git show origin/data:cfb.sqlite3.gz \| gunzip` (61M) |
| Command | `PYTHONPATH=. CFB_DATA_DIR=/workspace/cfb-analytics/data python3 -m cfb_analytics.cli backtest --start-year 2014 --end-year 2025` |
| Raw transcript | local only (gitignored `artifacts/`); regenerate via Command above |
| Promotion | `config/promotion.json` **status=`shadow`** (unchanged) |

---

## Market / SP+ baselines

| Baseline (§8) | Status |
|---|---|
| Vig-free market | **N/A** — `odds_snapshots` / `market_consensus` are **2026-only** (79,432 / 2,116 rows). No leakage-safe historical closes in store. See `docs/plans/2026-09-25-market-baseline-acquisition.md`. |
| SP+-only | **N/A** — `team_ratings` SP+ is `season_final` only; CFBD `/ratings/sp` ignores `week` historically (`backtest/moneyline.py` docstring). |
| Elo-only (CFBD weekly) | **Available** — scored below. |

---

## Fit metadata

| Metric | Value |
|---|---|
| `sigma_0` (fitted point-spread stdev on non-2020) | 19.95 |
| Games used to fit `sigma_0` | 7853 |
| Skipped insufficient season history | 1042 |
| Skipped unrated team (ridge) | 555 |
| Skipped no Elo baseline | 524 |
| Skipped no internal Elo | 311 |
| Skipped no P_logit | 0 |
| Ensemble weights (fit on non-stress) | ridge=0.25, internal_elo=0.65, logit=0.10 |

Stress season **2020** excluded from `sigma_0` / weight fit; scored separately (`STRESS_SEASONS` in `backtest/moneyline.py`).

---

## Main slice — 2014–2025 excl. 2020

| Model | n | Brier | Log loss | vs ridge (logloss) |
|---|---:|---:|---:|---|
| Internal ridge | 7853 | 0.1869 | 0.5627 | — |
| Elo-only (CFBD weekly) | 7348 | 0.1839 | 0.5458 | ridge does **not** beat Elo |
| Internal Elo | 7555 | 0.1851 | 0.5460 | ridge does **not** beat internal Elo |
| P_logit | 7853 | 0.2036 | 0.5938 | ridge **beats** logit |
| **Three-model ensemble** | **7853** | **0.1803** | **0.5332** | ensemble **beats** ridge |
| Market (vig-free close) | — | — | — | **N/A** |

Matches design-of-record §8a numbers in `outlier/docs/plans/2026-08-31-ncaaf-analytics-pipeline.md` (run 2026-09-06).

---

## Stress slice — 2020

| Model | n | Brier | Log loss |
|---|---:|---:|---:|
| Internal ridge | 441 | 0.1909 | 0.5792 |
| Elo-only | 422 | 0.1828 | 0.5443 |
| Internal Elo | 428 | 0.1874 | 0.5550 |
| P_logit | 441 | 0.1898 | 0.5677 |
| **Ensemble** | **441** | **0.1816** | **0.5414** |
| Market | — | — | **N/A** |

---

## Calibration gap (informal)

Approx. max \|bucket midpoint − empirical win rate\| on the **reliability curve**
(home-win probability, 10% bins), buckets with **n ≥ 100**
(`promotion.json`: `max_abs_calibration_gap=0.03`, `min_bucket_n_for_gap_test=100`).

| Model (main slice) | Max \|gap\| | Notes |
|---|---:|---|
| Ensemble | ~0.044 | At ~25% bin (n=530, wr=0.206) — **above** 0.03 gate |
| Ridge | ~0.078 | At ~65% bin |
| Elo-only | ~0.090 | At ~35% bin |

This is a **diagnostic** from published reliability tables, not the production
promotion calibrator. Gate evaluation still awaits market evidence + formal
gap computation in `backtest --promote`.

---

## Interpretation (not a promotion decision)

1. Ensemble is best among computable models on both main and 2020 stress slices.
2. Ridge alone does not beat Elo variants on log loss — consistent with §8a.
3. Two of three required §8 baselines remain uncomputable → **shadow stays**.
4. Next evidence unlock: execute market-baseline acquisition spike (season **2024**).

---

## Blockers for market column

None for this scorecard's Elo/ensemble numbers (run succeeded).  
Market column blocked solely by missing historical closes in DB (not by CLI failure).
