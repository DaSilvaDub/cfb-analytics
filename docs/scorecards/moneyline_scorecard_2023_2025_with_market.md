# Moneyline scorecard — 2023–2025 with market close baseline

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~00:05 EDT** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| DB | local `data/cfb.sqlite3` after historical backfill for **2023, 2024, 2025** |
| Command | `cli backtest --start-year 2023 --end-year 2025` |
| Promotion | `config/promotion.json` **status=`shadow`** (unchanged) |

## Historical backfill join rates (A4)

| Season | ML close / FBS | Rate |
|---|---:|---:|
| 2023 | 746 / 796 | **0.937** |
| 2024 | 783 / 803 | **0.975** |
| 2025 | 817 / 809 | **1.010** † |
| A2 leakage | 0 | pass |

† Denominator is the CLI’s FBS filter; a few priced games sit outside that count. Treat as ≥0.85 pass, not literal over-coverage.

Three seasons with leakage-safe closes now exist locally → `min_seasons_backtested: 3` **sample-size** dimension is meetable for market evidence. Gate still not flipped.

## Results (2023–2025 excl. 2020)

| Model | n | Brier | Log loss |
|---|---:|---:|---:|
| Internal ridge | 2204 | 0.1880 | 0.5680 |
| Elo-only | 2125 | 0.1869 | 0.5519 |
| Internal Elo | 2126 | 0.1856 | 0.5469 |
| P_logit | 2204 | 0.2045 | 0.5971 |
| **Ensemble** | **2204** | **0.1801** | **0.5320** |
| **Market close** | **1307** | **0.1846** | **0.5461** |

- Ensemble beats ridge (0.5320 vs 0.5680).
- Report line: ridge does not beat market (0.5680 vs 0.5461) — **caveat:** market n=1307 is the covered subset; 897 ridge games lacked close ML consensus. Honest gate compare must score ensemble/ridge on the **same** 1307 games (follow-up wiring if not already identical in `--promote`).
- Skipped (no close ML consensus): 897.
- SP+-only still N/A.

## Gate implication (still shadow)

| Promotion need | Status after this expand |
|---|---|
| ≥3 seasons backtested with market | **Locally yes** (2023–25) |
| OOS logloss beat market (same game set) | **Not decided** — ensemble headline logloss looks better than market but n differs; confirm apples-to-apples before any claim |
| Median CLV ≥ 0 | **Not computed** in this scorecard |
| Calibration gap ≤ 0.03 | Not re-run formally here |
| `status` flip | **No** — stays `shadow` |

## Next

1. Apples-to-apples market-overlap metrics in `backtest/moneyline.py` / `--promote` evidence JSON.
2. Median CLV (model/open vs close) on the covered set.
3. Ops: publish historical `cfbd_historical` odds onto `data` branch (gzip size) so CI/others share the store — separate from this model PR.
4. Optional: backfill 2021–2022 for robustness (skip 2020 as primary market year).
