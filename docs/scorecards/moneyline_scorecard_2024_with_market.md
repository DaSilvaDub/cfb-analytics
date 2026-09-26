# Moneyline scorecard — 2024 with market close baseline

| Field | Value |
|---|---|
| Generated | **2026-09-25 ~23:57 EDT** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| DB | local `data/cfb.sqlite3` after `backfill-lines --year 2024 --rebuild-market --historical-cfbd-min-books 1` |
| Command | `cli backtest --start-year 2024 --end-year 2024` |
| Promotion | `config/promotion.json` **status=`shadow`** (unchanged) |

## Results (season 2024 only)

| Model | n | Brier | Log loss |
|---|---:|---:|---:|
| Internal ridge | 738 | (see raw) | 0.5993 |
| Elo-only (CFBD) | (subset) | — | 0.5715 |
| Internal Elo | 714 | 0.1952 | 0.5692 |
| P_logit | 738 | 0.2100 | 0.6112 |
| **Ensemble** | **738** | **0.1902** | **0.5560** |
| **Market close** | **670** | **0.1844** | **0.5452** |

- Ridge does **not** beat market (0.5993 vs 0.5452).
- Ensemble does **not** beat market on this slice (0.5560 vs 0.5452).
- Skipped (no close ML consensus): 68 of the ridge game set.
- SP+-only baseline still N/A (CFBD weekly SP+ dead end).

## Gate implication

This is **evidence**, not a promotion decision. Two-key gate still needs multi-season market coverage (`min_seasons_backtested: 3`), formal calibration gap, and median CLV. Next: expand historical backfill to additional seasons with the same CLI.
