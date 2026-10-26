# CFBD 2024 lines probe + backfill notes

**Date:** 2026-09-25 (ET)  
**Branch / PR:** `feat/cfbd-historical-lines-2024` / #17  
**Status:** live probe + backfill **completed** (local `data/cfb.sqlite3`; not published to `data` branch)

## Probe (`backfill-lines --year 2024 --probe-only`)

| Metric | Value |
|---|---:|
| Games in CFBD feed (weeks 1–15 regular) | 1518 |
| Games with ML both sides | 783 |
| Games with spread open | 814 |
| Games with total open | 814 |
| Feed ML coverage rate | 0.516 |
| Providers | BOVADA, DRAFTKINGS, ESPN BET |

## Backfill (`--rebuild-market --historical-cfbd-min-books 1`)

| Metric | Value |
|---|---:|
| Priced locally (matched store games) | 873 |
| Odds rows (`source=cfbd_historical`) | 21642 |
| Movement rows | 7758 |
| Not in store (FCS / unmatched) | 645 |
| Open / close odds rows | 7766 / 13876 |
| Games with ML close | 783 |
| Market rebuild | 81 slates, 920 games, 1544 consensus rows |
| **A4 join rate (ML close / FBS games)** | **783/803 = 0.975** (target ≥ 0.85) |
| A2 leakage (`captured_utc >= kickoff`) | **0** |
| Live `source=cfbd` 2026 rows | unchanged (79432) |
| `promotion.json` | still `shadow` (A7) |

## Interpretation

- Acceptance **A4 passed**. Thin feed-wide ML rate (~52%) is mostly non-store / FCS games; among FBS store games with closable MLs we hit 97.5%.
- `historical_cfbd_min-books=1` used for baseline-only rebuild (documented; not for live CORE).
- Local DB only — publishing historical odds onto the `data` branch is a separate ops decision (gzip size / daily ingest scope).

---

## Expansion — 2023 + 2025 (2026-09-26 ET)

Same CLI: `backfill-lines --year {Y} --rebuild-market --historical-cfbd-min-books 1`

| Season | Join rate (ML close / FBS) | Odds rows (cumulative source) |
|---|---:|---|
| 2023 | 0.937 | +17928 |
| 2024 | 0.975 | (prior) |
| 2025 | ≥0.85 (CLI 1.010†) | +23150 |

A2 still 0. See `docs/scorecards/moneyline_scorecard_2023_2025_with_market.md`.


---

## Same-game-set promote evidence (2026-09-26 ET)

`backtest --start-year 2023 --end-year 2025 --promote` with
`historical_cfbd_min_books=1`:

| Field | Value |
|---|---|
| n_full / n_overlap / skipped | 2204 / 2103 / 101 |
| seasons_with_overlap | 2023–2025 |
| ensemble vs market logloss (same set) | 0.5548 vs 0.5287 → **beat_market=false** |
| median CLV (model-vs-close) | +0.0004 prob-pts (+3.7 bps) → non_negative |
| calibration_gap | 0.0659 (> 0.03) |
| `promotion.json` status | **shadow** (fail-closed) |

See `docs/scorecards/moneyline_scorecard_2023_2025_with_market.md`.
