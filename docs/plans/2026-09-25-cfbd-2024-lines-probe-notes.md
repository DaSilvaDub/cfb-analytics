# CFBD 2024 `/lines` probe notes

**Date:** 2026-09-25 (ET)  
**Spike:** `feat/cfbd-historical-lines-2024`  
**Status:** live probe deferred — `CFBD_API_KEY` not present in the agent environment

## What was implemented without the key

- Dual-stamp historical ingest (`source=cfbd_historical`) + CLI `backfill-lines`
- Unit/acceptance tests A1–A6 with fixtures (no invented prices)
- Thin moneyline wiring that reports `market: N/A (coverage)` when closes are absent

## What needs the key

```bash
# Read-only coverage tally (weeks 1–15)
cfb-analytics backfill-lines --year 2024 --probe-only

# Idempotent backfill + market rebuild (baseline-only min books = 1)
cfb-analytics backfill-lines --year 2024 --rebuild-market --historical-cfbd-min-books 1
```

Acceptance A4 target: `COUNT(DISTINCT game_id with ML close) / COUNT(FBS 2024 games) >= 0.85`.

## Promotion

`config/promotion.json` status remains **`shadow`**. No CORE / live betting changes.
