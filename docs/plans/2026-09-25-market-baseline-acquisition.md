# Market-baseline acquisition plan

**Date:** 2026-09-25 (ET)  
**Repo:** `DaSilvaDub/cfb-analytics` @ `master` (`3ccbf6a`)  
**Design of record:** `outlier` `docs/plans/2026-08-31-ncaaf-analytics-pipeline.md` §§8–9  
**Status:** draft plan only — no promotion flip, no live betting, no PR required

---

## 1. Goal

Enable an **honest walk-forward** comparison of model moneyline probabilities against a
**leakage-safe vig-free market baseline**, plus **median CLV vs close**, so
`config/promotion.json`'s two-key gate can pass or fail on evidence:

| Gate key (from `config/promotion.json`) | Needs |
|---|---|
| `require_oos_logloss_beat_market: true` | Historical, pre-kickoff market P(home) joined to the same games the walk-forward scores |
| `require_median_clv_non_negative: true` | Opening vs closing prices (or model price vs close) with settled outcomes |
| Sample-size / calibration floors | Already measurable today (see companion scorecard) |

Until market closes exist for ≥1 full historical season (and ideally ≥
`min_seasons_backtested: 3`), **status stays `shadow`** regardless of Elo/ensemble skill.

---

## 2. Current gap (grounded in code + store)

### 2.1 Documented in `cfb_analytics/backtest/moneyline.py`

The module docstring (lines 19–37) states the two missing §8 baselines explicitly:

1. **Market:** `odds_snapshots` only holds the **current season's live capture**
   (daily ingest started 2026); there is no historical market to compare against
   2014–2025 outcomes.
2. **SP+-only:** `team_ratings` only has **`season_final` SP+** snapshots
   (`ingest/cfbd_fundamentals`). CFBD `/ratings/sp` silently ignores `week` for
   historical seasons — not closable by backfilling harder.

Verified on `origin/data` tip `752ffee` (snapshot 2026-09-25T15:45:39Z):

| Table | Reality |
|---|---|
| `odds_snapshots` | **79,432** rows, **season 2026 only**, `source=cfbd`, books DraftKings/Bovada (plus `"DRAFT KINGS"` spelling variant) |
| `market_consensus` | **2,116** rows / 236 games, season **2026 only** |
| `line_movement` | 28,956 rows (open→current from CFBD `spreadOpen` / `overUnderOpen`) for 2026 |
| `team_ratings` SP+ | `snapshot_scope=season_final` only (Elo is weekly — baseline #3 works) |

### 2.2 What already works (do not reinvent)

- **As-of guard:** `cfb_analytics/features/asof.py` — `AsOfReader.admissible` /
  `check` raise or drop post-kickoff rows; `LeakageError` is fatal by design.
- **Market build path:** `features/build_market.py` reads `odds_snapshots` through
  `AsOfReader` (`as_of_field="captured_utc"`), writes `market_consensus` +
  `line_movement`.
- **Devig / consensus:** `features/market.py` + `models/devig.py` (multiplicative /
  shin / power per `config/settings.json` → `market.devig_*`).
- **CFBD lines ingest (live):** `ingest/cfbd_lines.py` + `sources/cfbd.py::fetch_lines`
  (`GET /lines?year=&week=&seasonType=`). Stores open+current for spread/total;
  ML prices when present; null juice on spread/total.
- **Eastern slate:** `utils.football_date` / `games.football_date` (US Eastern
  calendar date — not UTC date).
- **Walk-forward harness:** `backtest/harness.py` + `cli backtest` already scores
  ridge / CFBD Elo / internal Elo / logit / ensemble. Market column is the missing join.

### 2.3 Promotion gate remains blocked

`config/promotion.json` is still `"status": "shadow"`. Clearing Elo alone does
**not** clear the gate (`require_oos_logloss_beat_market` +
`require_median_clv_non_negative`).

---

## 3. Sources considered

| Source | Historical? | Open + close? | ML juice? | Auth / ops | Verdict |
|---|---|---|---|---|---|
| **CFBD `/lines`** | Yes (year required; cfbfastR docs: from ~2013) | Spread/total: `spreadOpen`/`overUnderOpen` + close fields. ML: typically **close-only** (`homeMoneyline`/`awayMoneyline`) — no separate open ML in schema | Yes on ML | Static `CFBD_API_KEY` (same as daily ingest) | **PRIMARY** |
| Outlier odds archive / live API | Live rich (12 books in-window); historical archive **not** in this store; session token ~24h + OTP | Capture-time series if we had stored it | Yes | Interactive Playwright session — unsuitable for unattended historical backfill | **FALLBACK / live-only** for CLV enrichment on 2026+ |
| Other public closes (e.g. sportsbooks scrapes, third-party dumps) | Uneven; ToS / provenance risk | Varies | Varies | High ops cost | **Do not use** for v1 |

**Primary:** CFBD `/lines` historical backfill into existing `odds_snapshots` /
`line_movement` shapes.  
**Fallback:** If a season's CFBD ML coverage is thin (< acceptance join rate),
supplement that season's **closes only** from any already-licensed Outlier
historical export Daniel already holds — never a new scrape in this spike.
Document coverage gaps; do not invent prices.

---

## 4. Exact as-of rule

### 4.1 Slate identity

- Game belongs to slate `games.football_date` = US Eastern calendar date of
  `kickoff_utc` (`utils.football_date`).
- Historical ingest keys joins via `(football_date, home, away)` exactly as
  `ingest/cfbd_lines.py::build_game_index` already does.

### 4.2 What is "open" vs "close"

| Role | CFBD fields | Synthetic `captured_utc` (because CFBD gives no capture clock) | Allowed uses |
|---|---|---|---|
| **Open** | `spreadOpen`, `overUnderOpen`; ML open **N/A** in CFBD schema | `kickoff_utc - 7 days` (week-ahead proxy), clamped so stamp &lt; kickoff | Movement, open→close CLV on spread/total; **not** a feature for moneyline logloss baseline |
| **Close** | `spread`, `overUnder`, `homeMoneyline`, `awayMoneyline` | `kickoff_utc - 60 seconds` | **Market baseline P** after multiplicative/shin/power devig; **CLV reference** |

Hard rules (must be tested):

1. Every stored historical row has `captured_utc < kickoff_utc` (strict).
2. Feature / consensus builders still go through `AsOfReader` — closing rows
   stamped at `kickoff-60s` are admissible for **scoring baselines**, but any
   code path that builds **model features** from market for the same game must
   use an earlier decision time (e.g. open, or T−24h) if/when market enters S/M
   blend — **out of scope for this spike** (baseline compare only).
3. Unparseable timestamps → `LeakageError` (existing `asof.py` behavior).
4. Never stamp close at or after kickoff; never use season-final aggregates as
   "pregame".

### 4.3 Consensus for baseline #1

For each settled game with ≥ `min_books_for_consensus` (default 3 in
`settings.json`; CFBD historically often has fewer providers — see acceptance):

- Prefer shin (`settings.market.devig_method`), fall back multiplicative.
- Side = HOME vig-free prob from close ML.
- If book count &lt; floor: still store raw closes, flag `thin_market`, and
  **exclude from promotion logloss compare** (same honesty as live path) unless
  we lower the floor for CFBD-only historical with an explicit
  `historical_cfbd_min_books` override documented in evidence.

---

## 5. Storage schema (compatible with existing SQLite)

**No new tables required for the spike.** Reuse:

### 5.1 `odds_snapshots` (already defined in `db.py` MIGRATION_001)

| Column | Historical backfill convention |
|---|---|
| `snapshot_id` | Existing deterministic hash (`ingest/store.insert_odds`) |
| `game_id` | Local id via `build_game_index` |
| `book` | Uppercased CFBD `provider` |
| `captured_utc` | Synthetic open/close stamps per §4 |
| `market` | `ML` / `SPREAD` / `TOTAL` |
| `side` / `line` / `price_*` | Same as `parse_line_rows` |
| `source` | **`cfbd_historical`** (distinct from live `cfbd`) |

Optional lightweight migration (if needed for queries): add generated column or
side table `odds_line_role(snapshot_id, role CHECK IN ('open','close'))` —
prefer encoding role in a **deterministic suffix on `market_id`** or documenting
that close rows are those with `captured_utc` within 2h of kickoff. Spike can
ship with convention-only + tests before a migration.

### 5.2 `line_movement` (MIGRATION_003)

Reuse `parse_movement_rows` / `_write_movement`: one row per
`(game_id, market, side, as_of_utc)` with `open_*` and `current_*` (= close).

### 5.3 `market_consensus`

Rebuild via `build_market_for_slate` over each Eastern `football_date` in the
spike season after odds land — same path as daily ingest.

### 5.4 Out of scope tables

Do not invent `settlements` / `clv_bps` tables in the spike; compute CLV in the
backtest report from close consensus + model prices, then add settlement
persistence in Phase 9 (design §9 / roadmap).

---

## 6. Spike scope

**ONE season of leakage-safe closes: 2024.**

| Why 2024 | Why not others |
|---|---|
| Full season already in `games` (n=920 in current DB) | **2026** is live-capture only — mixing semantics |
| Settled outcomes available for walk-forward scoring | **2020** is stress slice / irregular — bad first market year |
| CFBD `/lines` historically available; recent market structure | **2025** also fine as expansion #2 after 2024 acceptance |
| Large enough for meaningful logloss/CLV n, small enough for one PR | Multi-season backfill waits on spike green |

Stretch (post-spike, same code): 2019, 2021–2023, 2025 until
`min_seasons_backtested ≥ 3` with market coverage.

---

## 7. Acceptance tests

| ID | Test | Pass criterion |
|---|---|---|
| A1 | `AsOfReader` / `LeakageError` | Fixture game: close stamp ≥ kickoff → dropped or `LeakageError` on `check`; open/close &lt; kickoff admitted |
| A2 | Synthetic stamp invariant | Property: for every `cfbd_historical` row, `captured_utc < games.kickoff_utc` |
| A3 | Season isolation | After spike ingest, `odds_snapshots` for season 2024 `source='cfbd_historical'` &gt; 0; no mutation of 2026 live rows' semantics |
| A4 | Join rate to schedule | `COUNT(DISTINCT game_id with ML close) / COUNT(games WHERE season=2024 AND both teams FBS)` ≥ **0.85** (document actual; investigate below 0.85) |
| A5 | Consensus rebuild | `build_market_for_slate` on a 2024 Saturday produces `market_consensus` rows with `as_of_utc < kickoff` |
| A6 | Backtest wiring (thin) | Moneyline report gains a `market` slice **or** explicit `market: N/A (coverage)` — never silent skip |
| A7 | No promotion side effect | `config/promotion.json` status remains `shadow` after spike |

---

## 8. What does NOT change yet

- No flip of `promotion.json` `status` to `promoted`.
- No CORE-tier emission, no live betting, no bankroll advice.
- No SP+ weekly baseline (API limitation remains).
- No Outlier unattended historical scrape.
- No change to ridge / Elo / ensemble fit constants solely because market arrives.
- No force-push / data-branch rewrite beyond normal daily ingest gzip publish.

---

## 9. Effort estimate and ordered steps

**Estimate:** ~2–4 engineer-days for spike (ingest + stamps + tests + thin
backtest hook); +1–2 days to extend to 3 seasons once 2024 is green.

### Ordered implementation

1. **Probe CFBD 2024 coverage** (read-only): `fetch_lines(2024, week=w)` for
   weeks 1–15; tally games with ML both sides, provider set, open fields present.
2. **Extend `ingest/cfbd_lines.py`** (or sibling `cfbd_lines_historical.py`):
   dual-stamp open/close rows; `source='cfbd_historical'`; keep live path unchanged.
3. **CLI:** `backfill-lines --year 2024` (week loop), idempotent via existing
   snapshot hashes.
4. **Rebuild market** for all 2024 `football_date` values present in `games`.
5. **Tests:** A1–A5 in `tests/test_asof.py` / new `tests/test_cfbd_lines_historical.py`.
6. **Wire baseline into `backtest/moneyline.py`:** for games with close
   consensus, score market logloss/Brier on the **same game set** as ridge
   (harness apples-to-apples rule); report CLV median in bps when open+close ML
   exist (spread/total CLV first if ML open missing).
7. **Re-run scorecard** artifact; attach evidence JSON shape for a future
   `backtest --promote` (do not promote).
8. **Expand seasons** only after A4–A6 green on 2024.

### Dependencies / risks

- CFBD ML may lack open prices → median CLV may be spread/total-first or
  model-vs-close-only until Outlier fallback exists.
- Provider count often &lt; 3 → may need documented
  `historical_cfbd_min_books=1` for baseline-only scoring (never for live CORE).
- Name-matching unmatched games already handled (count + skip) in
  `ingest_lines` — spike must report unmatched rate.

---

## 10. Success for this plan doc

This file is the design of record for market acquisition. Companion artifact
`artifacts/moneyline_scorecard_2026-09-25.md` reports current Elo-comparable
skill with **market = N/A** until steps 1–6 land.
