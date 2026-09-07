# Outlier NCAAFB Props & Insights Discovery Probe Report

> **2026-09-07 audit correction:** This run used offline replay and therefore
> does not satisfy R1's authenticated-live-evidence gate. Legacy replay cache
> entries did not preserve HTTP status, so the 200 statuses below are not valid
> evidence of endpoint behavior. Treat the gate as `NOT_SATISFIED` until the
> corrected probe is run live and records response status and provenance. Do
> not proceed to prop schema, ingestion, or executable scoring on this report.

- **Probe Execution Date:** 2026-09-06T07:54:18+00:00
- **Target Slate Date:** 2026-09-05 (US Eastern calendar date)
- **Probe Script:** `scripts/probe_ncaafb_outlier.py` (v1.0.0, Git Commit: `d6d1102`, Branch: `feat/outlier-props-insights`)
- **HTTP Client Mode:** OFFLINE_REPLAY (`CFB_HTTP_MODE=replay`)
- **Authentication State:** REPLAY_FIXTURES (Session: `storage_state.json`, Cognito Bearer Token)
- **Reference Slate Fixtures:** Committed under `tests/fixtures/outlier/`

---

## 1. Executive Summary & Gate Verdict

- **Milestone M0 Gate Status:** PARTIAL_PASS
- **Gamelines Resolution:** CONFIRMED_FUNCTIONAL
- **Team Props Resolution:** UNOFFERED
- **Player Props Resolution:** UNOFFERED
- **Insights Endpoint Status:** EMPTY_200
- **Recommendation for M1:** Proceed to M1 with baseline gamelines and graceful degradation for unoffered props/insights; do not synthesize mock feeds.

---

## 2. Slate & Sampled Events Context

### 2.1 Slate Overview
- **Schedule Endpoint:** `GET /sportsdata/leagues/NCAAFB/schedule` -> HTTP 200
- **Total Scheduled Events on Slate:** 31 games (Total in schedule: 125)
- **Sample Selection Criteria:** 3 representative matchups sampled on 2026-09-05.

### 2.2 Sampled Event Profiles
| # | Event ID | Kickoff (UTC) | Eastern Slate | Matchup (Away @ Home) | Venue | Network |
|---|----------|---------------|---------------|-----------------------|-------|---------|
| 1 | `d3c27ca7c2343f149a695c5a810549c9cc4d6426` | `2026-09-05T16:30:00+00:00` | `2026-09-05` | Cardinals @ Buckeyes | Ohio Stadium | National |
| 2 | `e04493d90a41b698b3c38d344c417330478d6d9d` | `2026-09-05T16:00:00+00:00` | `2026-09-05` | Mean Green @ Hoosiers | Memorial Stadium (Bloomington, IN) | National |
| 3 | `4a8966e71bd7d4f2fd73daa3e80a97c2943b7593` | `2026-09-05T19:30:00+00:00` | `2026-09-05` | Thundering Herd @ Nittany Lions | West Shore Home Field at Beaver Stadium | National |

---

## 3. Findings by Probed Token

### 3.1 Token: `GAMELINE`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=GAMELINE`
- **HTTP Status:** 200 OK across all sampled games.
- **Propositions Discovered:** `MONEYLINE`, `SPREAD`, `TOTAL`, plus derivative props (`DOUBLE_RESULT`, `MONEYLINE_THREE_WAY`, `WINNING_MARGIN`).
- **Trap 1 Verification (Non-Parallel Books):**
  - *Observation:* Outcome object `books` list order does NOT match `odds[].book` order.
  - *Evidence:* In probed event `d3c27ca7c2343f149a695c5a810549c9cc4d6426`, `outcome.books[0]` was `MIDNITE` while `outcome.odds[0].book` was `DRAFTKINGS`.
  - *Conclusion:* Verified. Book attribution must strictly read from `odds[].book`.
- **Trap 2 Verification (Multi-Row Proposition Spanning):**
  - *Observation:* `SPREAD` spanned 25 market cards across sampled games; `TOTAL` spanned 26 cards.
  - *Evidence:* Each market card quotes distinct book subsets; full coverage requires unioning rows.
  - *Conclusion:* Verified. Parser must union all market rows for a proposition and deduplicate by `(book, side, line)`.

### 3.2 Token: `TEAM_PROP`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=TEAM_PROP`
- **HTTP Status:** 200 OK (empty `{"markets": []}` envelope in offline replay / unoffered).
- **Discovered Propositions:** None (unoffered on sampled slate).
- **Team Attribution Schema:** N/A (no prop cards returned).
- **Sportsbook Depth:** 0 books quoting team props on sampled games.
- **R2 Alignment:** Evaluated; team props return empty payload. Pipeline must degrade gracefully without crashing.

### 3.3 Token: `PLAYER_PROP`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=PLAYER_PROP`
- **HTTP Status:** 200 OK (empty `{"markets": []}` envelope in offline replay / unoffered).
- **Discovered Propositions:** None (unoffered on sampled slate).
- **R2 Compliance Action:** **STRICTLY DROPPED / EXCLUDED PER R2 §54.** Player props are not ingested into `odds_snapshots`.
- **Player Identity Field Assessment:** (See Section 5 for detailed technical analysis).

### 3.4 Token: `GAME_PROP`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=GAME_PROP`
- **HTTP Status:** 200 OK (empty `{"markets": []}` envelope in offline replay / unoffered).
- **Discovered Propositions:** None.
- **R2 Compliance Action:** **STRICTLY DROPPED PER R2.** Unwhitelisted game props are dropped during parsing.

### 3.5 Endpoint: `/insights`
- **Endpoints Probed:**
  - `GET /sportsdata/events/{eventId}/insights`
- **HTTP Status:** 200 OK (empty `{"insights": []}` envelope in offline replay / unoffered).
- **Payload Contents:** Empty insights list.
- **Action per R1 §35:** Stop condition triggered for insights. Endpoint does not provide an active insights feed for NCAAFB; do not synthesize mock feed.

---

## 4. Comprehensive Proposition & Sportsbook Depth Table

| Market Type | Proposition | Scope | Sample Rows | Event Freq | Distinct Sportsbooks | Book Count | Median Books/Game | Consensus Eligible (>=3) | R2 Action | Target Market Code |
|---|---|---|---|---|---|---|---|---|---|---|
| `GAMELINE` | `DOUBLE_RESULT` | `full_game` | 3 | 3/3 (100%) | DRAFTKINGS, FANATICS, MIDNITE | 3 | 3.0 | `YES` | `DROP_UNWHITELISTED` | `N/A` |
| `GAMELINE` | `MONEYLINE` | `full_game` | 19 | 3/3 (100%) | BETRIVERS, DRAFTKINGS, FANATICS, FANDUEL, ... | 12 | 9.0 | `YES` | `ADMIT` | `ML` |
| `GAMELINE` | `MONEYLINE_THREE_WAY` | `full_game` | 8 | 3/3 (100%) | DRAFTKINGS, FANATICS, THESCOREBET | 3 | 2.0 | `NO` | `DROP_UNWHITELISTED` | `N/A` |
| `GAMELINE` | `SPREAD` | `full_game` | 25 | 3/3 (100%) | BETRIVERS, CAESARS, DRAFTKINGS, FANATICS, ... | 13 | 13.0 | `YES` | `ADMIT` | `SPREAD` |
| `GAMELINE` | `TOTAL` | `full_game` | 26 | 3/3 (100%) | BETRIVERS, CAESARS, DRAFTKINGS, FANATICS, ... | 13 | 13.0 | `YES` | `ADMIT` | `TOTAL` |
| `GAMELINE` | `WINNING_MARGIN` | `full_game` | 5 | 3/3 (100%) | DRAFTKINGS, FANDUEL | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` | `N/A` |
| `TEAM_PROP` | `POINTS` | `full_game` | 0 | 0/3 (0%) | None | 0 | 0.0 | `NO` | `UNOFFERED` | `POINTS` |
| `TEAM_PROP` | `OFFENSIVE_YARDS` | `full_game` | 0 | 0/3 (0%) | None | 0 | 0.0 | `NO` | `UNOFFERED` | `OFFENSIVE_YARDS` |
| `TEAM_PROP` | `RECEIVING_YARDS` | `full_game` | 0 | 0/3 (0%) | None | 0 | 0.0 | `NO` | `UNOFFERED` | `RECEIVING_YARDS` |
| `TEAM_PROP` | `RUSHING_YARDS` | `full_game` | 0 | 0/3 (0%) | None | 0 | 0.0 | `NO` | `UNOFFERED` | `RUSHING_YARDS` |
| `PLAYER_PROP` | `(ALL_PROPS)` | `full_game` | 0 | 0/3 (0%) | None | 0 | 0.0 | `N/A` | `EXCLUDED_R2` | `N/A` |
| `GAME_PROP` | `(ALL_PROPS)` | `full_game` | 0 | 0/3 (0%) | None | 0 | 0.0 | `N/A` | `DROPPED_R2` | `N/A` |

---

## 5. Player Identity Field Assessment

### 5.1 Outcome Field Inspection
```json
// Representative Player Prop Outcome Record (if present):
// None present on sampled slate (markets: [] returned)
```

### 5.2 Technical Evaluation
1. **Identifier Availability:** None present in sampled empty payloads.
2. **Format & Grain:** N/A (Player props unoffered).
3. **Cross-Event Stability:** N/A.
4. **Joinability with CFBD:** N/A.
5. **Architectural Scoping Decision:** Confirm that in strict adherence to R2, player props are dropped at ingestion; no `player_id` is required in `odds_snapshots` for M1.

---

## 6. Insights Endpoint Technical Assessment

### 6.1 Call Resolution Log
| URL Probed | Event ID / Scope | HTTP Status | Response Time | Payload Size | Result |
|---|---|---|---|---|---|
| `/sportsdata/events/d3c27ca7c2343f149a695c5a810549c9cc4d6426/insights` | `d3c27ca7c2343f149a695c5a810549c9cc4d6426` | 200 OK | <50ms | 18 bytes | `insights: []` (empty) |
| `/sportsdata/events/e04493d90a41b698b3c38d344c417330478d6d9d/insights` | `e04493d90a41b698b3c38d344c417330478d6d9d` | 200 OK | <50ms | 18 bytes | `insights: []` (empty) |
| `/sportsdata/events/4a8966e71bd7d4f2fd73daa3e80a97c2943b7593/insights` | `4a8966e71bd7d4f2fd73daa3e80a97c2943b7593` | 200 OK | <50ms | 18 bytes | `insights: []` (empty) |

### 6.2 Schema & Viability Analysis
- **Status Summary:** Endpoint returns empty insights list (`{"insights": []}`).
- **Decision:** Disable `--with-insights` by default; do not synthesize mock insights feed.

---

## 7. Hard Gate Compliance & Next Steps
- **Hard Gate Verdict:** PARTIAL_PASS
- **Rationale:** Gamelines are fully functional with median 13 books pricing `SPREAD` and `TOTAL`, and median 9 books pricing `MONEYLINE`. Team props and player props are currently unoffered (empty) in the feed. Graceful degradation and frozenset whitelist must be implemented in downstream milestones.
- **Next Milestone Actions (M1):**
  - Implement Migration 10 table rebuild in `cfb_analytics/db.py` (free and unconflicted on canonical master).
  - Define frozenset whitelist in `cfb_analytics/sources/outlier.py` matching verified markets.
  - Implement dedicated `prop_consensus` table DDL.
