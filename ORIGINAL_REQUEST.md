# Original User Request

## 2026-09-06T07:19:33Z

# Teamwork Project Prompt — NCAAF Props & Insights (v3)

> Status: Launched.
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full Team

Extend the **existing** `cfb-analytics` NCAA football pipeline to ingest Outlier **player props, team props, and insights**, alongside the gameline odds it already collects.

**Working directory:** `C:\Users\dasil\Dev\GitHub\cfb-analytics`
**Integrity mode:** benchmark (strict adherence to existing repo architecture, no external AI/paid paths)
**Branch:** create `feat/outlier-props-insights` off `master`. Never commit to `master` directly.
**Deliverable:** one PR against `cfb-analytics` `master`.

---

## Why this changed from v1 (read before starting)

The NCAAF integration ships as a separate repo, `cfb-analytics`, importing nothing from `outlier`. The Outlier league token is `NCAAFB`, not `NCAAF` (which returns HTTP 502). The `cfb-analytics` repo already exists with SQLite ledgers, migrations, and gamelines. What is genuinely unbuilt is props and insights.

---

## Requirements

### R1. Discovery spike — a hard gate, before any schema or parsing work

Nothing below R1 may be designed until R1 is answered with recorded evidence. Using the existing authenticated session, read-only, against a slate inside the operating window (within ~5 days of kickoff):

- Which `marketType` tokens does `/sportsdata/events/{eventId}/markets?marketType=…` accept for `NCAAFB`? Probe at minimum `PLAYER_PROP`, `TEAM_PROP`, `GAME_PROP`, and confirm `GAMELINE` still behaves as documented.
- For every token that resolves: what `proposition` values appear, at what frequency, and how many distinct books price each? Report median books per game per proposition.
- Is there a player identity field on prop outcomes, and is it a stable `playerId` or a display name?
- Does an insights endpoint exist for `NCAAFB`? If it 404s or 403s, say so and stop — do not synthesise an insights feed.

Commit the probe script and payloads under `tests/fixtures/`, and write findings into `docs/probes/` as a dated markdown file. If a prop family is not actually offered, report that rather than building a mapping.

### R2. Target Markets, Whitelist, and Side Restrictions

Use the repository's existing `market_type` and `market` fields. Define this explicit whitelist as module-level frozensets before ingestion is wired up:

| Requested market | `market_type` | `market` | `scope` | Allowed sides |
|---|---|---|---|---|
| Full-game spread | `GAMELINE` | `SPREAD` | `full_game` | `HOME`, `AWAY` |
| First-half spread | `GAMELINE` | `SPREAD` | `first_half` | `HOME`, `AWAY` |
| Full-game total points | `GAMELINE` | `TOTAL` | `full_game` | `OVER`, `UNDER` |
| Moneyline | `GAMELINE` | `ML` | `full_game` | `HOME`, `AWAY` |
| Team total points | `TEAM_PROP` | `POINTS` | `full_game` | `OVER`, `UNDER` |
| Team offensive yards | `TEAM_PROP` | `OFFENSIVE_YARDS` | `full_game` | `OVER`, `UNDER` |
| Team receiving yards | `TEAM_PROP` | `RECEIVING_YARDS` | `full_game` | `OVER`, `UNDER` |
| Team rushing yards | `TEAM_PROP` | `RUSHING_YARDS` | `full_game` | `OVER`, `UNDER` |

*Note: These aliases become authoritative only after confirmation in R1. Explicitly out of scope: NCAAF player props, quarters, derivative spreads, executable plays, and bet sizing.*

### R3. Schema & Integrity

- `odds_snapshots` admits props via a **table-rebuild migration** (version `10` in `cfb_analytics/db.py`). Existing rows must survive with identical `snapshot_id`s.
- Decide explicitly whether to widen `odds_snapshots`/`market_consensus` with a nullable `player_id`, or use separate `prop_odds_snapshots`/`prop_consensus` tables. Do not overload existing tables by accident.
- `snapshot_id` must remain deterministic for idempotency.

### R4. Parsing

Extend `cfb_analytics/sources/outlier.py`. Respect two documented traps (and prove with fixtures):
1. `outcomes[].books` is **not** parallel to `outcomes[].odds`. Book is read from inside each odds entry.
2. One proposition spans several market rows per event. Consensus must union all rows and de-duplicate.

### R5. Ingestion & Reporting

- Thread props and insights through `ingest_slate()` behind explicit flags (`with_props`, `with_insights`), defaulting **off**.
- Degrade gracefully: one event's prop fetch failing must not lose the rest. Add failures to `source_health` and `IngestSummary`.
- Apply a `min_books_for_consensus = 3` floor to any prop consensus.

### R6. Constraints & Methodology

- **No external AI/paid reasoning:** No `run_desk`, no OpenAI/Gemini/Claude calls. 
- **Code constraints:** Nothing imported from `outlier` or `nba-props-pipeline`. Stdlib-only for all modelling math (no pandas/numpy).
- **TDD:** Write failing tests first against committed fixtures (`CFB_HTTP_MODE=replay`), then implement. Tests never reach the network.
- **Date logic:** Prefer earliest date from today through 14 days ahead with non-final NCAAF events. Fallback to past 14 days if needed.

## Acceptance Criteria

### Discovery
- [ ] R1 findings committed (`docs/probes/` and `tests/fixtures/`) stating HTTP status, propositions, and median books for each probed token.

### Correctness & Logic
- [ ] Fixture proves parser attributes prices to the correct book despite `outcomes[].books` ordering.
- [ ] Fixture proves proposition prices are unioned across multiple market rows and de-duplicate.
- [ ] Markets outside the explicit whitelist (R2) are dropped at ingestion.
- [ ] Home and away team props map correctly.
- [ ] Missing/empty prop endpoints degrade cleanly without crashing gameline ingestion.

### Migration & Integrity
- [ ] Migration 10 applies cleanly; pre-existing rows keep original `snapshot_id`s.
- [ ] Re-running ingestion is idempotent (no duplicate rows).

### Regression & Quality
- [ ] Existing test suite passes unchanged. No test edited to accommodate props without prose explanation.
- [ ] `ruff check .`, `mypy cfb_analytics`, and `pyright` run cleanly.
- [ ] `pytest --cov=cfb_analytics` achieves **>=80% coverage on every module touched**.
- [ ] `ingest --date <slate> --with-props` produces a summary giving events, games written, prop rows by family, distinct books, and failures.

---
## Notes for the team
- Use the reference slate `2026-09-05` for realistic fixtures (30 games, 12 books, ~630 prices).
- Read conventions in `README.md` and module docstrings before assuming a house style.
- If R1 shows Outlier offers no usable NCAAFB props, **say so and stop.** Truthful negative findings close this project successfully.

## 2026-09-23T23:49:25Z

Build an end-to-end, multi-factor reasoning and market-mispricing pipeline for NCAA College Football that evaluates Moneylines, Game Spreads, Game Totals, and Team Props using deep situational data (recent game tape/form, live/upcoming weather, opponent injuries, roster/talent composite grades, recruiting classes, and program structural stability) rather than defaulting to favorites.

Working directory: c:/Users/dasil/Dev/GitHub/cfb-analytics
Integrity mode: development

Reference material:
- Foundational architectural spec: docs/plans/NCAA College Football Outlier Pipeline — Three-Model Betting System.md
- Decision rules & governance: docs/grok_rules.md (compiled directly from the user's Grok conversation)

## Requirements

### R1. Multi-Factor Contextual Game Reasoning Engine
For any candidate pick across supported markets (Moneyline, Spread, Total, Team Props), generate a complete data-backed reasoning card before any pick is recommended. Every card must evaluate and synthesize:
- Recent Games & Form: Margin of victory, offensive/defensive EPA/success trends, game script tendencies, and quality of recent opponents (filtering out cupcake tape).
- Venue & Weather: Forecast conditions at kickoff and throughout the game (temperature, sustained winds, wind gusts, precipitation probability) and their historical impact on run/pass ratios and scoring.
- Injuries & Availability: Opponent and team injury reports, depth chart scratches, QB status/continuity, and trench (OL/DL) attrition.
- Roster Talent & Recruiting: 247/On3 roster talent composites, recruiting class rankings, blue-chip ratios, and transfer portal composite ratings comparing the two programs.
- Program Structure & Coaching: Coaching continuity, coordinator changes, rest advantages (bye weeks, short turnaround), and road/travel dynamics.
- Negative Gate: Strict refusal to endorse favorites purely based on market price or ranking without affirmative edge verification across these factors.

### R2. Comprehensive Mispriced Line Scanner (Spreads, Totals, Props)
Expand the pipeline's consensus and devig engines beyond moneyline and rush/receiving boards to systematically detect line discrepancies:
- Game Spreads & Alternate Spreads: Model-projected spread margin vs. devigged multi-book consensus spread.
- Game Totals & Alternate Totals: Fundamental offensive/defensive tempo and weather-adjusted scoring projections vs. consensus game totals.
- Team Props (Team Points, Team Yards): Opponent-adjusted team production projections vs. posted book lines.
- Edge Quantification: Report edge percentage, vig-free fair probability (Shin/Multiplicative), method spread, and confidence score for each identified misprice.

### R3. Grok Decision Rule Integration & Governance
Incorporate the specific decision rules and heuristic constraints documented in docs/grok_rules.md as mandatory gating criteria for bet qualification, market grading, and parlay construction.

### R4. CLI Interface & Reporting
Expose the multi-factor reasoning and mispriced board via CLI commands:
- Integration into python -m cfb_analytics.cli (e.g., cfb-analytics board --with-reasoning --date <YYYY-MM-DD> and cfb-analytics mispriced --date <YYYY-MM-DD>).
- Output structured, readable terminal cards as well as exportable structured JSON records.

## Acceptance Criteria

### Automated Reasoning Coverage
- [ ] Every recommended Moneyline, Spread, Game Total, or Team Prop outputs an auditable reasoning card containing all six contextual dimensions (recent form, weather, injuries, talent/recruiting, program structure, model edge).
- [ ] Any favorite where the underlying situational data contradicts the market line is automatically flagged or disqualified with an explicit counter-thesis.

### Market Mispricing Engine
- [ ] Automated scanner identifies and ranks mispriced opportunities across Game Spreads, Game Totals, and Team Props against consensus books.
- [ ] All detected edges compute and report vig-free fair price, market consensus line, edge percentage, and book disagreement spread.

### Rule Compliance & Test Verification
- [ ] Passes all existing test suites without breaking repository invariants (pytest tests/).
- [ ] New unit tests verify reasoning card assembly, edge calculation for spreads/totals, and enforcement of the Grok governance gates.
- [ ] All generated outputs maintain the shadow mode disclaimer: UNPROMOTED - shadow output, not decision-grade.

