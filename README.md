# cfb-analytics

NCAA FBS college football analytics pipeline. Three separate models by design —
**safety** (who is least likely to lose), **value** (is the price worth paying),
and **totals** — plus a parlay optimizer that can refuse a leg which buys payout
at too high a cost in win probability.

Design of record: `docs/plans/2026-08-31-ncaaf-analytics-pipeline.md` in the
`outlier` repo.

> **Shadow mode.** Nothing here is decision-grade. No CORE-tier leg or parlay is
> emitted until the Phase 4 backtest clears the two-key promotion gate in
> `config/promotion.json` (a sample-size floor **and** demonstrated out-of-sample
> skill against a vig-free market baseline). A high win probability still carries
> loss risk; `PASS` is a valid and often preferable output.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffold, config, SQLite migrations, CI | **done** |
| 1 | Outlier ingestion (schedule, gameline odds, injuries) | **done** |
| 1 | CFBD teams/venues/games backfill | **implemented**; live 2014–2025 load needs `CFBD_API_KEY` |
| 2a | Leakage guard, devig (3 methods), market consensus, line movement | **done** |
| 2a | Weather (Open-Meteo), venue geolocation | **done** |
| 2b–9 | Fundamentals features, models, backtest, totals, parlay, reporting | in progress |

ESPN was evaluated as a starting-QB source and **dropped**: it publishes no CFB
depth chart (`/teams/{id}/depthchart` returns `{}`; core-API variants 400/404),
and its roster carries no `starter` field. CFBD covers the rest better. No free
source confirms a starting quarterback, so that remains a hard CORE blocker.

### Scheduled capture

`.github/workflows/daily-ingest.yml` runs at 11:00 UTC daily (7am ET, ahead of
the day's moves — a *consistent* capture time matters more than the hour, since
day-over-day movement is only comparable between like snapshots). It restores
the SQLite store from the `data` branch, ingests, rebuilds the market, and
force-pushes a single-commit snapshot back.

Requires one repository secret: **`CFBD_API_KEY`**.

The Outlier leg is opt-in via workflow_dispatch and best-effort: its access
token lives **24 hours** and is refreshed by an interactive Playwright +
email-OTP login, so an unattended runner finds it expired on day two. CFBD is
the source a scheduled job can actually rely on — a static key, no session, and
it returns `spreadOpen`/`overUnderOpen` next to the current numbers, so movement
is measurable on the first run instead of after a week of accumulation.

## Setup

```bash
pip install -e ".[dev]"
python -m cfb_analytics.cli init-db
python -m cfb_analytics.cli doctor
```

`doctor` reports source readiness without printing any credential value.

### Credentials

| Variable | Needed for | How |
|---|---|---|
| `CFBD_API_KEY` | All fundamentals, ratings, and historical backfill | Free at <https://collegefootballdata.com/key> |
| *(none)* | Outlier odds and injuries | Reads the saved session from the `outlier` project; override the location with `OUTLIER_SESSION_DIR` |

Put `CFBD_API_KEY` in the environment or in a local `.env` (gitignored). It is
read via `os.getenv` only — never logged, never written to an artifact, never
included in an exception message.

## Usage

```bash
python -m cfb_analytics.cli schedule                    # available slate dates
python -m cfb_analytics.cli ingest --date 2026-09-05    # one slate
python -m cfb_analytics.cli backfill-cfbd --start-year 2014 --end-year 2025
python -m cfb_analytics.cli status                      # row counts, last run
python -m cfb_analytics.cli coverage                    # books per capture
```

`CFB_HTTP_MODE` controls the HTTP layer: `live` (default), `refresh` (ignore
cache TTL), `replay` (serve only from cache; a miss raises — this is what the
test suite uses, so the whole suite runs offline).

## Things this codebase knows that are easy to get wrong

Each is verified against the live feed and covered by a regression test.

1. **A slate is a US Eastern calendar date, not a UTC date.** For 2026-09-05,
   UTC grouping pulls in four Friday-night games and drops four Saturday-night
   West Coast games — 8 of 34 misassigned, and the count stays at 30 either way,
   so the error is invisible unless you look at the membership.
   (`utils.football_date`, `tests/test_football_date.py`)
2. **`outcomes[].books` is not parallel to `outcomes[].odds`.** Index-zipping
   mis-attributes every price to the wrong book. Read `book` from inside each
   odds entry. (`tests/test_outlier_parsing.py::TestBookAttribution`)
3. **One proposition spans several market rows**, each with a different book
   subset. Union across rows before calling anything "the market".
4. **Book coverage varies over time.** A 2026-08-31 probe saw 20 books including
   Pinnacle and Circa; hours later the same games returned 11 with no sharp
   books. Run `coverage` across ingests before relying on any book being there.
5. **The two sides of a spread carry opposite signs** — HOME −13.5 against
   AWAY +13.5 — so keying a market group on the raw line leaves them unpaired.
   That produced 694 phantom negative-hold rows out of 1594 spreads while ML
   and totals showed zero, because those already share a line value across
   sides. Spreads group on the absolute line.
   (`market.group_key`, `tests/test_market.py::TestSpreadSidesPairUp`)
6. **Devig each book against itself, then aggregate.** Median-each-side-then-
   devig mixes different book sets: on BALL@OSU the home side had 2 books and
   the away side 4, and the mixed devig produced a −18249 "consensus" on a
   market whose best real price was −10000.
7. **`-100000` means "no action", not "99.9%".** Placeholder quotes are dropped
   and flagged rather than averaged into a consensus.
8. **An injury feed is not a depth chart.** Outlier gives real designations
   (Out / Doubtful / Questionable / Probable / Out for Season) and covers QBs,
   but absence from it does not confirm a healthy starter — so unknown QB status
   is a hard CORE blocker, and there is **no OL coverage at all**.

## Development

```bash
python -m pytest          # offline; no network
python -m ruff check .
python -m mypy cfb_analytics
```

Modelling math is deliberately pure-stdlib — no numpy, scipy, scikit-learn, or
pandas. With ~800 FBS games a season, regularised linear models, Elo, and
isotonic calibration beat gradient boosting on variance grounds anyway. The one
runtime dependency is `tzdata`, a pure-data package needed because Windows ships
no IANA time-zone database and slate dates are Eastern.
