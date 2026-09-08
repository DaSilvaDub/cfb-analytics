"""Command-line entry point.

Only the commands backed by working code are registered. Phases 2-9 add
``features``, ``train``, ``slate``, ``parlay`` and ``settle``; they are
deliberately absent rather than present-and-stubbed, so ``--help`` never
advertises something that does not run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from cfb_analytics import config, db, paths
from cfb_analytics.errors import CfbAnalyticsError, SchemaError


def _cmd_init_db(args: argparse.Namespace) -> int:
    paths.ensure_dirs()
    # Open WITHOUT the implicit migrate, so this command can report what it
    # actually applied instead of always finding the schema already current.
    with db.open_db(migrate_on_open=False) as conn:
        applied = db.migrate(conn)
    target = paths.database_path()
    if applied:
        print(f"applied migrations {applied} -> {target}")
    else:
        print(f"schema already current -> {target}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report which sources are usable. Prints no credential values."""
    from cfb_analytics.sources import session

    print("cfb-analytics doctor\n")
    print(f"  data dir      : {paths.data_dir()}")
    print(
        f"  database      : {paths.database_path()} "
        f"({'present' if paths.database_path().exists() else 'not created'})"
    )
    mode = "SHADOW - no CORE tier emitted" if config.is_shadow_mode() else "PROMOTED"
    print(f"  mode          : {mode}")

    print("\n  sources:")
    session_dir = paths.outlier_session_dir()
    try:
        state = session.load_storage_state()
        token = session.extract_bearer_token(state)
        cookie = session.build_cookie_header(state)
        detail = []
        detail.append("bearer token present" if token else "NO bearer token")
        detail.append(f"{len(cookie.split('; ')) if cookie else 0} cookies")
        print(f"    outlier   : OK   ({', '.join(detail)}) [{session_dir}]")
    except CfbAnalyticsError as exc:
        print(f"    outlier   : FAIL {exc}")

    if config.has_cfbd_key():
        print("    cfbd      : OK   (key configured)")
    else:
        print(f"    cfbd      : BLOCKED - set {config.CFBD_ENV_VAR}. {config.CFBD_HOW}")
    print("    espn      : dropped - ESPN publishes no CFB depth chart (see sources/__init__)")
    print("    weather   : OK   (Open-Meteo; no credential required)")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from cfb_analytics.ingest.outlier_ingest import ingest_slate
    from cfb_analytics.sources.outlier import OutlierClient

    paths.ensure_dirs()
    client = OutlierClient()
    with db.open_db() as conn:
        summary = ingest_slate(
            conn,
            client,
            args.date,
            with_odds=not args.no_odds,
            with_injuries=not args.no_injuries,
            with_props=bool(getattr(args, "with_props", False)),
            limit=args.limit,
        )
    print(summary.as_text())
    if summary.events_seen == 0:
        print(
            "\nNo events on that date. Outlier carries a forward schedule only; "
            "use `cfb-analytics schedule` to list available dates."
        )
    return 0


def _cmd_backfill_cfbd(args: argparse.Namespace) -> int:
    from cfb_analytics.errors import SchemaError
    from cfb_analytics.ingest.cfbd_ingest import backfill_years
    from cfb_analytics.sources.cfbd import CFBDClient

    if args.start_year > args.end_year:
        raise SchemaError("start-year must be less than or equal to end-year")
    paths.ensure_dirs()
    client = CFBDClient()
    with db.open_db() as conn:
        summary = backfill_years(
            conn,
            client,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    print(summary.as_text())
    return 0


def _cmd_backfill_roster(args: argparse.Namespace) -> int:
    from cfb_analytics.errors import SchemaError
    from cfb_analytics.ingest.cfbd_players import ingest_roster
    from cfb_analytics.sources.cfbd import CFBDClient

    if args.start_year > args.end_year:
        raise SchemaError("start-year must be less than or equal to end-year")
    paths.ensure_dirs()
    client = CFBDClient()
    with db.open_db() as conn:
        for year in range(args.start_year, args.end_year + 1):
            summary = ingest_roster(conn, client, year)
            print(summary.as_text())
    return 0


def _cmd_backfill_passing(args: argparse.Namespace) -> int:
    from cfb_analytics.errors import SchemaError
    from cfb_analytics.ingest.cfbd_players import completed_weeks, ingest_game_passing
    from cfb_analytics.sources.cfbd import CFBDClient

    if args.start_year > args.end_year:
        raise SchemaError("start-year must be less than or equal to end-year")
    paths.ensure_dirs()
    client = CFBDClient()
    with db.open_db() as conn:
        for year in range(args.start_year, args.end_year + 1):
            for week in completed_weeks(conn, year, season_type=args.season_type):
                summary = ingest_game_passing(
                    conn, client, year, week, season_type=args.season_type
                )
                print(summary.as_text())
    return 0


def _cmd_backfill_fundamentals(args: argparse.Namespace) -> int:
    from cfb_analytics.errors import SchemaError
    from cfb_analytics.ingest.cfbd_fundamentals import backfill_fundamentals
    from cfb_analytics.sources.cfbd import CFBDClient

    if args.start_year > args.end_year:
        raise SchemaError("start-year must be less than or equal to end-year")
    paths.ensure_dirs()
    client = CFBDClient()
    with db.open_db() as conn:
        summary = backfill_fundamentals(
            conn,
            client,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    print(summary.as_text())
    return 0


def _cmd_backfill_elo(args: argparse.Namespace) -> int:
    """Backfill ONLY weekly CFBD Elo ratings.

    A separate command from ``backfill-fundamentals``: Elo's rows were the
    only ones a since-fixed CHECK-constraint bug silently dropped (see
    ``parse_elo_rating``'s docstring), so a full fundamentals re-run to pick
    up the fix would waste four other endpoints' worth of API calls per
    season for data that is already correct in the store.
    """
    from cfb_analytics.errors import SchemaError
    from cfb_analytics.ingest.cfbd_fundamentals import backfill_elo
    from cfb_analytics.sources.cfbd import CFBDClient

    if args.start_year > args.end_year:
        raise SchemaError("start-year must be less than or equal to end-year")
    paths.ensure_dirs()
    client = CFBDClient()
    with db.open_db() as conn:
        summary = backfill_elo(conn, client, start_year=args.start_year, end_year=args.end_year)
    print(summary.as_text())
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward moneyline backtest of the internal ridge model.

    See ``backtest/moneyline.py`` for why this reports calibration only, not
    a promotion decision: none of the three required baselines (market,
    SP+-only, Elo-only) have a leakage-safe historical series in this store
    yet.
    """
    from cfb_analytics.backtest.moneyline import DEFAULT_SEASONS, run_moneyline_backtest
    from cfb_analytics.errors import SchemaError

    if (args.start_year is None) != (args.end_year is None):
        raise SchemaError("--start-year and --end-year must be given together")
    seasons = (
        tuple(range(args.start_year, args.end_year + 1))
        if args.start_year is not None
        else DEFAULT_SEASONS
    )
    paths.ensure_dirs()
    with db.open_db() as conn:
        report = run_moneyline_backtest(conn, seasons)
    print(report.as_text())
    return 0


def _cmd_fit_ratings(args: argparse.Namespace) -> int:
    """Fit and persist internal ridge team-strength ratings as of a cutoff.

    Defaults ``--as-of`` to right now: the useful case for a scheduled run is
    "the freshest legal fit," and any earlier cutoff (a backtest walking
    forward through a season) is what ``--as-of`` exists to override.
    """
    from cfb_analytics.features.team_ratings import fit_ratings_as_of
    from cfb_analytics.ingest.store import upsert_internal_team_ratings
    from cfb_analytics.models.ridge import DEFAULT_RIDGE_LAMBDA
    from cfb_analytics.utils import utc_now_iso

    as_of_utc = args.as_of or utc_now_iso()
    ridge_lambda = args.ridge_lambda if args.ridge_lambda is not None else DEFAULT_RIDGE_LAMBDA
    paths.ensure_dirs()
    with db.open_db() as conn:
        ratings = fit_ratings_as_of(conn, args.season, as_of_utc, ridge_lambda=ridge_lambda)
        written = upsert_internal_team_ratings(
            conn, ratings, season=args.season, as_of_utc=as_of_utc
        )
    print(
        f"internal ridge fit: season={args.season} as_of={as_of_utc} "
        f"status={ratings.status} n_games={ratings.n_games} teams_written={written}"
    )
    return 0


def _cmd_fit_elo(args: argparse.Namespace) -> int:
    """Fit and persist internal Elo ratings as of a cutoff.

    Defaults ``--as-of`` to right now, same reasoning as ``fit-ratings``.
    """
    from cfb_analytics.features.elo_internal import fit_internal_elo_as_of
    from cfb_analytics.ingest.store import upsert_internal_elo_ratings
    from cfb_analytics.utils import utc_now_iso

    as_of_utc = args.as_of or utc_now_iso()
    paths.ensure_dirs()
    with db.open_db() as conn:
        ratings = fit_internal_elo_as_of(conn, args.season, as_of_utc)
        written = upsert_internal_elo_ratings(
            conn, ratings, season=args.season, as_of_utc=as_of_utc
        )
    print(
        f"internal elo fit: season={args.season} as_of={as_of_utc} "
        f"status={ratings.status} n_games={ratings.n_games} teams_written={written}"
    )
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    from collections import Counter
    from datetime import datetime

    from cfb_analytics.errors import SchemaError
    from cfb_analytics.sources.outlier import OutlierClient, parse_event
    from cfb_analytics.utils import FOOTBALL_TZ

    weekday_names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

    client = OutlierClient()
    counts: Counter[str] = Counter()
    weekdays: dict[str, str] = {}
    bad = mismatches = 0
    for event in client.fetch_schedule():
        try:
            record = parse_event(event)
        except SchemaError:
            bad += 1
            continue
        date = record["football_date"]
        counts[date] += 1
        weekdays[date] = weekday_names[
            datetime.fromisoformat(record["kickoff_utc"]).astimezone(FOOTBALL_TZ).weekday()
        ]
        if record["weekday_agrees"] is False:
            mismatches += 1

    print(f"{sum(counts.values())} scheduled events across {len(counts)} slate dates")
    print("(slate date = US Eastern calendar date of kickoff, not the UTC date)")
    if bad:
        print(f"({bad} events skipped: unparseable schema)")
    if mismatches:
        print(f"WARNING: {mismatches} events disagree with the feed's own dayOfWeek code")
    for date, count in sorted(counts.items()):
        print(f"  {date}  {weekdays[date]:<3} {count:>3}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    if not paths.database_path().exists():
        print("No database yet. Run: cfb-analytics init-db")
        return 1
    with db.open_db() as conn:
        for table in (
            "games",
            "teams",
            "team_seasons",
            "team_aliases",
            "venues",
            "team_ratings",
            "team_season_advanced",
            "returning_production",
            "team_talent",
            "odds_snapshots",
            "availability",
            "runs",
        ):
            count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            print(f"  {table:<25} {count:>8}")
        row = conn.execute(
            "SELECT command, started_utc, status, rows_written FROM runs "
            "ORDER BY started_utc DESC LIMIT 1"
        ).fetchone()
        if row:
            print(
                f"\n  last run: {row['command']} at {row['started_utc']} "
                f"-> {row['status']} ({row['rows_written']} rows)"
            )
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    """Book coverage per capture.

    Book coverage is time-varying: a 2026-08-31 probe saw 20 books including
    PS3838 (Pinnacle) and CIRCA on the same games that returned 11 books and no
    sharp books hours later. The sharp-anchor devig described in the plan is
    only viable if sharp books are reliably present, so coverage is measured
    across captures rather than asserted from one snapshot.
    """
    from datetime import datetime

    sharp = set(config.sources()["outlier"]["sharp_books"])
    if not paths.database_path().exists():
        print("No database yet. Run: cfb-analytics init-db")
        return 1
    with db.open_db() as conn:
        rows = conn.execute(
            """SELECT g.football_date          AS slate,
                      o.captured_utc           AS captured,
                      MIN(g.kickoff_utc)       AS first_kick,
                      COUNT(DISTINCT g.game_id) AS games,
                      COUNT(DISTINCT o.book)   AS books,
                      COUNT(*)                 AS prices,
                      GROUP_CONCAT(DISTINCT o.book) AS book_list
               FROM odds_snapshots o JOIN games g ON g.game_id = o.game_id
               GROUP BY g.football_date, o.captured_utc
               ORDER BY g.football_date, o.captured_utc"""
        ).fetchall()
        if not rows:
            print("No odds captured yet. Run: cfb-analytics ingest --date <YYYY-MM-DD>")
            return 0

        print(
            f"{'slate':<12} {'captured (UTC)':<21} {'d-to-kick':>9} {'games':>5} "
            f"{'books':>5} {'prices':>7} {'px/game':>8}  sharp"
        )
        any_sharp = False
        for row in rows:
            present = sorted(sharp & set((row["book_list"] or "").split(",")))
            any_sharp = any_sharp or bool(present)
            lead = (
                datetime.fromisoformat(row["first_kick"]) - datetime.fromisoformat(row["captured"])
            ).days
            print(
                f"{row['slate']:<12} {row['captured'][:19]:<21} {lead:>9} "
                f"{row['games']:>5} {row['books']:>5} {row['prices']:>7} "
                f"{row['prices'] // max(row['games'], 1):>8}  "
                f"{', '.join(present) if present else 'NONE'}"
            )

        all_books = conn.execute(
            "SELECT DISTINCT book FROM odds_snapshots ORDER BY book"
        ).fetchall()
        print(f"\n  books ever seen ({len(all_books)}): {', '.join(r['book'] for r in all_books)}")
        print(f"  sharp set tracked: {', '.join(sorted(sharp))}")
        if not any_sharp:
            print(
                "\n  No sharp book has appeared in ANY capture. The sharp-anchor devig is "
                "not currently supported by this feed:\n"
                "  consensus must fall back to all books, rows carry `no_sharp_anchor`, "
                "and CLV must be measured\n  against best-available price rather than a "
                "Pinnacle close."
            )
    return 0


def _cmd_market(args: argparse.Namespace) -> int:
    """Compute vig-free market consensus from stored odds."""
    from cfb_analytics.features.build_market import build_market_for_slate

    with db.open_db() as conn:
        summary = build_market_for_slate(conn, args.date)
    print(summary.as_text())
    if summary.games == 0:
        print(f"\nNo games stored for that slate. Run: cfb-analytics ingest --date {args.date}")
    return 0


def _cmd_board(args: argparse.Namespace) -> int:
    """Moneyline board for a slate: the M model's output, ranked."""
    if not paths.database_path().exists():
        print("No database yet. Run: cfb-analytics init-db")
        return 1
    with db.open_db() as conn:
        rows = conn.execute(
            """SELECT g.game_id, g.kickoff_utc,
                      ht.alias AS home, at.alias AS away,
                      c.side, c.consensus_price, c.best_price, c.best_book,
                      c.prob_shin, c.prob_multiplicative, c.prob_power,
                      c.prob_spread, c.hold, c.n_books, c.anchor, c.flags
               FROM market_consensus c
               JOIN games g ON g.game_id = c.game_id
               JOIN teams ht ON ht.team_id = g.home_team_id
               JOIN teams at ON at.team_id = g.away_team_id
               WHERE g.football_date = ? AND c.market = 'ML'
               ORDER BY c.prob_shin DESC""",
            (args.date,),
        ).fetchall()
    if not rows:
        print(
            f"No moneyline consensus for {args.date}. Run: cfb-analytics market --date {args.date}"
        )
        return 0

    print(
        f"MONEYLINE BOARD - {args.date}   [{config.SHADOW_STAMP}]"
        if config.is_shadow_mode()
        else f"MONEYLINE BOARD - {args.date}"
    )
    print(
        f"\n{'team':<7} {'opp':<7} {'price':>7} {'best':>7} {'book':<11} "
        f"{'fair%':>7} {'spread':>7} {'hold':>6} {'bk':>3}  flags"
    )
    for row in rows:
        prob = row["prob_shin"] or row["prob_multiplicative"]
        if prob is None or prob < args.min_prob:
            continue
        team = row["home"] if row["side"] == "HOME" else row["away"]
        opp = row["away"] if row["side"] == "HOME" else row["home"]
        flags = ",".join(json.loads(row["flags"] or "[]"))
        print(
            f"{team or '?':<7} {opp or '?':<7} {row['consensus_price']:>7} "
            f"{row['best_price']:>7} {(row['best_book'] or ''):<11} "
            f"{prob * 100:>6.1f}% {(row['prob_spread'] or 0) * 100:>6.2f}pp "
            f"{(row['hold'] or 0) * 100:>5.1f}% {row['n_books']:>3}  {flags}"
        )
    print(
        "\nfair% is the vig-free market probability (Shin). spread is the "
        "disagreement\nbetween devig methods - wide means the fair number is "
        "method-dependent."
    )
    if config.is_shadow_mode():
        print("This is the MARKET's view only. No model probability or edge exists yet.")
    return 0


def _cmd_futures(args: argparse.Namespace) -> int:
    """Emit season futures projection summary or JSON."""
    import math
    from dataclasses import asdict
    from datetime import UTC, datetime

    from cfb_analytics.errors import SchemaError
    from cfb_analytics.features.futures import project_team_futures_from_db

    try:
        as_of = datetime.fromisoformat(args.as_of)
    except ValueError as exc:
        raise SchemaError("--as-of must be a parseable ISO timestamp") from exc
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise SchemaError("--as-of must include a UTC offset (for example, +00:00 or Z)")
    canonical_as_of = as_of.astimezone(UTC).isoformat()
    input_manifest: dict[str, object] = {}

    parsed_lines: list[float] = []
    if args.posted_lines:
        for item in args.posted_lines:
            if isinstance(item, (int, float)):
                val = float(item)
                if val < 0:
                    raise SchemaError(f"Posted win-total line cannot be negative, got {val}")
                parsed_lines.append(val)
            elif isinstance(item, str):
                for part in item.split(","):
                    part = part.strip()
                    if part:
                        try:
                            val = float(part)
                        except ValueError as exc:
                            raise SchemaError(f"Invalid line value {part!r}") from exc
                        if val < 0:
                            raise SchemaError(
                                f"Posted win-total line cannot be negative, got {val}"
                            )
                        parsed_lines.append(val)

    if any(not math.isfinite(line) for line in parsed_lines):
        raise SchemaError("Posted win-total lines must be finite")
    if args.posted_lines and not parsed_lines:
        raise SchemaError("At least one posted win-total line is required")
    if args.mc_sims is not None and args.mc_sims < 1:
        raise SchemaError("--mc-sims must be a positive integer")

    nil_tier = args.nil_tier
    nil_budget = args.nil_budget_millions

    portal_comp = args.portal_net_composite
    qb_tier = args.qb_tier
    qb_continuity = args.qb_continuity

    with db.open_db() as conn:
        projection = project_team_futures_from_db(
            conn,
            args.team_id,
            args.season,
            as_of_utc=canonical_as_of,
            portal_composite=portal_comp,
            nil_tier=nil_tier,
            nil_budget_millions=nil_budget,
            qb_tier=qb_tier,
            qb_continuity=qb_continuity,
            posted_lines=parsed_lines,
            input_manifest=input_manifest,
        )

    sim_results = None
    if getattr(args, "mc_sims", None) is not None:
        from cfb_analytics.models.futures import simulate_season_monte_carlo

        probs = [gp.win_probability for gp in projection.schedule_projections]
        sim_results = simulate_season_monte_carlo(
            probs, posted_lines=parsed_lines, n_simulations=args.mc_sims
        )

    output_json = bool(getattr(args, "json", False))

    if output_json:
        payload = {
            "schema_version": 1,
            "model": "season_futures_v1",
            "model_status": "uncalibrated_shadow",
            "is_actionable": False,
            "as_of_utc": canonical_as_of,
            "season": args.season,
            "inputs": {
                "provenance": "caller_supplied_unverified",
                "portal_net_composite": portal_comp,
                "nil_tier": nil_tier,
                "nil_budget_millions": nil_budget,
                "qb_tier": qb_tier,
                "qb_continuity": qb_continuity,
                "posted_lines": parsed_lines,
            },
            "database_inputs": input_manifest,
            "projection": asdict(projection),
        }
        if sim_results is not None:
            payload["simulation"] = asdict(sim_results)
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
        return 0

    print(f"SEASON FUTURES PROJECTION - {args.team_id} ({args.season})   [{config.SHADOW_STAMP}]")
    rating = projection.adjusted_rating
    print(f"\n  True Talent Composite : {rating.true_talent_composite}")
    print(
        f"  Adjusted Power Rating : {rating.adjusted_power_rating:+.2f} "
        f"(Elo ~{rating.elo_equivalent:.0f})"
    )
    print(
        f"  Expected Wins         : {projection.expected_wins:.2f} "
        f"(stdev: {projection.win_stdev:.2f})"
    )
    print(
        f"  Conference ({projection.conference}) : "
        f"CCG Reach: {projection.prob_reach_conference_championship * 100:.1f}%, "
        f"CCG Win: {projection.prob_win_conference_championship * 100:.1f}%"
    )
    print(f"  CFP Appearance (12-tm): {projection.prob_cfp_appearance * 100:.1f}%")

    if projection.win_total_evaluations:
        print("\n  Win Total Line Evaluations:")
        print(
            f"    {'Line':>5}  {'Over %':>7}  {'Under %':>7}  "
            f"{'Fair Over':>9}  {'Fair Under':>10}  {'Edge':>7}  {'Side':<5}"
        )
        print("    " + "-" * 57)
        for line, eval_item in sorted(projection.win_total_evaluations.items()):
            over_pct = f"{eval_item.prob_over * 100:.1f}%"
            under_pct = f"{eval_item.prob_under * 100:.1f}%"
            fair_o = (
                f"{eval_item.fair_over_american:+d}"
                if eval_item.fair_over_american is not None
                else "N/A"
            )
            fair_u = (
                f"{eval_item.fair_under_american:+d}"
                if eval_item.fair_under_american is not None
                else "N/A"
            )
            edge_str = f"{eval_item.edge * 100:+.1f}%"
            print(
                f"    {line:>5.1f}  {over_pct:>7}  {under_pct:>7}  "
                f"{fair_o:>9}  {fair_u:>10}  {edge_str:>7}  {eval_item.recommended_side:<5}"
            )

    if sim_results is not None:
        print(f"\n  Monte Carlo Simulation ({sim_results.n_simulations:,} trials):")
        print(f"    Mean Wins   : {sim_results.mean_wins:.2f}")
        print(f"    Median Wins : {sim_results.median_wins:.1f}")
        print(f"    P10 - P90   : {sim_results.p10_wins:.1f} - {sim_results.p90_wins:.1f} wins")
        if sim_results.simulated_line_over_probs:
            print("    Simulated Over Probabilities:")
            for line, prob in sorted(sim_results.simulated_line_over_probs.items()):
                print(f"      Over {line:<4.1f}: {prob * 100:>5.1f}%")

    print("\nThis projection is uncalibrated shadow output. Not an actionable recommendation.")
    return 0


def _parse_clock(value: str | int | None) -> int:
    if value is None:
        return 900
    if isinstance(value, int):
        if not (0 <= value <= 900):
            raise SchemaError(f"Clock must be between 0 and 900 seconds (15:00), got {value}")
        return value
    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            try:
                mins = int(parts[0])
                secs = int(parts[1])
            except ValueError as exc:
                raise SchemaError(
                    f"--clock format must be MM:SS or integer seconds, got {value!r}"
                ) from exc
            if mins < 0 or not (0 <= secs <= 59):
                raise SchemaError(f"--clock format must be MM:SS with 0-59 seconds, got {value!r}")
            total = mins * 60 + secs
            if not (0 <= total <= 900):
                raise SchemaError(f"Clock must be between 0 and 900 seconds (15:00), got {value!r}")
            return total
        raise SchemaError(f"--clock format must be MM:SS or integer seconds, got {value!r}")
    try:
        total = int(text)
    except ValueError as exc:
        raise SchemaError(
            f"--clock format must be MM:SS or integer seconds, got {value!r}"
        ) from exc
    if not (0 <= total <= 900):
        raise SchemaError(f"Clock must be between 0 and 900 seconds (15:00), got {total}")
    return total


def _cmd_live(args: argparse.Namespace) -> int:
    """Track continuous in-game state and simulate live micro-markets."""
    from datetime import UTC, datetime

    from cfb_analytics.features.live import GameState
    from cfb_analytics.models.live import (
        calculate_win_probability,
        estimate_next_drive_outcomes,
        project_live_team_totals,
        win_probability_to_american_odds,
    )

    is_completed_in_db = False
    if args.game_id:
        paths.ensure_dirs()
        if not paths.database_path().exists():
            raise CfbAnalyticsError(
                "Database does not exist. Run 'cfb-analytics init-db' or provide --home and --away."
            )
        with db.open_db() as conn:
            row = conn.execute(
                "SELECT game_id, home_team_id, away_team_id, home_points, away_points, "
                "status, completed "
                "FROM games WHERE game_id = ?",
                (args.game_id,),
            ).fetchone()
        if row is None:
            raise CfbAnalyticsError(f"Game {args.game_id!r} not found in database")
        if args.home or args.away:
            raise SchemaError("--home and --away cannot override a stored game identity")
        home_team = str(row["home_team_id"]).strip()
        away_team = str(row["away_team_id"]).strip()
        home_score = (
            args.home_score
            if args.home_score is not None
            else (int(row["home_points"]) if row["home_points"] is not None else 0)
        )
        away_score = (
            args.away_score
            if args.away_score is not None
            else (int(row["away_points"]) if row["away_points"] is not None else 0)
        )
        game_id = args.game_id
        if row["completed"] == 1 or (
            row["status"] and str(row["status"]).lower() in ("completed", "final", "status_final")
        ):
            is_completed_in_db = True
        if is_completed_in_db and (row["home_points"] is None or row["away_points"] is None):
            raise SchemaError("Completed database game is missing final scores")
        if not is_completed_in_db:
            required = ("quarter", "clock", "down", "distance", "yardline", "possession")
            missing = [f"--{name}" for name in required if getattr(args, name) is None]
            if missing:
                raise SchemaError("Database has no live situation; provide " + ", ".join(missing))
            if (row["home_points"] is None and args.home_score is None) or (
                row["away_points"] is None and args.away_score is None
            ):
                raise SchemaError("Database scores are missing; provide explicit score flags")
    else:
        if not args.home or not args.away:
            raise SchemaError("Either --game or both --home and --away must be specified")
        home_team = args.home.strip()
        away_team = args.away.strip()
        home_score = args.home_score if args.home_score is not None else 0
        away_score = args.away_score if args.away_score is not None else 0
        game_id = f"sim_{home_team}_{away_team}"

    if home_team.lower() == away_team.lower():
        raise SchemaError("Home and away team IDs must differ")

    if home_score < 0 or away_score < 0:
        raise SchemaError("Scores cannot be negative")

    possession = (args.possession or home_team).strip()
    if possession not in (home_team, away_team):
        raise SchemaError(
            f"Possession team {possession!r} must be either home ({home_team!r}) "
            f"or away ({away_team!r})"
        )

    default_quarter = 4 if is_completed_in_db else 1
    quarter = args.quarter if args.quarter is not None else default_quarter
    if quarter < 1:
        raise SchemaError(f"Quarter must be >= 1, got {quarter}")

    default_clock = 0 if is_completed_in_db else 900
    clock_seconds = _parse_clock(args.clock) if args.clock is not None else default_clock

    down = args.down if args.down is not None else 1
    if not (1 <= down <= 4):
        raise SchemaError(f"Down must be in [1, 4], got {down}")

    distance = args.distance if args.distance is not None else 10
    if not (1 <= distance <= 99):
        raise SchemaError(f"Distance must be in [1, 99], got {distance}")

    yardline = args.yardline if args.yardline is not None else 75
    if not (1 <= yardline <= 99):
        raise SchemaError(f"Yardline must be in [1, 99], got {yardline}")

    # Overtime has no regulation clock: a lead at 0:00 does not establish a winner.
    is_final = bool(args.final or is_completed_in_db)
    if quarter == 4 and clock_seconds == 0 and home_score != away_score:
        is_final = True

    if is_final and home_score == away_score:
        raise SchemaError("A final college-football game cannot have a tied score")

    now_iso = datetime.now(UTC).isoformat()
    state = GameState(
        home_team_id=home_team,
        away_team_id=away_team,
        possession_team_id=possession,
        quarter=quarter,
        clock_seconds=clock_seconds,
        down=down,
        distance=distance,
        yardline=yardline,
        home_score=home_score,
        away_score=away_score,
        game_id=game_id,
        observed_utc=now_iso,
        ingested_utc=now_iso,
        source="cli_live",
        source_sequence=0,
        is_final=is_final,
    )

    pregame_margin = float(args.pregame_margin or 0.0)
    home_wp = calculate_win_probability(
        state, pregame_home_margin=pregame_margin, team_id=home_team
    )
    away_wp = round(1.0 - home_wp, 4)
    home_odds = win_probability_to_american_odds(home_wp)
    away_odds = win_probability_to_american_odds(away_wp)

    drive_outcomes = None if state.is_final else estimate_next_drive_outcomes(state.yardline)
    totals = project_live_team_totals(state)

    if getattr(args, "json", False):
        payload = {
            "schema_version": 1,
            "model": "live_micro_markets_v1",
            "model_status": "uncalibrated_shadow",
            "is_actionable": False,
            "game_id": game_id,
            "state": {
                "home_team_id": home_team,
                "away_team_id": away_team,
                "possession_team_id": possession,
                "quarter": quarter,
                "clock_seconds": clock_seconds,
                "clock_display": f"{clock_seconds // 60}:{clock_seconds % 60:02d}",
                "down": down,
                "distance": distance,
                "yardline": yardline,
                "home_score": home_score,
                "away_score": away_score,
                "home_timeouts": state.home_timeouts,
                "away_timeouts": state.away_timeouts,
                "is_final": state.is_final,
            },
            "win_probability": {
                "home_team_id": home_team,
                "away_team_id": away_team,
                "home_win_prob": home_wp,
                "away_win_prob": away_wp,
                "home_american_odds": home_odds,
                "away_american_odds": away_odds,
            },
            "next_drive_outcome": None
            if drive_outcomes is None
            else {
                "possession_team_id": possession,
                "start_yardline": yardline,
                "touchdown": drive_outcomes.touchdown,
                "field_goal": drive_outcomes.field_goal,
                "punt": drive_outcomes.punt,
                "turnover_downs": drive_outcomes.turnover_downs,
                "safety": drive_outcomes.safety,
                "expected_points": drive_outcomes.expected_points,
            },
            "projected_totals": {
                "home": {
                    "team_id": home_team,
                    "current_score": totals.home.current_score,
                    "projected_total": totals.home.projected_total,
                    "remaining_expected_points": totals.home.remaining_expected_points,
                    "remaining_possessions": totals.home.remaining_possessions,
                },
                "away": {
                    "team_id": away_team,
                    "current_score": totals.away.current_score,
                    "projected_total": totals.away.projected_total,
                    "remaining_expected_points": totals.away.remaining_expected_points,
                    "remaining_possessions": totals.away.remaining_possessions,
                },
                "projected_game_total": totals.projected_game_total,
            },
        }
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
        return 0

    clock_str = f"{clock_seconds // 60}:{clock_seconds % 60:02d}"
    print(f"LIVE GAME STATE & MICRO-MARKETS   [{config.SHADOW_STAMP}]")
    print(f"\n  Game       : {home_team} vs {away_team} (ID: {game_id})")
    status_suffix = " [FINAL]" if state.is_final else ""
    print(
        f"  Situation  : Q{quarter} {clock_str}{status_suffix} | Down {down} & {distance} "
        f"at yardline {yardline} "
        f"({possession} ball)"
    )
    print(f"  Score      : {home_team} {home_score} - {away_score} {away_team}")

    home_odds_str = f"{home_odds:+d}" if home_odds is not None else "N/A"
    away_odds_str = f"{away_odds:+d}" if away_odds is not None else "N/A"
    print("\n  Win Probability:")
    print(f"    {home_team:<20} : {home_wp * 100:>6.1f}% ({home_odds_str})")
    print(f"    {away_team:<20} : {away_wp * 100:>6.1f}% ({away_odds_str})")

    if drive_outcomes is None:
        print("\n  Next Drive: N/A (game is final)")
    else:
        print(f"\n  Next Drive Outcome Distribution ({possession} at yardline {yardline}):")
        print(f"    Touchdown (TD)        : {drive_outcomes.touchdown * 100:>5.1f}%")
        print(f"    Field Goal (FG)       : {drive_outcomes.field_goal * 100:>5.1f}%")
        print(f"    Punt                  : {drive_outcomes.punt * 100:>5.1f}%")
        print(f"    Turnover / Downs      : {drive_outcomes.turnover_downs * 100:>5.1f}%")
        print(f"    Safety                : {drive_outcomes.safety * 100:>5.1f}%")
        print(f"    Drive Expected Points : {drive_outcomes.expected_points:>+6.2f} pts")

    print("\n  Live Projected Totals:")
    print(
        f"    {home_team:<20} Projected: {totals.home.projected_total:>5.1f} pts  "
        f"(current: {home_score}, rem exp: {totals.home.remaining_expected_points:>+5.1f})"
    )
    print(
        f"    {away_team:<20} Projected: {totals.away.projected_total:>5.1f} pts  "
        f"(current: {away_score}, rem exp: {totals.away.remaining_expected_points:>+5.1f})"
    )
    print(f"    Projected Game Total : {totals.projected_game_total:>5.1f} pts")

    print("\nThis simulation is uncalibrated shadow output. Not an actionable recommendation.")
    return 0


def _cmd_daily(args: argparse.Namespace) -> int:
    """The scheduled job: ingest what is credentialed, then rebuild the market."""
    from cfb_analytics.daily import run_daily

    paths.ensure_dirs()
    with db.open_db() as conn:
        report = run_daily(
            conn,
            season=args.season,
            with_outlier=not args.no_outlier,
            with_weather=not args.no_weather,
            with_player_passing=not args.no_player_passing,
            with_internal_ratings=not args.no_internal_ratings,
            with_internal_elo=not args.no_internal_elo,
            with_props=bool(getattr(args, "with_props", False)),
            with_scoring=bool(getattr(args, "with_scoring", False)),
            bootstrap=not args.no_bootstrap,
        )
    text = report.as_text()
    print(text)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            fence = "```"
            handle.write(f"## Daily ingest\n\n{fence}\n{text}\n{fence}\n")

    # Exit non-zero only when EVERY source failed. A skipped source is an
    # expected state, not a red build every morning.
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfb-analytics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create or migrate the SQLite store").set_defaults(
        func=_cmd_init_db
    )
    sub.add_parser("doctor", help="report source and credential readiness").set_defaults(
        func=_cmd_doctor
    )
    sub.add_parser("schedule", help="list available slate dates from Outlier").set_defaults(
        func=_cmd_schedule
    )
    sub.add_parser("status", help="row counts and last run").set_defaults(func=_cmd_status)
    sub.add_parser(
        "coverage", help="book coverage per capture, and whether sharp books appear"
    ).set_defaults(func=_cmd_coverage)

    ingest = sub.add_parser("ingest", help="ingest one slate from Outlier")
    ingest.add_argument("--date", required=True, help="slate date, YYYY-MM-DD")
    ingest.add_argument("--no-odds", action="store_true", help="skip gameline odds")
    ingest.add_argument("--no-injuries", action="store_true", help="skip the injury feed")
    ingest.add_argument(
        "--with-props", action="store_true", help="ingest supported full-game team props"
    )
    ingest.add_argument("--limit", type=int, default=None, help="cap events (for smoke tests)")
    ingest.set_defaults(func=_cmd_ingest)

    cfbd = sub.add_parser("backfill-cfbd", help="backfill historical FBS teams, venues, and games")
    cfbd.add_argument("--start-year", type=int, required=True, help="first season year, inclusive")
    cfbd.add_argument("--end-year", type=int, required=True, help="last season year, inclusive")
    cfbd.set_defaults(func=_cmd_backfill_cfbd)

    roster = sub.add_parser(
        "backfill-roster", help="backfill CFBD season rosters (position, class year)"
    )
    roster.add_argument(
        "--start-year", type=int, required=True, help="first season year, inclusive"
    )
    roster.add_argument("--end-year", type=int, required=True, help="last season year, inclusive")
    roster.set_defaults(func=_cmd_backfill_roster)

    passing = sub.add_parser(
        "backfill-passing", help="backfill CFBD per-game passing stats for completed weeks"
    )
    passing.add_argument(
        "--start-year", type=int, required=True, help="first season year, inclusive"
    )
    passing.add_argument("--end-year", type=int, required=True, help="last season year, inclusive")
    passing.add_argument("--season-type", default="regular", help="regular | postseason")
    passing.set_defaults(func=_cmd_backfill_passing)

    fundamentals = sub.add_parser(
        "backfill-fundamentals",
        help="backfill CFBD SP+, SRS, Elo, advanced stats, returning production, and talent",
    )
    fundamentals.add_argument(
        "--start-year", type=int, required=True, help="first season year, inclusive"
    )
    fundamentals.add_argument(
        "--end-year", type=int, required=True, help="last season year, inclusive"
    )
    fundamentals.set_defaults(func=_cmd_backfill_fundamentals)

    elo_cmd = sub.add_parser(
        "backfill-elo", help="backfill ONLY weekly CFBD Elo ratings (see backfill-fundamentals)"
    )
    elo_cmd.add_argument(
        "--start-year", type=int, required=True, help="first season year, inclusive"
    )
    elo_cmd.add_argument("--end-year", type=int, required=True, help="last season year, inclusive")
    elo_cmd.set_defaults(func=_cmd_backfill_elo)

    fit_ratings = sub.add_parser(
        "fit-ratings", help="fit and persist internal ridge team-strength ratings"
    )
    fit_ratings.add_argument("--season", type=int, required=True)
    fit_ratings.add_argument("--as-of", default=None, help="ISO cutoff timestamp (default: now)")
    fit_ratings.add_argument(
        "--ridge-lambda", type=float, default=None, help="override the default ridge penalty"
    )
    fit_ratings.set_defaults(func=_cmd_fit_ratings)

    fit_elo = sub.add_parser("fit-elo", help="fit and persist internal Elo ratings")
    fit_elo.add_argument("--season", type=int, required=True)
    fit_elo.add_argument("--as-of", default=None, help="ISO cutoff timestamp (default: now)")
    fit_elo.set_defaults(func=_cmd_fit_elo)

    backtest_cmd = sub.add_parser(
        "backtest", help="walk-forward moneyline backtest of the internal ridge model"
    )
    backtest_cmd.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="first season, inclusive (default: 2014, the full stored history)",
    )
    backtest_cmd.add_argument(
        "--end-year", type=int, default=None, help="last season, inclusive (default: 2025)"
    )
    backtest_cmd.set_defaults(func=_cmd_backtest)

    market_cmd = sub.add_parser("market", help="compute vig-free consensus from stored odds")
    market_cmd.add_argument("--date", required=True, help="slate date, YYYY-MM-DD")
    market_cmd.set_defaults(func=_cmd_market)

    board = sub.add_parser("board", help="moneyline board for a slate")
    board.add_argument("--date", required=True, help="slate date, YYYY-MM-DD")
    board.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="only show sides at or above this fair probability",
    )
    board.set_defaults(func=_cmd_board)

    from cfb_analytics.models.futures import NILTier, QBContinuity, QBTier

    futures = sub.add_parser(
        "futures",
        help="season futures projections, win total evaluations, and CFP odds",
    )
    futures.add_argument(
        "--team", "--team-id", dest="team_id", required=True, help="canonical CFBD team ID"
    )
    futures.add_argument("--season", type=int, required=True, help="season year (e.g. 2026)")
    futures.add_argument(
        "--as-of",
        required=True,
        help="required ISO point-in-time cutoff; never defaults to now",
    )
    futures.add_argument(
        "--lines",
        "--posted-line",
        dest="posted_lines",
        action="extend",
        nargs="+",
        default=None,
        help="posted win total lines; repeated, space-separated, or comma-separated",
    )
    futures.add_argument(
        "--mc-sims",
        type=int,
        default=None,
        help="number of Monte Carlo season simulations (e.g. 10000)",
    )
    futures.add_argument(
        "--portal-net-composite",
        type=float,
        default=None,
        help="externally sourced transfer-portal net composite (required by DB projection)",
    )
    nil_source = futures.add_mutually_exclusive_group(required=False)
    nil_source.add_argument(
        "--nil-tier",
        choices=tuple(tier.value for tier in NILTier),
        default=None,
        help="externally sourced NIL tier",
    )
    nil_source.add_argument(
        "--nil-budget-millions",
        type=float,
        default=None,
        help="externally sourced NIL budget estimate in millions",
    )
    futures.add_argument(
        "--qb-tier",
        choices=tuple(tier.value for tier in QBTier),
        default=None,
        help="externally assessed quarterback tier",
    )
    futures.add_argument(
        "--qb-continuity",
        choices=tuple(state.value for state in QBContinuity),
        default=None,
        help="externally assessed quarterback continuity",
    )
    futures.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="emit output as structured JSON",
    )
    futures.set_defaults(func=_cmd_futures)

    live = sub.add_parser(
        "live",
        help="live in-game state tracking and micro-market simulation",
    )
    live.add_argument("--game", "--game-id", dest="game_id", default=None, help="stored game ID")
    live.add_argument("--home", default=None, help="home team ID")
    live.add_argument("--away", default=None, help="away team ID")
    live.add_argument("--quarter", type=int, default=None, help="current quarter (>= 1)")
    live.add_argument("--clock", default=None, help="game clock (MM:SS or seconds)")
    live.add_argument("--down", type=int, default=None, help="current down (1-4)")
    live.add_argument("--distance", type=int, default=None, help="yards to go (1-99)")
    live.add_argument(
        "--yardline", type=int, default=None, help="yards to opponent goal line (1-99)"
    )
    live.add_argument("--home-score", type=int, default=None, help="home score")
    live.add_argument("--away-score", type=int, default=None, help="away score")
    live.add_argument("--possession", default=None, help="possession team ID")
    live.add_argument(
        "--pregame-margin", type=float, default=0.0, help="pregame home expected margin"
    )
    live.add_argument(
        "--final",
        "--is-final",
        dest="final",
        action="store_true",
        default=False,
        help="mark game as completed / final",
    )
    live.add_argument(
        "--json", action="store_true", default=False, help="emit output as structured JSON"
    )
    live.set_defaults(func=_cmd_live)

    daily = sub.add_parser("daily", help="scheduled job: ingest available sources, rebuild market")
    daily.add_argument("--season", type=int, default=None, help="season year (default: current)")
    daily.add_argument(
        "--no-outlier",
        action="store_true",
        help="skip the Outlier leg (its token expires after 24h)",
    )
    daily.add_argument("--no-weather", action="store_true", help="skip the Open-Meteo leg")
    daily.add_argument(
        "--no-player-passing",
        action="store_true",
        help="skip incremental per-game passing-stat capture",
    )
    daily.add_argument(
        "--no-internal-ratings",
        action="store_true",
        help="skip fitting internal ridge team-strength ratings",
    )
    daily.add_argument(
        "--no-internal-elo", action="store_true", help="skip fitting internal Elo ratings"
    )
    daily.add_argument(
        "--with-props", action="store_true", help="ingest supported full-game team props"
    )
    daily.add_argument(
        "--with-scoring", action="store_true", help="evaluate opt-in team-prop candidates"
    )
    daily.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="do not auto-load this season's schedule when the store is empty",
    )
    daily.set_defaults(func=_cmd_daily)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CfbAnalyticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
