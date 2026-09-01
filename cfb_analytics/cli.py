"""Command-line entry point.

Only the commands backed by working code are registered. Phases 2-9 add
``features``, ``train``, ``backtest``, ``slate``, ``parlay`` and ``settle``;
they are deliberately absent rather than present-and-stubbed, so ``--help``
never advertises something that does not run.
"""

from __future__ import annotations

import argparse
import sys

from cfb_analytics import config, db, paths
from cfb_analytics.errors import CfbAnalyticsError


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
    print(f"  database      : {paths.database_path()} "
          f"({'present' if paths.database_path().exists() else 'not created'})")
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
    print("    espn      : not implemented (Phase 2; needed for starting-QB confirmation)")
    print("    weather   : not implemented (Phase 2)")
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
            limit=args.limit,
        )
    print(summary.as_text())
    if summary.events_seen == 0:
        print("\nNo events on that date. Outlier carries a forward schedule only; "
              "use `cfb-analytics schedule` to list available dates.")
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
        for table in ("games", "teams", "odds_snapshots", "availability", "runs"):
            count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            print(f"  {table:<16} {count:>8}")
        row = conn.execute(
            "SELECT command, started_utc, status, rows_written FROM runs "
            "ORDER BY started_utc DESC LIMIT 1"
        ).fetchone()
        if row:
            print(f"\n  last run: {row['command']} at {row['started_utc']} "
                  f"-> {row['status']} ({row['rows_written']} rows)")
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

        print(f"{'slate':<12} {'captured (UTC)':<21} {'d-to-kick':>9} {'games':>5} "
              f"{'books':>5} {'prices':>7} {'px/game':>8}  sharp")
        any_sharp = False
        for row in rows:
            present = sorted(sharp & set((row["book_list"] or "").split(",")))
            any_sharp = any_sharp or bool(present)
            lead = (
                datetime.fromisoformat(row["first_kick"])
                - datetime.fromisoformat(row["captured"])
            ).days
            print(f"{row['slate']:<12} {row['captured'][:19]:<21} {lead:>9} "
                  f"{row['games']:>5} {row['books']:>5} {row['prices']:>7} "
                  f"{row['prices'] // max(row['games'], 1):>8}  "
                  f"{', '.join(present) if present else 'NONE'}")

        all_books = conn.execute(
            "SELECT DISTINCT book FROM odds_snapshots ORDER BY book").fetchall()
        print(f"\n  books ever seen ({len(all_books)}): "
              f"{', '.join(r['book'] for r in all_books)}")
        print(f"  sharp set tracked: {', '.join(sorted(sharp))}")
        if not any_sharp:
            print("\n  No sharp book has appeared in ANY capture. The sharp-anchor devig is "
                  "not currently supported by this feed:\n"
                  "  consensus must fall back to all books, rows carry `no_sharp_anchor`, "
                  "and CLV must be measured\n  against best-available price rather than a "
                  "Pinnacle close.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfb-analytics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create or migrate the SQLite store").set_defaults(
        func=_cmd_init_db)
    sub.add_parser("doctor", help="report source and credential readiness").set_defaults(
        func=_cmd_doctor)
    sub.add_parser("schedule", help="list available slate dates from Outlier").set_defaults(
        func=_cmd_schedule)
    sub.add_parser("status", help="row counts and last run").set_defaults(func=_cmd_status)
    sub.add_parser(
        "coverage", help="book coverage per capture, and whether sharp books appear"
    ).set_defaults(func=_cmd_coverage)

    ingest = sub.add_parser("ingest", help="ingest one slate from Outlier")
    ingest.add_argument("--date", required=True, help="slate date, YYYY-MM-DD")
    ingest.add_argument("--no-odds", action="store_true", help="skip gameline odds")
    ingest.add_argument("--no-injuries", action="store_true", help="skip the injury feed")
    ingest.add_argument("--limit", type=int, default=None, help="cap events (for smoke tests)")
    ingest.set_defaults(func=_cmd_ingest)

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
