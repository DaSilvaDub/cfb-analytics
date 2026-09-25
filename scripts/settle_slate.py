#!/usr/bin/env python3
"""Settle pre-game market consensus and model projections against actual final scores.

Usage:
    python scripts/settle_slate.py --date 2026-09-12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfb_analytics import db, paths
from cfb_analytics.backtest.settle import settle_slate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Settle market consensus against actual game scores."
    )
    parser.add_argument("--date", required=True, help="Slate date in YYYY-MM-DD format")
    parser.add_argument(
        "--min-books",
        type=int,
        default=1,
        help="Minimum books required on a market consensus to include (default: 1)",
    )
    parser.add_argument(
        "--json", action="store_true", default=False, help="Emit output as structured JSON"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()
    if not paths.database_path().exists():
        print("Database not found. Run 'cfb-analytics init-db' first.", file=sys.stderr)
        return 1

    with db.open_db() as conn:
        scored_games = conn.execute(
            "SELECT COUNT(*) AS n FROM games WHERE football_date = ? AND completed = 1",
            (args.date,),
        ).fetchone()["n"]

        if scored_games == 0:
            print(
                f"No completed games found for {args.date}.\n"
                f"Run: python -m cfb_analytics.cli backfill-cfbd "
                f"--start-year {args.date[:4]} --end-year {args.date[:4]}",
                file=sys.stderr,
            )
            return 1

        settlement = settle_slate(conn, args.date, min_books=args.min_books)

    if args.json:
        print(json.dumps(settlement.as_dict(), indent=2))
    else:
        print(settlement.as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
