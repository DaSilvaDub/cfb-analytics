"""Print the commands that re-ingest AP poll weeks 5-9.

This is a dry run. It does not open the database, does not call CFBD, and
does not write ``team_polls``. Daniel runs the printed commands himself
after PR #20's canonical ``cfbd:`` name resolver is what
``backfill-rankings`` will execute. Those commands were not run here.
"""

from __future__ import annotations

import argparse

WEEKS = (5, 6, 7, 8, 9)


def commands(year: int) -> list[str]:
    return [
        f"python -m cfb_analytics.cli backfill-rankings --year {year} --week {week}"
        for week in WEEKS
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args(argv)
    print("DRY RUN — not executed. No CFBD request and no database write.")
    for command in commands(args.year):
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
