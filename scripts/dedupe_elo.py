"""Dry-run plan for duplicate and pre-stamped CFBD Elo rows.

Prints how many rows would remain after:

* deleting future-week rows (2026 weeks 11, 12, 13, and 15, which were
  stamped before those weeks were played)
* collapsing weeks 1-10 to one row per (team, season, week, source, scope),
  keeping the latest ``ingested_utc`` and then ``snapshot_id``

The default is a dry run. It opens the database read-only and deletes
nothing. ``--apply`` performs the deletes; do not pass it against the
published data branch from an unattended run.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

FUTURE_WEEKS = (11, 12, 13, 15)
DUPLICATE_WEEKS = tuple(range(1, 11))


def _connect(path: Path, *, write: bool) -> sqlite3.Connection:
    if write:
        conn = sqlite3.connect(path)
    else:
        uri = f"file:{path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _week_rows(conn: sqlite3.Connection, season: int, week: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """SELECT team_id, season, week, source, snapshot_scope, snapshot_id,
                      ingested_utc, rating
               FROM team_ratings
               WHERE season = ? AND week = ? AND source = 'elo_cfbd'
                 AND snapshot_scope = 'weekly'""",
            (season, week),
        )
    )


def _keep_latest(rows: list[sqlite3.Row]) -> set[str]:
    """Snapshot ids to keep: latest ingested_utc, then snapshot_id, per key."""
    grouped: dict[tuple, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        key = (
            row["team_id"],
            row["season"],
            row["week"],
            row["source"],
            row["snapshot_scope"],
        )
        grouped[key].append(row)
    keep: set[str] = set()
    for group in grouped.values():
        winner = max(group, key=lambda item: (str(item["ingested_utc"]), str(item["snapshot_id"])))
        keep.add(str(winner["snapshot_id"]))
    return keep


def plan(conn: sqlite3.Connection, season: int) -> dict[str, dict[int, dict[str, int | float]]]:
    future: dict[int, dict[str, int | float]] = {}
    for week in FUTURE_WEEKS:
        rows = _week_rows(conn, season, week)
        ratings = [float(row["rating"]) for row in rows if row["rating"] is not None]
        future[week] = {
            "before": len(rows),
            "after": 0,
            "would_delete": len(rows),
            "avg_rating": (sum(ratings) / len(ratings)) if ratings else 0.0,
        }
    duplicates: dict[int, dict[str, int | float]] = {}
    for week in DUPLICATE_WEEKS:
        rows = _week_rows(conn, season, week)
        keep = _keep_latest(rows)
        duplicates[week] = {
            "before": len(rows),
            "after": len(keep),
            "would_delete": len(rows) - len(keep),
        }
    return {"future": future, "duplicates": duplicates}


def render(season: int, counts: dict[str, dict[int, dict[str, int | float]]]) -> str:
    lines = [
        "DRY RUN — deletes nothing",
        f"season {season} elo_cfbd weekly rows",
    ]
    future_before = future_delete = 0
    for week, stats in counts["future"].items():
        future_before += int(stats["before"])
        future_delete += int(stats["would_delete"])
        lines.append(
            f"future week {week}: before={stats['before']} after={stats['after']} "
            f"would_delete={stats['would_delete']} avg_rating={stats['avg_rating']:.2f}"
        )
    lines.append(
        f"future weeks {','.join(str(w) for w in FUTURE_WEEKS)}: "
        f"before={future_before} after=0 would_delete={future_delete}"
    )
    dup_before = dup_after = dup_delete = 0
    for week, stats in counts["duplicates"].items():
        dup_before += int(stats["before"])
        dup_after += int(stats["after"])
        dup_delete += int(stats["would_delete"])
        if int(stats["would_delete"]) or int(stats["before"]):
            lines.append(
                f"week {week} duplicates: before={stats['before']} after={stats['after']} "
                f"would_delete={stats['would_delete']} "
                f"(keep latest ingested_utc, then snapshot_id)"
            )
    lines.append(
        f"weeks 1-10: before={dup_before} after={dup_after} would_delete={dup_delete}"
    )
    return "\n".join(lines)


def apply_plan(conn: sqlite3.Connection, season: int) -> None:
    for week in FUTURE_WEEKS:
        conn.execute(
            """DELETE FROM team_ratings
               WHERE season = ? AND week = ? AND source = 'elo_cfbd'
                 AND snapshot_scope = 'weekly'""",
            (season, week),
        )
    for week in DUPLICATE_WEEKS:
        rows = _week_rows(conn, season, week)
        keep = _keep_latest(rows)
        losers = [row["snapshot_id"] for row in rows if str(row["snapshot_id"]) not in keep]
        conn.executemany(
            "DELETE FROM team_ratings WHERE snapshot_id = ?",
            [(snapshot_id,) for snapshot_id in losers],
        )
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="path to cfb.sqlite3")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the rows this script would remove (not used by the dry run)",
    )
    args = parser.parse_args(argv)
    conn = _connect(args.db, write=bool(args.apply))
    try:
        counts = plan(conn, args.season)
        print(render(args.season, counts))
        if args.apply:
            apply_plan(conn, args.season)
            print("APPLY committed the deletes above")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
