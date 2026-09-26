"""Export CFB over-board picks into ncaafb_only.csv for Outlier Drive packs.

Preferred adapter: lives in cfb-analytics so it can call over-board internals
directly (same path as ``python -m cfb_analytics.cli over-board --date ... --json``).

Writes:
  - artifacts/exports/ncaafb_only.csv  (stable CFB-side path)
  - <outlier>/data/NCAAFB/exports/ncaafb_only.csv  (organize_today_run2 staging)

Always writes a header even when there are zero rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Any

CFB_ROOT = Path(__file__).resolve().parents[1]
if str(CFB_ROOT) not in sys.path:
    sys.path.insert(0, str(CFB_ROOT))

OUTLIER_ROOT = Path(r"C:\Users\dasil\Dev\GitHub\outlier")
CFB_OUT = CFB_ROOT / "artifacts" / "exports" / "ncaafb_only.csv"
OUTLIER_OUT = OUTLIER_ROOT / "data" / "NCAAFB" / "exports" / "ncaafb_only.csv"

FIELDNAMES = [
    "sport",
    "rank",
    "matchup",
    "selection",
    "side",
    "market",
    "family",
    "line",
    "projected",
    "over_prob",
    "confidence",
    "tier",
    "kickoff_et",
    "kickoff_utc",
    "game_id",
    "inclusion_reasons",
    "flags",
]


def _join_list(val: Any) -> str:
    if isinstance(val, (list, tuple)):
        return ";".join(str(x) for x in val if x is not None)
    if val is None:
        return ""
    return str(val)


def build_rows(slate_date: str, min_prob: float) -> list[dict[str, Any]]:
    from cfb_analytics import config, db, paths
    from cfb_analytics.features.over_confidence import (
        build_over_confidence_board,
        persist_over_board,
    )

    if not paths.database_path().exists():
        print("WARNING: No CFB database yet. Run: cfb-analytics init-db", file=sys.stderr)
        return []

    with db.open_db() as conn:
        picks = build_over_confidence_board(conn, slate_date, min_prob=min_prob)
        persist_over_board(conn, slate_date, picks)
        conn.commit()

    _ = config.SHADOW_STAMP if config.is_shadow_mode() else None

    rows: list[dict[str, Any]] = []
    for row in picks:
        rows.append(
            {
                "sport": "NCAAFB",
                "rank": row.rank,
                "matchup": row.game_label,
                "selection": row.pick,
                "side": row.side,
                "market": row.market,
                "family": row.family,
                "line": row.line if row.line is not None else "",
                "projected": row.projected if row.projected is not None else "",
                "over_prob": row.over_prob if row.over_prob is not None else "",
                "confidence": row.confidence if row.confidence is not None else "",
                "tier": row.tier,
                "kickoff_et": row.kickoff_et or "",
                "kickoff_utc": row.kickoff_utc or "",
                "game_id": row.game_id or "",
                "inclusion_reasons": _join_list(row.inclusion_reasons),
                "flags": _join_list(row.flags),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_ncaafb_only(
    slate_date: str | None = None,
    min_prob: float = 0.50,
    out_paths: list[Path] | None = None,
) -> tuple[list[Path], int]:
    day = slate_date or date.today().isoformat()
    try:
        rows = build_rows(day, min_prob=min_prob)
    except Exception as exc:  # noqa: BLE001 -- surface as empty CSV + warning
        print(f"WARNING: over-board failed for {day}: {exc}", file=sys.stderr)
        rows = []

    targets = out_paths or [CFB_OUT, OUTLIER_OUT]
    written: list[Path] = []
    for path in targets:
        write_csv(path, rows)
        written.append(path)
        print(f"{path} ({len(rows)} rows)")
    return written, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=None,
        help="Slate date YYYY-MM-DD (default: today local)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.50,
        help="Minimum OVER model probability (default: 0.50)",
    )
    parser.add_argument(
        "--out",
        action="append",
        default=None,
        help="Extra/override output CSV path (repeatable). Default writes CFB+Outlier paths.",
    )
    args = parser.parse_args()
    outs = [Path(p) for p in args.out] if args.out else None
    export_ncaafb_only(slate_date=args.date, min_prob=args.min_prob, out_paths=outs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())