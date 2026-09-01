"""Compute market consensus from stored odds and persist it.

Reads ``odds_snapshots`` through an ``AsOfReader`` so no post-kickoff price can
reach a feature, groups by (game, market, line), devigs each group, and writes
``market_consensus`` plus ``line_movement``.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cfb_analytics import config
from cfb_analytics.errors import LeakageError
from cfb_analytics.features import market
from cfb_analytics.features.asof import reader_for_game


@dataclass
class MarketBuildSummary:
    slate_date: str
    games: int = 0
    groups: int = 0
    consensus_rows: int = 0
    movement_rows: int = 0
    unpriced_groups: int = 0
    leaked_rows_dropped: int = 0
    anchors: dict[str, int] = field(default_factory=dict)
    flag_counts: dict[str, int] = field(default_factory=dict)

    def as_text(self) -> str:
        lines = [
            f"market build {self.slate_date}",
            f"  games            : {self.games}",
            f"  market groups    : {self.groups}",
            f"  consensus rows   : {self.consensus_rows}",
            f"  movement rows    : {self.movement_rows}",
            f"  unpriced groups  : {self.unpriced_groups}",
        ]
        if self.leaked_rows_dropped:
            lines.append(f"  post-kickoff rows dropped: {self.leaked_rows_dropped}")
        if self.anchors:
            lines.append(f"  anchors          : {dict(sorted(self.anchors.items()))}")
        if self.flag_counts:
            lines.append(f"  flags            : {dict(sorted(self.flag_counts.items()))}")
        return "\n".join(lines)


def _games_for_slate(conn: sqlite3.Connection, slate_date: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT game_id, kickoff_utc, season, football_date
           FROM games WHERE football_date = ? ORDER BY kickoff_utc""",
        (slate_date,),
    ).fetchall()
    return [dict(row) for row in rows]


def _odds_for_game(conn: sqlite3.Connection, game_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT game_id, market, side, line, book, price_american, captured_utc
           FROM odds_snapshots WHERE game_id = ? ORDER BY captured_utc""",
        (game_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def build_market_for_slate(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    markets: tuple[str, ...] = ("ML", "SPREAD", "TOTAL"),
) -> MarketBuildSummary:
    settings = config.settings()["market"]
    outlier = config.sources()["outlier"]
    sharp_books = tuple(outlier.get("sharp_books", ()))
    min_books = int(settings.get("min_books_for_consensus", 3))

    summary = MarketBuildSummary(slate_date=slate_date)
    games = _games_for_slate(conn, slate_date)
    summary.games = len(games)

    for game in games:
        reader = reader_for_game(game)
        rows = _odds_for_game(conn, game["game_id"])

        # The guard is the point: post-kickoff prices (closing lines) exist in
        # the store and must never reach a feature.
        admissible = reader.admissible(rows, what="odds_snapshots", as_of_field="captured_utc")
        summary.leaked_rows_dropped += len(rows) - len(admissible)
        if not admissible:
            continue

        latest_capture = max(r["captured_utc"] for r in admissible)

        grouped: dict[tuple[str, str, float | None], list[Mapping[str, Any]]] = defaultdict(list)
        for row in admissible:
            if row["market"] not in markets:
                continue
            grouped[market.group_key(row)].append(row)

        for (game_id, market_code, line), group in grouped.items():
            summary.groups += 1
            current = [r for r in group if r["captured_utc"] == latest_capture] or group

            consensus = market.build_consensus(
                game_id,
                market_code,
                current,
                as_of_utc=latest_capture,
                sharp_books=sharp_books,
                min_books_for_consensus=min_books,
            )
            if consensus is None or not consensus.sides:
                summary.unpriced_groups += 1
                if consensus is not None:
                    summary.anchors[consensus.anchor] = summary.anchors.get(consensus.anchor, 0) + 1
                continue

            summary.anchors[consensus.anchor] = summary.anchors.get(consensus.anchor, 0) + 1
            for flag in consensus.flags:
                summary.flag_counts[flag] = summary.flag_counts.get(flag, 0) + 1

            summary.consensus_rows += _write_consensus(conn, consensus, line)

        # Movement is computed ACROSS line groups, not within one.
        # Grouping by line holds the line constant by construction, so a
        # within-group comparison could only ever report price drift and would
        # score every spread move as 0.0.
        summary.movement_rows += _write_movement(
            conn, [r for r in admissible if r["market"] in markets], latest_capture
        )

    return summary


def _write_consensus(
    conn: sqlite3.Connection, consensus: market.MarketConsensus, line: float | None
) -> int:
    written = 0
    for quote in consensus.sides:
        spread = market.summarise_probability_spread(consensus, quote.side)
        conn.execute(
            """INSERT OR REPLACE INTO market_consensus
               (game_id, market, line, side, as_of_utc, n_books, consensus_price,
                best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
                prob_power, prob_spread, flags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                consensus.game_id, consensus.market,
                quote.line if quote.line is not None else (line or 0.0), quote.side,
                consensus.as_of_utc, quote.n_books, quote.consensus_price,
                quote.best_price, quote.best_book, consensus.hold, consensus.anchor,
                quote.probs.get("multiplicative"), quote.probs.get("shin"),
                quote.probs.get("power"), spread,
                json.dumps(list(consensus.flags)),
            ),
        )
        written += 1
    return written


def _representative(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The line best describing a market at one capture: the most-booked one.

    Books post a primary line plus a ladder of alternates. The alternates are
    real markets but they are not "the spread", so the line with the widest
    book coverage is taken as the capture's representative.
    """
    if not rows:
        return None
    by_line: dict[float | None, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_line[row.get("line")].append(row)
    best_line = max(
        by_line.values(),
        key=lambda group: (len({str(r["book"]).upper() for r in group}), -abs(_line_of(group))),
    )
    return best_line[0]


def _line_of(rows: list[Mapping[str, Any]]) -> float:
    for row in rows:
        if row.get("line") is not None:
            return float(row["line"])
    return 0.0


def _write_movement(
    conn: sqlite3.Connection,
    rows: list[Mapping[str, Any]],
    as_of_utc: str,
) -> int:
    """Open-to-current movement per (market, side), across line groups."""
    written = 0
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("side"):
            by_key[(str(row["market"]), str(row["side"]).upper())].append(row)

    for (market_code, side), side_rows in by_key.items():
        captures = sorted({r["captured_utc"] for r in side_rows})
        if len(captures) < 2:
            # One capture: nothing has moved yet. Writing a zero would look
            # like a measured flat line rather than an absence of data.
            continue
        opening = _representative([r for r in side_rows if r["captured_utc"] == captures[0]])
        current = _representative([r for r in side_rows if r["captured_utc"] == captures[-1]])
        if opening is None or current is None:
            continue
        move = market.line_movement(opening, current)
        game_id = str(side_rows[0]["game_id"])
        conn.execute(
            """INSERT OR REPLACE INTO line_movement
               (game_id, market, side, as_of_utc, open_line, open_price,
                current_line, current_price, move_magnitude, move_direction,
                rlm_flag, rlm_basis)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id, market_code, side, as_of_utc,
                move.get("open_line"), move.get("open_price"),
                move.get("current_line"), move.get("current_price"),
                move.get("move_magnitude"), move.get("move_direction"),
                1 if move.get("rlm_flag") else 0, move.get("rlm_basis", "line_only"),
            ),
        )
        written += 1
    return written


__all__ = ["MarketBuildSummary", "build_market_for_slate", "LeakageError"]
