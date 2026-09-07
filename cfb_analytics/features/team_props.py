"""Build point-in-time, team-aware consensus for supported team props."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

from cfb_analytics.features.market import build_consensus


def build_team_prop_consensus(
    conn: sqlite3.Connection,
    game_ids: list[str],
    *,
    as_of_utc: str,
    min_books: int = 3,
) -> int:
    """Publish consensus from the latest eligible capture for each game.

    Only captures at or before both ``as_of_utc`` and kickoff are eligible.
    A group is published only with at least three books and a complete two-sided
    devig. Existing rows at the same as-of point are replaced atomically.
    """
    if not game_ids:
        return 0
    placeholders = ",".join("?" for _ in game_ids)
    rows = conn.execute(
        f"""WITH eligible AS (
                SELECT p.*,
                       MAX(p.captured_utc) OVER (PARTITION BY p.game_id) AS latest_capture
                FROM team_prop_odds_snapshots p
                JOIN games g ON g.game_id = p.game_id
                WHERE p.game_id IN ({placeholders})
                  AND p.captured_utc <= ?
                  AND p.captured_utc < g.kickoff_utc
            )
            SELECT * FROM eligible WHERE captured_utc = latest_capture""",
        [*game_ids, as_of_utc],
    ).fetchall()
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        grouped[
            (str(row["game_id"]), str(row["team_id"]), str(row["market"]), float(row["line"]))
        ].append(row)

    conn.execute(
        f"DELETE FROM team_prop_consensus WHERE game_id IN ({placeholders}) AND as_of_utc = ?",
        [*game_ids, as_of_utc],
    )
    written = 0
    for (game_id, team_id, market, line), market_rows in grouped.items():
        consensus = build_consensus(
            game_id,
            market,
            market_rows,
            as_of_utc=as_of_utc,
            min_books_for_consensus=min_books,
        )
        if (
            consensus is None
            or consensus.n_books < min_books
            or len(consensus.sides) != 2
            or any(side.vig_free_prob is None for side in consensus.sides)
        ):
            continue
        for side in consensus.sides:
            conn.execute(
                """INSERT INTO team_prop_consensus
                   (game_id, team_id, market, line, side, as_of_utc, n_books,
                    consensus_price, best_price, best_book, hold,
                    prob_multiplicative, prob_shin, prob_power, flags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    game_id,
                    team_id,
                    market,
                    line,
                    side.side,
                    as_of_utc,
                    consensus.n_books,
                    side.consensus_price,
                    side.best_price,
                    side.best_book,
                    consensus.hold,
                    side.probs.get("multiplicative"),
                    side.probs.get("shin"),
                    side.probs.get("power"),
                    json.dumps(consensus.flags),
                ),
            )
            written += 1
    return written
