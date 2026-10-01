"""The *current* market for a slate: one capture, one primary line per game.

``odds_snapshots`` and ``market_consensus`` are append-only, so every capture
and every alternate line a book has ever posted stays in the store. That is
right for the backtest and wrong for a report. Two failure modes this module
exists to prevent:

1. **Stale line groups.** Selecting "the latest row per (game, market, line,
   side)" resurrects a line that was posted at 9am and pulled at noon: its
   newest row is still the 9am one. Current means "present in the newest
   capture of that game's market", never "newest row of that line".
2. **Alternate-line clutter.** A capture carries a primary line plus a ladder
   of alternates. They are real prices, but a slate report wants the primary
   line: the one the most books hang, then the most balanced.

Readers (``board``, ``mispriced``, ``slate-report``) go through here so they
cannot each drift into their own definition of "current".
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

GAME_MARKETS: tuple[str, ...] = ("ML", "SPREAD", "TOTAL")

# Restricts a ``market_consensus c`` query to the newest capture of each
# (game, market). Shared as SQL so the per-row readers in ``mispriced`` and
# ``board`` stay single queries.
LATEST_CAPTURE_SQL = """c.as_of_utc = (
               SELECT MAX(m.as_of_utc) FROM market_consensus m
               WHERE m.game_id = c.game_id AND m.market = c.market
             )"""


def current_consensus(
    conn: sqlite3.Connection,
    slate_date: str,
    markets: Sequence[str] = GAME_MARKETS,
) -> list[dict[str, Any]]:
    """Every consensus row from the newest capture of each (game, market)."""
    if not markets:
        return []
    placeholders = ",".join("?" for _ in markets)
    rows = conn.execute(
        f"""SELECT g.game_id, g.kickoff_utc,
                   ht.alias AS home, at.alias AS away,
                   g.home_team_id, g.away_team_id,
                   c.market, c.line, c.side, c.as_of_utc,
                   c.consensus_price, c.best_price, c.best_book,
                   c.prob_shin, c.prob_multiplicative, c.prob_power,
                   c.prob_spread, c.hold, c.n_books, c.anchor, c.flags
            FROM market_consensus c
            JOIN games g ON g.game_id = c.game_id
            JOIN teams ht ON ht.team_id = g.home_team_id
            JOIN teams at ON at.team_id = g.away_team_id
            WHERE g.football_date = ? AND c.market IN ({placeholders})
              AND {LATEST_CAPTURE_SQL}
            ORDER BY g.kickoff_utc, g.game_id, c.market, c.line, c.side""",
        (slate_date, *markets),
    ).fetchall()
    return [dict(row) for row in rows]


def fair_prob(row: Mapping[str, Any]) -> float | None:
    """Shin first, multiplicative as the fallback - the house devig order."""
    value = row.get("prob_shin")
    if value is None:
        value = row.get("prob_multiplicative")
    return float(value) if value is not None else None


def _line_key(row: Mapping[str, Any]) -> float:
    # Spreads are posted per side (HOME -7 / AWAY +7); one market is one
    # absolute number. Totals and moneylines already share a line.
    line = row.get("line")
    return abs(float(line)) if line is not None else 0.0


def primary_lines(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the primary line of each (game, market).

    Primary = the widest book coverage (the thinner side's ``n_books``, so a
    one-sided alternate never wins), then the most balanced pricing (fair
    probability nearest 50%), then the lower hold. Moneylines have one line
    and pass through unchanged.
    """
    groups: dict[tuple[str, str], dict[float, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (str(row["game_id"]), str(row["market"]))
        if key not in groups:
            order.append(key)
        groups[key][_line_key(row)].append(dict(row))

    selected: list[dict[str, Any]] = []
    for key in order:
        by_line = groups[key]
        best = max(by_line.values(), key=_primary_score)
        selected.extend(sorted(best, key=lambda r: str(r.get("side") or "")))
    return selected


def _primary_score(group: list[dict[str, Any]]) -> tuple[int, int, float, float]:
    sides = {str(r.get("side") or "") for r in group}
    coverage = min(int(r.get("n_books") or 0) for r in group)
    probs = [p for p in (fair_prob(r) for r in group) if p is not None]
    balance = -min((abs(p - 0.5) for p in probs), default=0.5)
    hold = -float(group[0].get("hold") or 0.0)
    return (len(sides), coverage, balance, hold)


def current_primary_market(
    conn: sqlite3.Connection,
    slate_date: str,
    markets: Sequence[str] = GAME_MARKETS,
) -> list[dict[str, Any]]:
    """The slate's current market, reduced to one line per (game, market)."""
    return primary_lines(current_consensus(conn, slate_date, markets))


__all__ = [
    "GAME_MARKETS",
    "LATEST_CAPTURE_SQL",
    "current_consensus",
    "current_primary_market",
    "fair_prob",
    "primary_lines",
]
