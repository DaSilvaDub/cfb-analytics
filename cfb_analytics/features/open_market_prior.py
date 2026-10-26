"""Leakage-safe opening-line market prior for early-season moneyline.

CFBD historical ``/lines`` carries ``spreadOpen`` but no moneyline open and no
capture clock. ``ingest/cfbd_lines_historical.py`` stamps open rows at
``kickoff - 7d`` and close rows at ``kickoff - 60s``. Close ML/spread must
**not** be used as a same-game prediction feature when scoring against close
(that would leak / tautologize the promote compare).

This module exposes only the **open** HOME spread (median across books at the
open stamp, with ``line_movement.open_line`` fallback) and converts it to
P(home) via the same ``Phi(M / sigma)`` map the ridge path uses
(``M = -spread``).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from statistics import median

from cfb_analytics.backtest.calibration import margin_to_prob

# Synthetic open stamp is kickoff-7d; tolerate small float noise on julianday.
OPEN_LOOKBACK_DAYS_LO = 6.0
OPEN_LOOKBACK_DAYS_HI = 8.0

# Weeks where ratings are weak / week-1 is blacked out at min_games=30.
OPEN_MARKET_PRIOR_MAX_WEEK = 2
# Spike 2026-09-26: weight 1.0 (replace ens with open-implied) won W1–2 on
# 2023–2025 WF; blend 0.75 was close. See
# docs/scorecards/moneyline_scorecard_2023_2025_market_prior_early.md.
OPEN_MARKET_PRIOR_WEIGHT = 1.0

# Soft open blend for weeks after the full-replace window (typically 3–4).
# Weight must stay < 1.0 (full replace through week 4 was deferred). Spike
# 2026-09-26: w=0.75 won W3–4 / full gap on 2023–2025 WF without moving
# weeks 5+. See docs/scorecards/moneyline_scorecard_2023_2025_open_blend_w34.md.
OPEN_MARKET_BLEND_MAX_WEEK = 4
OPEN_MARKET_BLEND_WEIGHT = 0.75


def open_prior_weight_for_week(
    week: int,
    *,
    prior_max_week: int = OPEN_MARKET_PRIOR_MAX_WEEK,
    prior_weight: float = OPEN_MARKET_PRIOR_WEIGHT,
    blend_max_week: int = OPEN_MARKET_BLEND_MAX_WEEK,
    blend_weight: float = OPEN_MARKET_BLEND_WEIGHT,
) -> float:
    """Open-prior blend weight by week: full replace W≤prior, soft W≤blend.

    Weeks 1–2 keep production replace (``prior_weight``, typically 1.0).
    Weeks ``prior_max_week < week ≤ blend_max_week`` use ``blend_weight``
    (``w < 1`` soft blend). Later weeks return 0.
    """
    w = int(week)
    if w <= int(prior_max_week):
        return float(prior_weight)
    if w <= int(blend_max_week):
        return float(blend_weight)
    return 0.0


def load_open_home_spreads(
    conn: sqlite3.Connection,
    game_ids: list[str],
) -> dict[str, float]:
    """Median HOME open spread per game (cfbd_historical open stamp).

    Prefers ``odds_snapshots`` rows with captured ≈ kickoff-7d. Falls back to
    ``line_movement.open_line`` (CFBD ``spreadOpen``) when the open stamp is
    missing. Never reads close-stamp spreads.
    """
    if not game_ids:
        return {}
    out: dict[str, float] = {}
    for i in range(0, len(game_ids), 400):
        chunk = game_ids[i : i + 400]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT o.game_id, o.line
            FROM odds_snapshots o
            JOIN games g ON g.game_id = o.game_id
            WHERE o.game_id IN ({placeholders})
              AND o.source = 'cfbd_historical'
              AND o.market = 'SPREAD' AND o.side = 'HOME'
              AND o.line IS NOT NULL
              AND (julianday(g.kickoff_utc) - julianday(o.captured_utc))
                    BETWEEN ? AND ?
            """,
            (*chunk, OPEN_LOOKBACK_DAYS_LO, OPEN_LOOKBACK_DAYS_HI),
        ).fetchall()
        by_game: dict[str, list[float]] = defaultdict(list)
        for gid, line in rows:
            by_game[str(gid)].append(float(line))
        for gid, lines in by_game.items():
            out[gid] = float(median(lines))

        missing = [g for g in chunk if g not in out]
        if not missing:
            continue
        ph2 = ",".join("?" * len(missing))
        lm = conn.execute(
            f"""
            SELECT game_id, open_line FROM line_movement
            WHERE market = 'SPREAD' AND side = 'HOME'
              AND open_line IS NOT NULL
              AND game_id IN ({ph2})
            """,
            missing,
        ).fetchall()
        by_lm: dict[str, list[float]] = defaultdict(list)
        for gid, line in lm:
            by_lm[str(gid)].append(float(line))
        for gid, lines in by_lm.items():
            out.setdefault(gid, float(median(lines)))
    return out


def open_spread_to_home_prob(spread: float, sigma: float) -> float:
    """HOME spread → P(home wins). Home -14 => expected margin +14."""
    return margin_to_prob(-float(spread), float(sigma))


def open_home_probs(
    conn: sqlite3.Connection,
    game_ids: list[str],
    sigma: float,
) -> dict[str, float]:
    """Convenience: open HOME spreads converted to vig-free-ish P(home)."""
    spreads = load_open_home_spreads(conn, game_ids)
    return {
        gid: open_spread_to_home_prob(spread, sigma) for gid, spread in spreads.items()
    }


__all__ = [
    "OPEN_LOOKBACK_DAYS_HI",
    "OPEN_LOOKBACK_DAYS_LO",
    "OPEN_MARKET_BLEND_MAX_WEEK",
    "OPEN_MARKET_BLEND_WEIGHT",
    "OPEN_MARKET_PRIOR_MAX_WEEK",
    "OPEN_MARKET_PRIOR_WEIGHT",
    "load_open_home_spreads",
    "open_home_probs",
    "open_prior_weight_for_week",
    "open_spread_to_home_prob",
]
