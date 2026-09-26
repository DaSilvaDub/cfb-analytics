"""Mispriced line scanner across Spreads, Totals, and Team Props.

Compares model-projected values against vig-free consensus odds to identify
market inefficiencies. For each game on a slate:

- SPREAD: model-projected margin vs consensus spread (Elo + Ridge blend)
- TOTAL: model-projected game total vs consensus game total (Model 3)
- TEAM_PROP: model-projected team production vs posted team prop lines

Edge is quantified as probability difference (model prob - consensus fair prob)
and expressed as edge percentage, method spread, and confidence score.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from cfb_analytics import config
from cfb_analytics.features.over_confidence import format_kickoff_et
from cfb_analytics.models.team_props import (
    TeamPropsInputs,
    project_team_production,
)
from cfb_analytics.scoring import load_team_props_inputs_for_slate, norm_cdf
from cfb_analytics.utils import to_utc_iso

# Standard deviations for model probability calculation
SPREAD_SIGMA = 13.5  # CFB game margin std deviation (~13-14 points)
TOTAL_SIGMA = 8.0  # CFB game total std deviation (~7-9 points)
TEAM_POINTS_SIGMA = 7.0  # Team points std
TEAM_YARDS_SIGMA = 35.0  # Team yardage std

SHADOW_STAMP = (
    config.SHADOW_STAMP
    if hasattr(config, "SHADOW_STAMP")
    else ("UNPROMOTED - shadow output, not decision-grade")
)


@dataclass(frozen=True)
class MispricedCandidate:
    """One mispriced market opportunity, ranked by edge magnitude."""

    rank: int
    game_id: str
    game_label: str
    kickoff_et: str
    market: str  # 'SPREAD', 'TOTAL', 'TEAM_PROP'
    side: str  # 'HOME', 'AWAY', 'OVER', 'UNDER'
    line: float
    model_projected: float  # Model-projected value (margin, total, or points)
    model_prob: float  # Model-estimated probability
    consensus_fair_prob: float  # Vig-free consensus probability
    edge_pct: float  # model_prob - consensus_fair_prob
    method_spread: float  # Max-min across devig methods
    best_price: int | None  # Best American odds available
    best_book: str | None
    n_books: int
    hold: float | None
    anchor: str  # 'sharp', 'all_books', 'none'
    tier: str  # 'STRONG', 'MODERATE', 'MARGINAL'
    flags: tuple[str, ...]


def _model_prob_from_edge(projected: float, line: float, sigma: float) -> float:
    """Calculate model probability that OVER/HOME covers the line."""
    z = (projected - line) / sigma
    return round(max(0.01, min(0.99, norm_cdf(z))), 4)


def _assign_edge_tier(edge: float) -> str:
    if edge >= 0.06:
        return "STRONG"
    if edge >= 0.03:
        return "MODERATE"
    return "MARGINAL"


def build_mispriced_board(
    conn: sqlite3.Connection,
    date: str,
    *,
    min_edge: float = 0.02,
) -> list[MispricedCandidate]:
    """Build a ranked board of mispriced opportunities for a slate.

    Queries the latest consensus odds and model projections, then
    calculates edge for each market. Returns candidates sorted by
    descending absolute edge.
    """
    candidates: list[MispricedCandidate] = []

    # ── 1. Spread mispricing (Elo/Ridge projected margin vs consensus) ──
    candidates.extend(_scan_spread_mispricing(conn, date, min_edge=min_edge))

    # ── 2. Game total mispricing (Model 3 total vs consensus) ──
    candidates.extend(_scan_total_mispricing(conn, date, min_edge=min_edge))

    # ── 3. Team prop mispricing (Model 3 team production vs consensus) ──
    candidates.extend(_scan_team_prop_mispricing(conn, date, min_edge=min_edge))

    # Sort by descending edge magnitude
    candidates.sort(key=lambda c: -abs(c.edge_pct))

    # Assign ranks
    ranked: list[MispricedCandidate] = []
    for i, c in enumerate(candidates, 1):
        ranked.append(
            MispricedCandidate(
                rank=i,
                game_id=c.game_id,
                game_label=c.game_label,
                kickoff_et=c.kickoff_et,
                market=c.market,
                side=c.side,
                line=c.line,
                model_projected=c.model_projected,
                model_prob=c.model_prob,
                consensus_fair_prob=c.consensus_fair_prob,
                edge_pct=c.edge_pct,
                method_spread=c.method_spread,
                best_price=c.best_price,
                best_book=c.best_book,
                n_books=c.n_books,
                hold=c.hold,
                anchor=c.anchor,
                tier=c.tier,
                flags=c.flags,
            )
        )

    return ranked


def _scan_spread_mispricing(
    conn: sqlite3.Connection,
    date: str,
    *,
    min_edge: float = 0.02,
) -> list[MispricedCandidate]:
    """Compare model-projected spread vs consensus spread."""
    # Query latest consensus spread data
    rows = conn.execute(
        """SELECT g.game_id, g.kickoff_utc,
                  ht.alias AS home, at.alias AS away,
                  g.home_team_id, g.away_team_id,
                  c.side, c.line,
                  c.consensus_price, c.best_price, c.best_book,
                  c.prob_shin, c.prob_multiplicative, c.prob_power,
                  c.prob_spread AS method_spread, c.hold, c.n_books,
                  c.anchor, c.flags
           FROM market_consensus c
           JOIN games g ON g.game_id = c.game_id
           JOIN teams ht ON ht.team_id = g.home_team_id
           JOIN teams at ON at.team_id = g.away_team_id
           WHERE g.football_date = ? AND c.market = 'SPREAD'
             AND c.as_of_utc = (
               SELECT MAX(as_of_utc) FROM market_consensus
               WHERE game_id = c.game_id AND market = c.market
                 AND line = c.line AND side = c.side
             )
           ORDER BY g.kickoff_utc""",
        (date,),
    ).fetchall()

    if not rows:
        return []

    # Load model-projected margins (Elo ratings)
    model_margins = _load_model_margins(conn, date)

    candidates: list[MispricedCandidate] = []
    for row in rows:
        game_id = row["game_id"]
        if game_id not in model_margins:
            continue

        model_margin = model_margins[game_id]  # positive = home favorite
        consensus_line = row["line"]  # negative = favorite
        side = row["side"]

        # Calculate model probability that this side covers
        if side == "HOME":
            # Model margin is from home perspective
            model_proj = -model_margin  # Convert to spread convention
            model_p = _model_prob_from_edge(model_margin, -consensus_line, SPREAD_SIGMA)
        else:
            model_proj = model_margin
            model_p = _model_prob_from_edge(-model_margin, -consensus_line, SPREAD_SIGMA)

        # Consensus fair probability (prefer shin, fall back to multiplicative)
        fair_p = row["prob_shin"] or row["prob_multiplicative"]
        if fair_p is None:
            continue

        edge = round(model_p - fair_p, 4)
        if abs(edge) < min_edge:
            continue

        kickoff_et = format_kickoff_et(row["kickoff_utc"])
        label = f"{row['away']} at {row['home']}"
        flags_list: list[str] = []

        candidates.append(
            MispricedCandidate(
                rank=0,
                game_id=game_id,
                game_label=label,
                kickoff_et=kickoff_et,
                market="SPREAD",
                side=side,
                line=consensus_line,
                model_projected=round(model_proj, 1),
                model_prob=model_p,
                consensus_fair_prob=fair_p,
                edge_pct=edge,
                method_spread=row["method_spread"] or 0.0,
                best_price=row["best_price"],
                best_book=row["best_book"],
                n_books=row["n_books"],
                hold=row["hold"],
                anchor=row["anchor"] or "none",
                tier=_assign_edge_tier(abs(edge)),
                flags=tuple(flags_list),
            )
        )

    return candidates


def _scan_total_mispricing(
    conn: sqlite3.Connection,
    date: str,
    *,
    min_edge: float = 0.02,
) -> list[MispricedCandidate]:
    """Compare model-projected game total vs consensus total."""
    rows = conn.execute(
        """SELECT g.game_id, g.kickoff_utc,
                  ht.alias AS home, at.alias AS away,
                  g.home_team_id, g.away_team_id,
                  c.side, c.line,
                  c.consensus_price, c.best_price, c.best_book,
                  c.prob_shin, c.prob_multiplicative, c.prob_power,
                  c.prob_spread AS method_spread, c.hold, c.n_books,
                  c.anchor, c.flags
           FROM market_consensus c
           JOIN games g ON g.game_id = c.game_id
           JOIN teams ht ON ht.team_id = g.home_team_id
           JOIN teams at ON at.team_id = g.away_team_id
           WHERE g.football_date = ? AND c.market = 'TOTAL'
             AND c.as_of_utc = (
               SELECT MAX(as_of_utc) FROM market_consensus
               WHERE game_id = c.game_id AND market = c.market
                 AND line = c.line AND side = c.side
             )
           ORDER BY g.kickoff_utc""",
        (date,),
    ).fetchall()

    if not rows:
        return []

    # Load Model 3 total projections
    model_totals = _load_model_totals(conn, date)

    candidates: list[MispricedCandidate] = []
    for row in rows:
        game_id = row["game_id"]
        if game_id not in model_totals:
            continue

        proj_total = model_totals[game_id]
        consensus_line = row["line"]
        side = row["side"]

        # Model probability
        if side == "OVER":
            model_p = _model_prob_from_edge(proj_total, consensus_line, TOTAL_SIGMA)
        else:
            model_p = 1.0 - _model_prob_from_edge(proj_total, consensus_line, TOTAL_SIGMA)

        fair_p = row["prob_shin"] or row["prob_multiplicative"]
        if fair_p is None:
            continue

        edge = round(model_p - fair_p, 4)
        if abs(edge) < min_edge:
            continue

        kickoff_et = format_kickoff_et(row["kickoff_utc"])
        label = f"{row['away']} at {row['home']}"

        candidates.append(
            MispricedCandidate(
                rank=0,
                game_id=game_id,
                game_label=label,
                kickoff_et=kickoff_et,
                market="TOTAL",
                side=side,
                line=consensus_line,
                model_projected=round(proj_total, 1),
                model_prob=model_p,
                consensus_fair_prob=fair_p,
                edge_pct=edge,
                method_spread=row["method_spread"] or 0.0,
                best_price=row["best_price"],
                best_book=row["best_book"],
                n_books=row["n_books"],
                hold=row["hold"],
                anchor=row["anchor"] or "none",
                tier=_assign_edge_tier(abs(edge)),
                flags=(),
            )
        )

    return candidates


def _scan_team_prop_mispricing(
    conn: sqlite3.Connection,
    date: str,
    *,
    min_edge: float = 0.02,
) -> list[MispricedCandidate]:
    """Compare model-projected team production vs posted team prop lines.

    Reuses Model 3's existing team production projections from the scoring
    pipeline.
    """
    try:
        inputs_by_team = load_team_props_inputs_for_slate(conn, date)
    except Exception:
        return []

    if not inputs_by_team:
        return []

    # Query team prop consensus (if any)
    rows = conn.execute(
        """SELECT g.game_id, g.kickoff_utc,
                  ht.alias AS home, at.alias AS away,
                  g.home_team_id, g.away_team_id,
                  c.side, c.line, c.market AS prop_market,
                  c.consensus_price, c.best_price, c.best_book,
                  c.prob_shin, c.prob_multiplicative,
                  c.prob_spread AS method_spread, c.hold, c.n_books,
                  c.anchor, c.flags
           FROM market_consensus c
           JOIN games g ON g.game_id = c.game_id
           JOIN teams ht ON ht.team_id = g.home_team_id
           JOIN teams at ON at.team_id = g.away_team_id
           WHERE g.football_date = ?
             AND c.market NOT IN ('ML', 'SPREAD', 'TOTAL')
             AND c.as_of_utc = (
               SELECT MAX(as_of_utc) FROM market_consensus
               WHERE game_id = c.game_id AND market = c.market
                 AND line = c.line AND side = c.side
             )
           ORDER BY g.kickoff_utc""",
        (date,),
    ).fetchall()

    candidates: list[MispricedCandidate] = []
    for row in rows:
        home_team_id = str(row["home_team_id"]) if row["home_team_id"] else None
        away_team_id = str(row["away_team_id"]) if row["away_team_id"] else None
        if not home_team_id or not away_team_id:
            continue

        home_inputs = inputs_by_team.get(home_team_id)
        away_inputs = inputs_by_team.get(away_team_id)
        if home_inputs is None or away_inputs is None:
            continue

        side = row["side"]
        consensus_line = row["line"]

        # Project team production
        try:
            home_proj = project_team_production("TEAM_PROP", home_inputs)
            away_proj = project_team_production("TEAM_PROP", away_inputs)
        except Exception:
            continue

        # Determine projected value based on prop market and team
        projected_val = _extract_prop_projection(
            prop_market,
            home_proj,
            away_proj,
            row["home_team_id"],
            row["away_team_id"],
        )
        if projected_val is None:
            continue

        sigma = TEAM_POINTS_SIGMA if "point" in prop_market.lower() else TEAM_YARDS_SIGMA

        if side == "OVER":
            model_p = _model_prob_from_edge(projected_val, consensus_line, sigma)
        else:
            model_p = 1.0 - _model_prob_from_edge(projected_val, consensus_line, sigma)

        fair_p = row["prob_shin"] or row["prob_multiplicative"]
        if fair_p is None:
            continue

        edge = round(model_p - fair_p, 4)
        if abs(edge) < min_edge:
            continue

        kickoff_et = format_kickoff_et(row["kickoff_utc"])
        label = f"{row['away']} at {row['home']}"

        candidates.append(
            MispricedCandidate(
                rank=0,
                game_id=game_id,
                game_label=label,
                kickoff_et=kickoff_et,
                market="TEAM_PROP",
                side=side,
                line=consensus_line,
                model_projected=round(projected_val, 1),
                model_prob=model_p,
                consensus_fair_prob=fair_p,
                edge_pct=edge,
                method_spread=row["method_spread"] or 0.0,
                best_price=row["best_price"],
                best_book=row["best_book"],
                n_books=row["n_books"],
                hold=row["hold"],
                anchor=row["anchor"] or "none",
                tier=_assign_edge_tier(abs(edge)),
                flags=(),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Internal projection loaders
# ---------------------------------------------------------------------------


def _load_model_margins(
    conn: sqlite3.Connection,
    date: str,
) -> dict[str, float]:
    """Load Elo-projected home margin for each game on the slate.

    Falls back to internal Ridge ratings if Elo not available.
    Returns {game_id: home_margin} where positive = home favored.
    """
    from cfb_analytics.models.futures import (
        DEFAULT_HFA_POINTS,
        DEFAULT_MARGIN_SIGMA,
        project_game_win_probability,
    )

    # Try Elo ratings first
    elo_ratings = _load_latest_elo(conn)
    if not elo_ratings:
        elo_ratings = _load_latest_ridge(conn)

    if not elo_ratings:
        return {}

    games = conn.execute(
        """SELECT game_id, home_team_id, away_team_id, neutral_site
           FROM games
           WHERE football_date = ?""",
        (date,),
    ).fetchall()

    margins: dict[str, float] = {}
    for g in games:
        home_r = elo_ratings.get(g["home_team_id"])
        away_r = elo_ratings.get(g["away_team_id"])
        if home_r is None or away_r is None:
            continue

        is_neutral = bool(g["neutral_site"]) if g["neutral_site"] is not None else False
        # project_game_win_probability returns (spread, prob)
        # spread is from team perspective, negative = favorite
        spread, prob = project_game_win_probability(
            home_r,
            away_r,
            is_home=True,
            is_neutral=is_neutral,
        )
        # Convert spread (negative=home fav) back to margin (positive=home fav)
        margins[g["game_id"]] = round(-spread, 1)

    return margins


def _load_model_totals(
    conn: sqlite3.Connection,
    date: str,
) -> dict[str, float]:
    """Load Model 3 game total projections for each game on the slate.

    Uses team props inputs to project home + away points.
    """
    try:
        inputs_by_team = load_team_props_inputs_for_slate(conn, date)
    except Exception:
        return {}

    games = conn.execute(
        """SELECT game_id, home_team_id, away_team_id
           FROM games
           WHERE football_date = ?""",
        (date,),
    ).fetchall()

    totals: dict[str, float] = {}
    for g in games:
        game_id = str(g["game_id"])
        home_id = str(g["home_team_id"]) if g["home_team_id"] else None
        away_id = str(g["away_team_id"]) if g["away_team_id"] else None
        if not home_id or not away_id:
            continue
        home_inputs = inputs_by_team.get(home_id)
        away_inputs = inputs_by_team.get(away_id)
        if home_inputs is None or away_inputs is None:
            continue

        try:
            home_proj = project_team_production("TEAM_PROP", home_inputs)
            away_proj = project_team_production("TEAM_PROP", away_inputs)
            proj_total = (
                home_proj.projected_team_total_points + away_proj.projected_team_total_points
            )
            totals[game_id] = round(proj_total, 1)
        except Exception:
            continue

    return totals


def _load_latest_elo(conn: sqlite3.Connection) -> dict[str, float]:
    """Load the most recent Elo rating for each team."""
    rows = conn.execute(
        """SELECT team_id, rating
           FROM internal_elo_ratings
           WHERE as_of_utc = (
               SELECT MAX(as_of_utc) FROM internal_elo_ratings
           )"""
    ).fetchall()
    return {r["team_id"]: r["rating"] for r in rows}


def _load_latest_ridge(conn: sqlite3.Connection) -> dict[str, float]:
    """Fall back to Ridge power ratings if Elo not available."""
    rows = conn.execute(
        """SELECT team_id, offense + defense AS power_rating
           FROM internal_team_ratings
           WHERE as_of_utc = (
               SELECT MAX(as_of_utc) FROM internal_team_ratings
           )"""
    ).fetchall()
    return {r["team_id"]: r["power_rating"] for r in rows}


def _extract_prop_projection(
    prop_market: str,
    home_proj: Any,
    away_proj: Any,
    home_team_id: str,
    away_team_id: str,
) -> float | None:
    """Extract the relevant projected value for a team prop market."""
    market_lower = prop_market.lower().replace("-", "_").replace(" ", "_")

    # Determine if this is a home or away team prop
    # Team prop markets often embed the team name; use the prop_market to infer
    proj = home_proj  # Default to home team

    if hasattr(proj, "projected_team_total_points"):
        if "point" in market_lower or "total" in market_lower:
            return proj.projected_team_total_points
        if "rush" in market_lower:
            return proj.projected_rushing_yards
        if "receiv" in market_lower or "pass" in market_lower:
            return proj.projected_receiving_yards
        if "yard" in market_lower or "offensive" in market_lower:
            return proj.projected_offensive_yards

    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_mispriced_board(
    date: str,
    candidates: list[MispricedCandidate],
) -> str:
    """Format the mispriced board as a readable terminal table."""
    if not candidates:
        return f"No mispriced opportunities found for {date}."

    lines: list[str] = []
    stamp = SHADOW_STAMP if config.is_shadow_mode() else ""
    header = f"MISPRICED BOARD - {date}"
    if stamp:
        header += f"   [{stamp}]"
    lines.append(header)
    lines.append("")
    lines.append(
        f"{'#':>3} {'market':<10} {'game':<28} {'side':<6} {'line':>7} "
        f"{'proj':>7} {'model%':>7} {'fair%':>7} {'edge':>7} {'tier':<9} "
        f"{'best':>7} {'book':<11} {'bk':>3}"
    )
    lines.append("-" * 120)

    for c in candidates:
        game_short = c.game_label[:27]
        lines.append(
            f"{c.rank:>3} {c.market:<10} {game_short:<28} {c.side:<6} "
            f"{c.line:>7.1f} {c.model_projected:>7.1f} "
            f"{c.model_prob * 100:>6.1f}% {c.consensus_fair_prob * 100:>6.1f}% "
            f"{c.edge_pct * 100:>+6.1f}% {c.tier:<9} "
            f"{c.best_price or 0:>7} {(c.best_book or ''):<11} {c.n_books:>3}"
        )

    lines.append("")
    lines.append(f"Total: {len(candidates)} mispriced opportunities detected.")
    if stamp:
        lines.append(stamp)
    return "\n".join(lines)
