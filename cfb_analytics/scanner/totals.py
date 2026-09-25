"""Model 3 Totals and Weather-Adjusted Game Scoring Scanner.

Compares tempo, pace, and weather-adjusted scoring projections against multi-book
consensus game totals:
- Over/Under probabilities via normal CDF: Z = (projected_total - market_total) / 10.75
- Expected value (EV): EV = P(win) * decimal_price - 1.0
- Total edge: edge_points = projected_total - market_total (for OVER)
- Total probability edge: edge_pct = model_prob - consensus_fair_prob
"""

from __future__ import annotations

import math

from cfb_analytics.scanner.models import (
    SHADOW_MODE_DISCLAIMER,
    MispricedOpportunity,
    PlayTier,
    QualificationStatus,
)
from cfb_analytics.scanner.scoring import calculate_play_score
from cfb_analytics.utils import american_to_decimal, decimal_to_american

# Standard deviation for college football total points distribution (per config/settings.json)
TOTAL_SIGMA_BASE: float = 10.75


def norm_cdf(z: float) -> float:
    """Cumulative distribution function for standard normal distribution using stdlib math.erf.

    Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def totals_cover_probability(
    projected_total: float,
    market_total: float,
    side: str = "OVER",
    sigma: float = TOTAL_SIGMA_BASE,
) -> float:
    """Calculate Over/Under win probability via normal CDF (sigma = 10.75).

    Args:
        projected_total: Fundamental projected combined score.
        market_total: Posted sportsbook game total line.
        side: 'OVER' or 'UNDER'.
        sigma: Standard deviation of game total distribution (default 10.75).

    Returns:
        Probability in [0.001, 0.999].
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    side_clean = side.strip().upper()
    if side_clean == "OVER":
        z = (projected_total - market_total) / sigma
    elif side_clean == "UNDER":
        z = (market_total - projected_total) / sigma
    else:
        raise ValueError(f"side must be 'OVER' or 'UNDER', got {side!r}")

    p = norm_cdf(z)
    return round(max(0.001, min(0.999, p)), 4)


def calculate_total_edge(projected_total: float, market_total: float, side: str = "OVER") -> float:
    """Calculate point edge for game total based on evaluated side."""
    side_clean = side.strip().upper()
    if side_clean == "OVER":
        return round(projected_total - market_total, 2)
    elif side_clean == "UNDER":
        return round(market_total - projected_total, 2)
    else:
        raise ValueError(f"side must be 'OVER' or 'UNDER', got {side!r}")


def calculate_expected_value(win_prob: float, decimal_price: float) -> float:
    """Calculate expected value (EV) per unit staked: EV = win_prob * decimal_price - 1.0."""
    return round(win_prob * decimal_price - 1.0, 4)


def project_game_total_adjusted(
    home_raw_points: float,
    away_raw_points: float,
    *,
    weather_multiplier: float = 1.0,
    pace_multiplier: float = 1.0,
) -> tuple[float, float, float]:
    """Calculate adjusted home points, away points, and combined game total.

    Applies weather attenuation (wind/precip/temp) and tempo adjustments.
    Returns (home_adj_points, away_adj_points, game_adj_total).
    """
    combined_mult = weather_multiplier * pace_multiplier
    home_adj = round(home_raw_points * combined_mult, 2)
    away_adj = round(away_raw_points * combined_mult, 2)
    game_total = round(home_adj + away_adj, 2)
    return home_adj, away_adj, game_total


def evaluate_total_candidate(
    game_id: str,
    side: str,
    line: float,
    projected_total: float,
    posted_price_american: int,
    consensus_fair_prob: float,
    *,
    sigma: float = TOTAL_SIGMA_BASE,
    method_spread: float = 0.0,
    n_books: int = 3,
    historical_hit_rate: float | None = None,
    sample_size: int = 20,
    confirming_insights_count: int = 0,
    open_line: float | None = None,
    current_line: float | None = None,
    rlm_flag: bool = False,
    best_book: str | None = None,
    game_label: str = "",
    kickoff_et: str = "",
    flags: tuple[str, ...] = (),
    data_quality_score: float = 100.0,
) -> MispricedOpportunity:
    """Evaluate a game total betting candidate and return an immutable MispricedOpportunity.

    Args:
        game_id: Unique game identifier.
        side: 'OVER' or 'UNDER'.
        line: Posted total line (e.g. 54.5).
        projected_total: Model-projected total score (e.g. 58.2).
        posted_price_american: American odds price (e.g. -110).
        consensus_fair_prob: Vig-free consensus probability for this side.
        sigma: Standard deviation for game total distribution (default 10.75).
        method_spread: Devig method spread (|prob_shin - prob_mult|).
        n_books: Number of sportsbooks pricing the market.
        historical_hit_rate: Historical win rate percentage (0 to 100).
        sample_size: Number of historical observations.
        confirming_insights_count: Number of independent confirming factors.
        open_line: Opening total line for movement scoring.
        current_line: Current total line for movement scoring.
        rlm_flag: True if reverse line movement is detected.
        best_book: Name of sportsbook with the best price.
        game_label: Descriptive game label (e.g. "Away at Home").
        kickoff_et: Kickoff time formatted in US Eastern.
        flags: Any market/injury/situational flags.
        data_quality_score: Data quality percentage (0 to 100).

    Returns:
        MispricedOpportunity instance.
    """
    side_clean = side.strip().upper()
    if side_clean not in ("OVER", "UNDER"):
        raise ValueError(f"side must be 'OVER' or 'UNDER', got {side!r}")

    model_p = totals_cover_probability(projected_total, line, side=side_clean, sigma=sigma)
    points_edge = calculate_total_edge(projected_total, line, side=side_clean)
    prob_edge = round(model_p - consensus_fair_prob, 4)

    decimal_price = american_to_decimal(posted_price_american)
    if decimal_price is None:
        decimal_price = 1.0

    ev = calculate_expected_value(model_p, decimal_price)

    # Calculate fair American price from consensus fair prob
    if 0.0 < consensus_fair_prob < 1.0:
        fair_dec = 1.0 / consensus_fair_prob
        fair_american = decimal_to_american(fair_dec) or 0
    else:
        fair_american = 0

    # Score candidate on 0-100 scale
    play_score_obj = calculate_play_score(
        market_type="TOTAL",
        edge_metric=points_edge,
        model_prob=model_p,
        posted_price_american=posted_price_american,
        side=side_clean,
        historical_hit_rate=historical_hit_rate,
        sample_size=sample_size,
        confirming_insights_count=confirming_insights_count,
        open_line=open_line,
        current_line=current_line,
        rlm_flag=rlm_flag,
        ev=ev,
        n_books=n_books,
        consensus_fair_prob=consensus_fair_prob,
        data_quality_score=data_quality_score,
        flags=flags,
    )

    tier_str = (
        play_score_obj.tier.value
        if isinstance(play_score_obj.tier, PlayTier)
        else str(play_score_obj.tier)
    )
    status_str = (
        play_score_obj.qualification_status.value
        if isinstance(play_score_obj.qualification_status, QualificationStatus)
        else str(play_score_obj.qualification_status)
    )

    return MispricedOpportunity(
        game_id=game_id,
        market_type="TOTAL",
        market="TOTAL",
        side=side_clean,
        line=line,
        posted_price_american=posted_price_american,
        posted_price_decimal=round(decimal_price, 4),
        consensus_fair_prob=round(consensus_fair_prob, 4),
        consensus_fair_price_american=fair_american,
        model_projected_line=round(projected_total, 2),
        model_prob=model_p,
        edge_pct=prob_edge,
        ev=ev,
        method_spread=round(method_spread, 4),
        n_books=n_books,
        play_score=play_score_obj.total_score,
        play_tier=tier_str,
        qual_status=status_str,
        best_book=best_book,
        game_label=game_label,
        kickoff_et=kickoff_et,
        flags=flags,
        rejection_reasons=play_score_obj.rejection_reasons,
        shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
    )
