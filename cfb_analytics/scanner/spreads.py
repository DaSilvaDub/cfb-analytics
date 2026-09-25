"""Model 2 Spread Margin and Cover Probability Scanner.

Evaluates model-projected point margin against multi-book devigged consensus spread:
- Cover probability via standard normal CDF erf: Z = (projected_margin - market_spread) / 16.5
- Expected value (EV): EV = P(cover) * decimal_price - 1.0
- Spread edge: edge_points = projected_margin - market_spread
- Spread probability edge: edge_pct = model_prob - consensus_fair_prob
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

# Standard deviation for college football margin model (per config/settings.json)
SPREAD_SIGMA_BASE: float = 16.5


def norm_cdf(z: float) -> float:
    """Cumulative distribution function for standard normal distribution using stdlib math.erf.

    Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cover_probability(
    projected_margin: float,
    market_spread: float,
    sigma: float = SPREAD_SIGMA_BASE,
) -> float:
    """Calculate cover probability via normal CDF erf: Z = (projected_margin - market_spread) / 16.5.

    Args:
        projected_margin: Team's projected margin of victory (positive = win by X, negative = lose by X).
        market_spread: Market hurdle the team must cover (e.g. +7.5 if team is a 7.5-point favorite).
        sigma: Standard deviation of game point margin distribution (default 16.5).

    Returns:
        Probability in [0.001, 0.999].
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    z = (projected_margin - market_spread) / sigma
    p = norm_cdf(z)
    return round(max(0.001, min(0.999, p)), 4)


def calculate_spread_edge(projected_margin: float, market_spread: float) -> float:
    """Calculate quantitative spread edge in points (projected_margin - market_spread)."""
    return round(projected_margin - market_spread, 2)


def calculate_expected_value(win_prob: float, decimal_price: float) -> float:
    """Calculate expected value (EV) per unit staked: EV = win_prob * decimal_price - 1.0."""
    return round(win_prob * decimal_price - 1.0, 4)


def convert_spread_line_to_hurdle(line: float) -> float:
    """Convert a standard betting spread line to the cover hurdle.

    In standard sports betting notation:
    - Favorite is laying points (e.g. -7.5): to cover, team must win by > 7.5 points. Hurdle is +7.5.
    - Underdog is getting points (e.g. +7.5): to cover, team must not lose by >= 7.5 points. Hurdle is -7.5.
    Hurdle = -line.
    """
    return -line


def evaluate_spread_candidate(
    game_id: str,
    side: str,
    line: float,
    projected_margin: float,
    posted_price_american: int,
    consensus_fair_prob: float,
    *,
    market_spread: float | None = None,
    sigma: float = SPREAD_SIGMA_BASE,
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
    """Evaluate a game spread betting candidate and return an immutable MispricedOpportunity.

    Args:
        game_id: Unique game identifier.
        side: 'HOME' or 'AWAY'.
        line: Posted spread line (e.g. -7.5 for favorite, +7.5 for dog).
        projected_margin: Model projected margin for this side (positive = win, negative = lose).
        posted_price_american: American odds price (e.g. -110, +105).
        consensus_fair_prob: Vig-free consensus probability for this side.
        market_spread: Optional explicit cover hurdle. If None, derived as -line.
        sigma: Standard deviation for margin distribution (default 16.5).
        method_spread: Devig method spread (|prob_shin - prob_mult|).
        n_books: Number of sportsbooks pricing the market.
        historical_hit_rate: Historical win rate percentage (0 to 100).
        sample_size: Number of historical observations.
        confirming_insights_count: Number of independent confirming factors.
        open_line: Opening spread line for movement scoring.
        current_line: Current spread line for movement scoring.
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
    if side_clean not in ("HOME", "AWAY"):
        raise ValueError(f"side must be 'HOME' or 'AWAY', got {side!r}")

    # Determine cover hurdle: Z = (projected_margin - market_spread) / sigma
    hurdle = market_spread if market_spread is not None else convert_spread_line_to_hurdle(line)

    model_p = cover_probability(projected_margin, hurdle, sigma=sigma)
    points_edge = calculate_spread_edge(projected_margin, hurdle)
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
        market_type="SPREAD",
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
        market_type="SPREAD",
        market="SPREAD",
        side=side_clean,
        line=line,
        posted_price_american=posted_price_american,
        posted_price_decimal=round(decimal_price, 4),
        consensus_fair_prob=round(consensus_fair_prob, 4),
        consensus_fair_price_american=fair_american,
        model_projected_line=round(projected_margin, 2),
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
