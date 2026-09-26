"""Candidate Play Scoring (0-100 scale) and Minimum Qualification Gates.

Implements the authoritative 6-factor composite scoring engine and qualification gating
architecture for NCAA College Football wagering candidates.

Scoring Components (Max 100 points):
1. Historical Hit Rate (25 pts max)
2. Estimated Quantitative Edge (20 pts max)
3. Line Movement Confirmation (15 pts max)
4. Price / Odds Value (15 pts max)
5. Historical Sample Size (10 pts max)
6. Independent Confirming Insights (15 pts max)

Play Tiers:
- ELITE: >= 90.0
- STRONG: >= 80.0
- QUALIFIED: >= 70.0 (Actionable recommendation threshold)
- LEAN: >= 60.0
- PASS: < 60.0

Qualification Gates (Mandatory Veto Criteria):
- Measurable quantitative edge > 0.0
- Expected Value (EV) > 0.0
- Book coverage n_books >= 3
- Data quality score >= 75.0
- Sample size >= 5
- No disqualifying market/integrity flags
"""

from __future__ import annotations

from collections.abc import Sequence

from cfb_analytics.scanner.models import (
    SHADOW_MODE_DISCLAIMER,
    PlayScore,
    PlayTier,
    QualificationStatus,
)
from cfb_analytics.utils import implied_probability

DISQUALIFYING_FLAG_TOKENS: tuple[str, ...] = (
    "arb",
    "invalid",
    "placeholder",
    "single_book",
    "thin_market",
    "incomplete",
    "vetoed",
    "compromised",
    "disqualified",
)


def score_hit_rate(hit_rate: float | None) -> float:
    """Component 1: Historical Trend Win Rate (0 to 25 points).

    Brackets:
    - >= 80%: 25 pts
    - 75 - 79.9%: 22 pts
    - 70 - 74.9%: 19 pts
    - 65 - 69.9%: 15 pts
    - 60 - 64.9%: 10 pts
    - 55 - 59.9%: 5 pts
    - < 55%: 0 pts
    """
    if hit_rate is None:
        return 0.0
    rate = hit_rate / 100.0 if hit_rate > 1.0 else hit_rate
    if rate >= 0.80:
        return 25.0
    elif rate >= 0.75:
        return 22.0
    elif rate >= 0.70:
        return 19.0
    elif rate >= 0.65:
        return 15.0
    elif rate >= 0.60:
        return 10.0
    elif rate >= 0.55:
        return 5.0
    return 0.0


def score_edge(market_category: str, edge_value: float) -> float:
    """Component 2: Estimated Quantitative Edge (0 to 20 points).

    Evaluated across points, yards, or probability scales:
    - Points (Spread, Total, Team Points):
      >= 7.0: 20 pts, >= 4.5: 16 pts, >= 2.5: 12 pts, >= 1.0: 7 pts, > 0.0: 4 pts, <= 0.0: 0 pts
    - Yards (Rushing, Receiving, Offensive Yards):
      >= 35.0: 20 pts, >= 20.0: 16 pts, >= 10.0: 12 pts, >= 3.0: 7 pts, > 0.0: 4 pts, <= 0.0: 0 pts
    - Probability / Moneyline:
      >= 0.10: 20 pts, >= 0.06: 16 pts, >= 0.03: 12 pts, >= 0.01: 7 pts, > 0.0: 4 pts, <= 0.0: 0 pts
    """
    cat = market_category.strip().upper()
    is_yardage = "YARD" in cat or cat in ("RUSHING_YARDS", "RECEIVING_YARDS", "OFFENSIVE_YARDS")
    is_prob = "ML" in cat or "PROB" in cat or "MONEYLINE" in cat

    if is_yardage:
        if edge_value >= 35.0:
            return 20.0
        elif edge_value >= 20.0:
            return 16.0
        elif edge_value >= 10.0:
            return 12.0
        elif edge_value >= 3.0:
            return 7.0
        elif edge_value > 0.0:
            return 4.0
        return 0.0
    elif is_prob:
        if edge_value >= 0.10:
            return 20.0
        elif edge_value >= 0.06:
            return 16.0
        elif edge_value >= 0.03:
            return 12.0
        elif edge_value >= 0.01:
            return 7.0
        elif edge_value > 0.0:
            return 4.0
        return 0.0
    else:
        # Standard points scale (Spread, Total, Team Points)
        if edge_value >= 7.0:
            return 20.0
        elif edge_value >= 4.5:
            return 16.0
        elif edge_value >= 2.5:
            return 12.0
        elif edge_value >= 1.0:
            return 7.0
        elif edge_value > 0.0:
            return 4.0
        return 0.0


def score_line_movement(
    side: str,
    open_line: float | None,
    current_line: float | None,
    rlm_flag: bool = False,
) -> float:
    """Component 3: Line Movement Confirmation (0 to 15 points).

    Brackets:
    - RLM (Reverse Line Movement) Flag: 15 pts
    - Confirming move > 1.5 pts: 14 pts
    - Confirming move > 0 pts: 10 pts
    - Neutral (flat / None): 4 pts
    - Adverse move <= 1.5 pts: 2 pts
    - Adverse move > 1.5 pts: 1 pt
    """
    if rlm_flag:
        return 15.0
    if open_line is None or current_line is None:
        return 4.0

    diff = current_line - open_line
    side_upper = side.strip().upper()

    # Determine whether diff is confirming or adverse based on side
    # For OVER: line dropped (diff < 0) means the bar is lower -> confirming
    # For UNDER: line rose (diff > 0) means the bar is higher -> confirming
    # For HOME/AWAY spread:
    # If favorite line went from -7.0 to -6.0 (diff = +1.0), hurdle became easier -> confirming
    if side_upper == "OVER":
        if diff < -1.5:
            return 14.0
        elif diff < 0.0:
            return 10.0
        elif diff == 0.0:
            return 4.0
        elif diff <= 1.5:
            return 2.0
        return 1.0
    elif side_upper == "UNDER":
        if diff > 1.5:
            return 14.0
        elif diff > 0.0:
            return 10.0
        elif diff == 0.0:
            return 4.0
        elif diff >= -1.5:
            return 2.0
        return 1.0
    else:
        # Side spreads: if line moved towards pick (+1.5 easier)
        # diff > 1.5 confirming for positive movement
        if abs(diff) < 1e-9:
            return 4.0
        # When evaluating spread hurdle value for both favorites and underdogs,
        # diff = current_line - open_line > 0 always represents getting more points
        # or laying fewer points (easier cover hurdle).
        confirming_move = diff
        if confirming_move > 1.5:
            return 14.0
        elif confirming_move > 0.0:
            return 10.0
        elif confirming_move >= -1.5:
            return 2.0
        return 1.0


def score_price_value(
    model_prob: float | None,
    actionable_price_american: int | None,
) -> float:
    """Component 4: Price / Odds Value (0 to 15 points).

    Evaluates price edge = model_prob - best_implied_prob:
    - >= +5% (+0.05): 15 pts
    - >= +3% (+0.03): 12 pts
    - >= +1% (+0.01): 9 pts
    - >= -2% (-0.02): 6 pts
    - < -2%: 2 pts
    - Missing price or invalid prob: 0 pts
    """
    if actionable_price_american is None or model_prob is None or not (0.0 < model_prob < 1.0):
        return 0.0
    implied = implied_probability(actionable_price_american)
    if implied is None:
        return 0.0
    price_edge = model_prob - implied
    if price_edge >= 0.05:
        return 15.0
    elif price_edge >= 0.03:
        return 12.0
    elif price_edge >= 0.01:
        return 9.0
    elif price_edge >= -0.02:
        return 6.0
    return 2.0


def score_sample_size(sample_size: int) -> float:
    """Component 5: Historical Sample Size (0 to 10 points).

    - >= 50: 10 pts
    - 30 - 49: 8 pts
    - 20 - 29: 6 pts
    - 10 - 19: 4 pts
    - 5 - 9: 2 pts
    - < 5: 0 pts
    """
    if sample_size >= 50:
        return 10.0
    elif sample_size >= 30:
        return 8.0
    elif sample_size >= 20:
        return 6.0
    elif sample_size >= 10:
        return 4.0
    elif sample_size >= 5:
        return 2.0
    return 0.0


def score_confirming_insights(confirming_count: int) -> float:
    """Component 6: Independent Confirming Insights (0 to 15 points).

    - >= 4: 15 pts
    - 3: 12 pts
    - 2: 8 pts
    - 1: 4 pts
    - 0: 0 pts
    """
    if confirming_count >= 4:
        return 15.0
    elif confirming_count == 3:
        return 12.0
    elif confirming_count == 2:
        return 8.0
    elif confirming_count == 1:
        return 4.0
    return 0.0


def assign_play_tier(total_score: float, edge: float | None = None) -> PlayTier:
    """Map composite play score onto PlayTier enum.

    If edge <= 0.0, automatically downgrades to PASS per Minimum Qualification Gates.
    """
    if edge is not None and edge <= 0.0:
        return PlayTier.PASS
    if total_score >= 90.0:
        return PlayTier.ELITE
    elif total_score >= 80.0:
        return PlayTier.STRONG
    elif total_score >= 70.0:
        return PlayTier.QUALIFIED
    elif total_score >= 60.0:
        return PlayTier.LEAN
    return PlayTier.PASS


def evaluate_qualification_gates(
    edge: float,
    ev: float | None,
    n_books: int,
    total_score: float,
    *,
    data_quality_score: float = 100.0,
    sample_size: int = 20,
    has_price: bool = True,
    has_consensus_prob: bool = True,
    flags: Sequence[str] = (),
    is_vetoed: bool = False,
) -> tuple[QualificationStatus, tuple[str, ...]]:
    """Evaluate mandatory qualification gates and return status and rejection reasons.

    Gates:
    1. Edge > 0.0 (non_positive_edge)
    2. EV > 0.0 (non_positive_ev)
    3. n_books >= 3 (thin_market)
    4. Data quality score >= 75.0 (insufficient_model_data)
    5. Sample size >= 5 (sample_too_small)
    6. Price available (price_unavailable)
    7. Consensus prob available (missing_fair_probability)
    8. Disqualifying flags absent (disqualifying_market_flag)
    9. Favorite not vetoed by negative gate (favorite_vetoed)
    10. Play score >= 70.0 (score_below_threshold)
    """
    insufficient_reasons: list[str] = []
    failed_value_reasons: list[str] = []

    if n_books < 3:
        insufficient_reasons.append("thin_market")
    if not has_consensus_prob:
        insufficient_reasons.append("missing_fair_probability")
    if not has_price:
        insufficient_reasons.append("price_unavailable")
    if data_quality_score < 75.0:
        insufficient_reasons.append("insufficient_model_data")
    if sample_size < 5:
        insufficient_reasons.append("sample_too_small")

    # Check for disqualifying flags
    for flag in flags:
        flag_lower = flag.lower()
        if any(token in flag_lower for token in DISQUALIFYING_FLAG_TOKENS):
            insufficient_reasons.append(f"disqualifying_flag_{flag}")

    if is_vetoed:
        insufficient_reasons.append("favorite_vetoed_by_negative_gate")

    # Value checks
    if edge <= 0.0:
        failed_value_reasons.append("non_positive_edge")
    if ev is None or ev <= 0.0:
        failed_value_reasons.append("non_positive_ev")
    if total_score < 70.0:
        failed_value_reasons.append("score_below_qualified_threshold")

    reasons = tuple(insufficient_reasons + failed_value_reasons)

    if insufficient_reasons:
        if is_vetoed:
            return QualificationStatus.REVIEW, reasons
        return QualificationStatus.INSUFFICIENT_DATA, reasons
    elif failed_value_reasons:
        return QualificationStatus.PASS, reasons
    else:
        return QualificationStatus.QUALIFIED, reasons


def calculate_play_score(
    market_type: str,
    edge_metric: float,
    model_prob: float,
    posted_price_american: int | None,
    side: str,
    *,
    historical_hit_rate: float | None = None,
    sample_size: int = 20,
    confirming_insights_count: int = 0,
    open_line: float | None = None,
    current_line: float | None = None,
    rlm_flag: bool = False,
    ev: float | None = None,
    n_books: int = 3,
    consensus_fair_prob: float | None = None,
    data_quality_score: float = 100.0,
    flags: Sequence[str] = (),
    is_vetoed: bool = False,
) -> PlayScore:
    """Calculate the comprehensive 0-100 play score and evaluate qualification gates."""
    s_hit = score_hit_rate(historical_hit_rate)
    s_edge = score_edge(market_type, edge_metric)
    s_move = score_line_movement(side, open_line, current_line, rlm_flag=rlm_flag)
    s_price = score_price_value(model_prob, posted_price_american)
    s_sample = score_sample_size(sample_size)
    s_conf = score_confirming_insights(confirming_insights_count)

    total_score = round(min(100.0, max(0.0, s_hit + s_edge + s_move + s_price + s_sample + s_conf)), 1)

    qual_status, rejection_reasons = evaluate_qualification_gates(
        edge=edge_metric,
        ev=ev,
        n_books=n_books,
        total_score=total_score,
        data_quality_score=data_quality_score,
        sample_size=sample_size,
        has_price=posted_price_american is not None,
        has_consensus_prob=consensus_fair_prob is not None,
        flags=flags,
        is_vetoed=is_vetoed,
    )

    # Downgrade tier to PASS if not QUALIFIED
    if qual_status != QualificationStatus.QUALIFIED:
        tier = PlayTier.PASS if not is_vetoed else PlayTier.AVOID
    else:
        tier = assign_play_tier(total_score, edge=edge_metric)

    return PlayScore(
        hit_rate_score=s_hit,
        edge_score=s_edge,
        movement_score=s_move,
        price_value_score=s_price,
        sample_size_score=s_sample,
        confirming_score=s_conf,
        total_score=total_score,
        tier=tier,
        qualification_status=qual_status,
        rejection_reasons=rejection_reasons,
        shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
    )
