"""Comprehensive unit tests for Mispriced Line Scanner (Milestone 2).

Verifies:
1. Immutability and structural integrity of frozen models and enums.
2. Model 2 spread cover probability via normal CDF erf (sigma=16.5) and EV calculation.
3. Model 3 game totals cover probability via normal CDF (sigma=10.75) and weather/tempo adjustments.
4. Model 3 team props, probability calibration capped at 84%, strict player prop prohibition,
   and blowout sit-QB filter.
5. Candidate Play Scoring (0-100 scale) across all 6 analytical components and play tiers.
6. Minimum qualification gates (EV > 0, edge > 0, n_books >= 3, data quality, flags).
7. Multi-method devigging (Shin & Multiplicative) and method spread calculation.
8. MispricedScanner orchestration, ReasoningCard interface, and slate evaluation.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
import pytest

from cfb_analytics.errors import DevigError, SchemaError
from cfb_analytics.reasoning.models import ReasoningCard
from cfb_analytics.scanner import (
    BLOWOUT_GAME_YARDS,
    DOG_REC_BLOWOUT,
    PROB_CAP,
    SHADOW_MODE_DISCLAIMER,
    SIT_QB_SPREAD,
    SPREAD_SIGMA_BASE,
    TOTAL_SIGMA_BASE,
    MispricedOpportunity,
    MispricedScanner,
    PlayScore,
    PlayTier,
    PlayerPropProhibitedError,
    QualificationStatus,
    assign_play_tier,
    calculate_play_score,
    calculate_prop_edge,
    calculate_spread_edge,
    calculate_total_edge,
    calibrated_prop_prob,
    convert_spread_line_to_hurdle,
    cover_probability,
    evaluate_qualification_gates,
    evaluate_spread_candidate,
    evaluate_team_prop_candidate,
    evaluate_total_candidate,
    is_sit_qb_candidate,
    project_game_total_adjusted,
    score_confirming_insights,
    score_edge,
    score_hit_rate,
    score_line_movement,
    score_price_value,
    score_sample_size,
    totals_cover_probability,
    validate_team_prop,
)


# ---------------------------------------------------------------------------
# 1. Model Immutability, Enums, and Contract Invariants
# ---------------------------------------------------------------------------


def test_mispriced_opportunity_immutability() -> None:
    """MispricedOpportunity is frozen and cannot be mutated."""
    opp = MispricedOpportunity(
        game_id="game_1",
        market_type="SPREAD",
        market="SPREAD",
        side="HOME",
        line=-7.5,
        posted_price_american=-110,
        posted_price_decimal=1.9091,
        consensus_fair_prob=0.5238,
        consensus_fair_price_american=-110,
        model_projected_line=-10.0,
        model_prob=0.5602,
        edge_pct=0.0364,
        ev=0.0695,
        method_spread=0.0082,
        n_books=5,
        play_score=78.5,
        play_tier=PlayTier.STRONG.value,
        qual_status=QualificationStatus.QUALIFIED.value,
    )

    with pytest.raises(FrozenInstanceError):
        opp.line = -8.0  # type: ignore[misc]

    assert opp.shadow_mode_disclaimer == SHADOW_MODE_DISCLAIMER
    assert opp.shadow_mode_disclaimer == "UNPROMOTED - shadow output, not decision-grade"
    assert opp.is_actionable is True


def test_play_score_immutability_and_properties() -> None:
    """PlayScore is frozen and exposes is_actionable and is_qualified properties."""
    ps = PlayScore(
        hit_rate_score=22.0,
        edge_score=16.0,
        movement_score=10.0,
        price_value_score=12.0,
        sample_size_score=8.0,
        confirming_score=8.0,
        total_score=76.0,
        tier=PlayTier.QUALIFIED,
        qualification_status=QualificationStatus.QUALIFIED,
    )

    with pytest.raises(FrozenInstanceError):
        ps.total_score = 99.0  # type: ignore[misc]

    assert ps.is_qualified is True
    assert ps.is_actionable is True


def test_play_tier_and_qualification_enums() -> None:
    """Verify tier and status enum values."""
    assert PlayTier.ELITE.value == "ELITE"
    assert PlayTier.STRONG.value == "STRONG"
    assert PlayTier.QUALIFIED.value == "QUALIFIED"
    assert PlayTier.LEAN.value == "LEAN"
    assert PlayTier.PASS.value == "PASS"
    assert PlayTier.AVOID.value == "AVOID"

    assert QualificationStatus.QUALIFIED.value == "QUALIFIED"
    assert QualificationStatus.PASS.value == "PASS"
    assert QualificationStatus.INSUFFICIENT_DATA.value == "INSUFFICIENT_DATA"
    assert QualificationStatus.STALE.value == "STALE"
    assert QualificationStatus.REVIEW.value == "REVIEW"


# ---------------------------------------------------------------------------
# 2. Spreads Scanning, Normal CDF Erf, and Expected Value
# ---------------------------------------------------------------------------


def test_spread_cover_probability_erf_formula() -> None:
    """Cover probability follows Z = (projected_margin - market_spread) / 16.5."""
    # When projected margin exactly equals market spread hurdle, Z = 0 -> P = 0.50
    p_even = cover_probability(projected_margin=7.0, market_spread=7.0, sigma=16.5)
    assert math.isclose(p_even, 0.50, abs_tol=1e-3)

    # When projected margin exceeds market spread by exactly 1.0 sigma (16.5 pts)
    # Z = 1.0 -> Phi(1.0) = 0.5 * (1 + erf(1 / sqrt(2))) ≈ 0.8413
    p_plus_1sigma = cover_probability(projected_margin=23.5, market_spread=7.0, sigma=16.5)
    expected_prob = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    assert math.isclose(p_plus_1sigma, round(expected_prob, 4), abs_tol=1e-3)

    # When projected margin is 1.0 sigma below market spread hurdle
    # Z = -1.0 -> Phi(-1.0) ≈ 0.1587
    p_minus_1sigma = cover_probability(projected_margin=7.0, market_spread=23.5, sigma=16.5)
    assert math.isclose(p_minus_1sigma, round(1.0 - expected_prob, 4), abs_tol=1e-3)


def test_spread_line_to_hurdle_conversion() -> None:
    """Spread line converts correctly to cover hurdle: hurdle = -line."""
    # Favorite laying 7.5 points (-7.5 line) must win by > 7.5 (hurdle +7.5)
    assert convert_spread_line_to_hurdle(-7.5) == 7.5
    # Underdog getting 3.5 points (+3.5 line) must not lose by >= 3.5 (hurdle -3.5)
    assert convert_spread_line_to_hurdle(3.5) == -3.5


def test_spread_edge_and_ev_calculations() -> None:
    """Spread edge and EV calculations follow quantitative formulations."""
    edge = calculate_spread_edge(projected_margin=10.0, market_spread=7.0)
    assert edge == 3.0

    # EV = win_prob * decimal_price - 1.0
    # At +100 odds (decimal 2.0) and 55% win prob: EV = 0.55 * 2.0 - 1.0 = +0.10
    from cfb_analytics.scanner.spreads import calculate_expected_value

    ev_positive = calculate_expected_value(win_prob=0.55, decimal_price=2.0)
    assert math.isclose(ev_positive, 0.10, abs_tol=1e-4)

    # At -110 odds (decimal 1.9091) and 50% win prob: EV = 0.50 * 1.9091 - 1.0 = -0.0455
    ev_negative = calculate_expected_value(win_prob=0.50, decimal_price=1.9091)
    assert math.isclose(ev_negative, -0.0455, abs_tol=1e-4)


def test_evaluate_spread_candidate() -> None:
    """evaluate_spread_candidate constructs valid MispricedOpportunity."""
    opp = evaluate_spread_candidate(
        game_id="cfbd_12345",
        side="HOME",
        line=-7.0,
        projected_margin=10.5,
        posted_price_american=-110,
        consensus_fair_prob=0.5238,
        method_spread=0.012,
        n_books=5,
        historical_hit_rate=75.0,
        sample_size=30,
        confirming_insights_count=3,
    )

    assert opp.game_id == "cfbd_12345"
    assert opp.market_type == "SPREAD"
    assert opp.side == "HOME"
    assert opp.line == -7.0
    # Hurdle is +7.0, margin is 10.5 -> edge is 3.5 points
    # Z = (10.5 - 7.0) / 16.5 = 3.5 / 16.5 ≈ 0.2121
    # P ≈ 0.5840
    assert opp.model_prob > 0.55
    assert opp.edge_pct > 0.0
    assert opp.ev > 0.0
    assert opp.qual_status == QualificationStatus.QUALIFIED.value
    assert opp.play_score >= 70.0


def test_evaluate_spread_invalid_side_raises() -> None:
    """evaluate_spread_candidate rejects invalid side strings."""
    with pytest.raises(ValueError, match="side must be 'HOME' or 'AWAY'"):
        evaluate_spread_candidate(
            game_id="g1",
            side="OVER",
            line=-7.0,
            projected_margin=10.0,
            posted_price_american=-110,
            consensus_fair_prob=0.50,
        )


# ---------------------------------------------------------------------------
# 3. Totals Scanning, Tempo/Weather Adjustments, and Normal CDF
# ---------------------------------------------------------------------------


def test_totals_cover_probability_sigma() -> None:
    """Total points cover probability uses normal CDF with baseline sigma = 10.75."""
    # Even total: projected == market -> P = 0.50
    p_even = totals_cover_probability(projected_total=52.0, market_total=52.0, side="OVER")
    assert math.isclose(p_even, 0.50, abs_tol=1e-3)

    # OVER side: projected 1.0 sigma above market (62.75 vs 52.0, diff = 10.75)
    p_over_1sigma = totals_cover_probability(
        projected_total=62.75,
        market_total=52.0,
        side="OVER",
        sigma=TOTAL_SIGMA_BASE,
    )
    expected_prob = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    assert math.isclose(p_over_1sigma, round(expected_prob, 4), abs_tol=1e-3)

    # UNDER side: market 1.0 sigma above projected -> UNDER wins with 1.0 sigma prob
    p_under_1sigma = totals_cover_probability(
        projected_total=52.0,
        market_total=62.75,
        side="UNDER",
        sigma=TOTAL_SIGMA_BASE,
    )
    assert math.isclose(p_under_1sigma, round(expected_prob, 4), abs_tol=1e-3)


def test_calculate_total_edge() -> None:
    """Total edge calculation is positive when projected is favorable to chosen side."""
    # OVER edge
    assert calculate_total_edge(projected_total=55.5, market_total=50.0, side="OVER") == 5.5
    # UNDER edge
    assert calculate_total_edge(projected_total=45.0, market_total=50.0, side="UNDER") == 5.0

    with pytest.raises(ValueError, match="side must be 'OVER' or 'UNDER'"):
        calculate_total_edge(50.0, 50.0, side="HOME")


def test_project_game_total_adjusted() -> None:
    """project_game_total_adjusted attenuates home/away points by weather and pace multipliers."""
    home_raw, away_raw = 30.0, 24.0
    # Weather multiplier 0.85 (e.g. heavy rain/wind), pace multiplier 0.95
    h_adj, a_adj, g_tot = project_game_total_adjusted(
        home_raw,
        away_raw,
        weather_multiplier=0.85,
        pace_multiplier=0.95,
    )
    combined = 0.85 * 0.95
    assert math.isclose(h_adj, round(30.0 * combined, 2))
    assert math.isclose(a_adj, round(24.0 * combined, 2))
    assert math.isclose(g_tot, round(h_adj + a_adj, 2))


def test_evaluate_total_candidate() -> None:
    """evaluate_total_candidate constructs valid total MispricedOpportunity."""
    opp = evaluate_total_candidate(
        game_id="cfbd_total_1",
        side="OVER",
        line=48.5,
        projected_total=56.0,
        posted_price_american=-105,
        consensus_fair_prob=0.5122,
        historical_hit_rate=78.0,
        sample_size=40,
        confirming_insights_count=4,
    )

    assert opp.market_type == "TOTAL"
    assert opp.side == "OVER"
    assert opp.line == 48.5
    assert opp.model_projected_line == 56.0
    assert opp.edge_pct > 0.10
    assert opp.ev > 0.20
    assert opp.qual_status == QualificationStatus.QUALIFIED.value
    assert opp.play_score >= 80.0


# ---------------------------------------------------------------------------
# 4. Team Props Scanning, Player Prop Prohibition, and Sit-QB Filter
# ---------------------------------------------------------------------------


def test_strictly_team_prop_validation() -> None:
    """Model 3 strictly requires TEAM_PROP and raises PlayerPropProhibitedError for player props."""
    # Valid canonical team props
    assert validate_team_prop("team_rushing_yards") == "team_rushing_yards"
    assert validate_team_prop("team_receiving_yards") == "team_receiving_yards"
    assert validate_team_prop("team_total_points") == "team_total_points"
    assert validate_team_prop("team_offensive_yards") == "team_offensive_yards"
    assert validate_team_prop("rushing_yards") == "team_rushing_yards"
    assert validate_team_prop("points") == "team_total_points"

    # Player props must raise PlayerPropProhibitedError
    player_props = [
        "player_rushing_yards",
        "passer_rating",
        "rusher_yards",
        "receiver_receptions",
        "anytime_touchdown",
        "first_td",
        "player_pass_yards",
    ]
    for pp in player_props:
        with pytest.raises(PlayerPropProhibitedError) as exc_info:
            validate_team_prop(pp)
        # Verify dual inheritance
        assert isinstance(exc_info.value, SchemaError)
        assert isinstance(exc_info.value, ValueError)

    # Invalid market_type
    with pytest.raises(PlayerPropProhibitedError) as exc_info:
        validate_team_prop("team_rushing_yards", market_type="PLAYER_PROP")
    assert isinstance(exc_info.value, SchemaError)
    assert isinstance(exc_info.value, ValueError)


def test_calibrated_prop_prob_hard_cap_84() -> None:
    """Calibrated prop probability is hard capped at 84% (PROB_CAP = 0.84)."""
    # Massive mismatch: projected 450 yards vs line of 120 yards
    # Z = (450 - 120) / 35 = 9.42 -> normal CDF erf would yield 0.9999+
    prob_uncapped = 0.5 * (1.0 + math.erf((450.0 - 120.0) / (35.0 * math.sqrt(2.0))))
    assert prob_uncapped > 0.999

    prob_capped = calibrated_prop_prob(
        canonical_market="team_rushing_yards",
        projected=450.0,
        line=120.0,
        side="OVER",
        posted=True,
    )
    assert prob_capped == PROB_CAP
    assert prob_capped == 0.84

    # UNDER side hard cap
    prob_under_capped = calibrated_prop_prob(
        canonical_market="team_receiving_yards",
        projected=50.0,
        line=350.0,
        side="UNDER",
        posted=True,
    )
    assert prob_under_capped == PROB_CAP
    assert prob_under_capped == 0.84


def test_blowout_sit_qb_filter() -> None:
    """Favorite receiving props dropped at spread <= -20.0; underdog at spread >= +28.0."""
    # Favorite receiving props at spread <= -20.0
    assert is_sit_qb_candidate("team_receiving_yards", spread=-20.0, is_favorite=True) is True
    assert is_sit_qb_candidate("team_receiving_yards", spread=-28.5, is_favorite=True) is True
    # At -19.5 (below sit-QB threshold), not dropped
    assert is_sit_qb_candidate("team_receiving_yards", spread=-19.5, is_favorite=True) is False

    # Underdog receiving props at spread >= +28.0
    assert is_sit_qb_candidate("team_receiving_yards", spread=28.0, is_favorite=False) is True
    assert is_sit_qb_candidate("team_receiving_yards", spread=35.0, is_favorite=False) is True
    # At +27.5, not dropped
    assert is_sit_qb_candidate("team_receiving_yards", spread=27.5, is_favorite=False) is False

    # Rushing props are NEVER dropped by sit-QB filter
    assert is_sit_qb_candidate("team_rushing_yards", spread=-25.0, is_favorite=True) is False
    assert is_sit_qb_candidate("team_total_points", spread=-25.0, is_favorite=True) is False


def test_evaluate_team_prop_candidate_sit_qb_drop() -> None:
    """evaluate_team_prop_candidate returns None when sit-QB filter drops favorite receiving."""
    # Favorite receiving prop at spread -21.0
    candidate = evaluate_team_prop_candidate(
        game_id="game_blowout",
        market="team_receiving_yards",
        side="OVER",
        line=285.5,
        projected_value=320.0,
        posted_price_american=-110,
        consensus_fair_prob=0.5238,
        spread=-21.0,
        is_favorite=True,
        drop_sit_qb=True,
    )
    assert candidate is None

    # Underdog receiving prop at spread +30.0
    candidate_dog = evaluate_team_prop_candidate(
        game_id="game_blowout",
        market="team_receiving_yards",
        side="OVER",
        line=150.5,
        projected_value=190.0,
        posted_price_american=-110,
        consensus_fair_prob=0.5238,
        spread=30.0,
        is_favorite=False,
        drop_sit_qb=True,
    )
    assert candidate_dog is None

    # Rushing prop at spread -21.0 is NOT dropped
    candidate_rush = evaluate_team_prop_candidate(
        game_id="game_blowout",
        market="team_rushing_yards",
        side="OVER",
        line=180.5,
        projected_value=220.0,
        posted_price_american=-110,
        consensus_fair_prob=0.5238,
        spread=-21.0,
        is_favorite=True,
        drop_sit_qb=True,
    )
    assert candidate_rush is not None
    assert candidate_rush.market == "team_rushing_yards"


# ---------------------------------------------------------------------------
# 5. Candidate Play Scoring (0-100 scale) and Tiers
# ---------------------------------------------------------------------------


def test_score_hit_rate_brackets() -> None:
    """Component 1: Historical hit rate correctly brackets 0 to 25 points."""
    assert score_hit_rate(85.0) == 25.0
    assert score_hit_rate(0.80) == 25.0
    assert score_hit_rate(77.5) == 22.0
    assert score_hit_rate(72.0) == 19.0
    assert score_hit_rate(67.0) == 15.0
    assert score_hit_rate(62.0) == 10.0
    assert score_hit_rate(57.0) == 5.0
    assert score_hit_rate(50.0) == 0.0
    assert score_hit_rate(None) == 0.0


def test_score_edge_brackets() -> None:
    """Component 2: Estimated quantitative edge correctly brackets points, yards, and prob."""
    # Points
    assert score_edge("SPREAD", 8.0) == 20.0
    assert score_edge("TOTAL", 5.0) == 16.0
    assert score_edge("POINTS", 3.0) == 12.0
    assert score_edge("SPREAD", 1.5) == 7.0
    assert score_edge("SPREAD", 0.5) == 4.0
    assert score_edge("SPREAD", 0.0) == 0.0
    assert score_edge("SPREAD", -2.0) == 0.0

    # Yards
    assert score_edge("RUSHING_YARDS", 40.0) == 20.0
    assert score_edge("RECEIVING_YARDS", 25.0) == 16.0
    assert score_edge("OFFENSIVE_YARDS", 12.0) == 12.0
    assert score_edge("RUSHING_YARDS", 5.0) == 7.0
    assert score_edge("RUSHING_YARDS", 1.0) == 4.0
    assert score_edge("RUSHING_YARDS", 0.0) == 0.0

    # Probability (ML)
    assert score_edge("ML", 0.12) == 20.0
    assert score_edge("MONEYLINE", 0.07) == 16.0
    assert score_edge("ML", 0.04) == 12.0
    assert score_edge("ML", 0.015) == 7.0
    assert score_edge("ML", 0.005) == 4.0
    assert score_edge("ML", 0.0) == 0.0


def test_score_line_movement_brackets() -> None:
    """Component 3: Line movement correctly brackets confirmation."""
    # RLM flag grants max points
    assert score_line_movement("OVER", open_line=52.0, current_line=54.0, rlm_flag=True) == 15.0

    # OVER side: line dropped by > 1.5 pts (54.0 -> 52.0, diff = -2.0)
    assert score_line_movement("OVER", open_line=54.0, current_line=52.0) == 14.0
    # Line dropped by 1.0 pt (54.0 -> 53.0)
    assert score_line_movement("OVER", open_line=54.0, current_line=53.0) == 10.0
    # Flat line
    assert score_line_movement("OVER", open_line=54.0, current_line=54.0) == 4.0
    # Adverse move <= 1.5
    assert score_line_movement("OVER", open_line=54.0, current_line=55.0) == 2.0
    # Adverse move > 1.5
    assert score_line_movement("OVER", open_line=54.0, current_line=56.5) == 1.0

    # Missing line
    assert score_line_movement("OVER", open_line=None, current_line=54.0) == 4.0


def test_score_price_value_brackets() -> None:
    """Component 4: Price value evaluates model prob vs best implied prob."""
    # At +100 (implied prob 0.50):
    # model prob 0.56 -> edge +0.06 >= 0.05 -> 15 pts
    assert score_price_value(model_prob=0.56, actionable_price_american=100) == 15.0
    # model prob 0.54 -> edge +0.04 >= 0.03 -> 12 pts
    assert score_price_value(model_prob=0.54, actionable_price_american=100) == 12.0
    # model prob 0.52 -> edge +0.02 >= 0.01 -> 9 pts
    assert score_price_value(model_prob=0.52, actionable_price_american=100) == 9.0
    # model prob 0.49 -> edge -0.01 >= -0.02 -> 6 pts
    assert score_price_value(model_prob=0.49, actionable_price_american=100) == 6.0
    # model prob 0.45 -> edge -0.05 < -0.02 -> 2 pts
    assert score_price_value(model_prob=0.45, actionable_price_american=100) == 2.0


def test_score_sample_size_brackets() -> None:
    """Component 5: Sample size brackets 0 to 10 points."""
    assert score_sample_size(55) == 10.0
    assert score_sample_size(35) == 8.0
    assert score_sample_size(25) == 6.0
    assert score_sample_size(15) == 4.0
    assert score_sample_size(7) == 2.0
    assert score_sample_size(3) == 0.0


def test_score_confirming_insights_brackets() -> None:
    """Component 6: Independent confirming insights brackets 0 to 15 points."""
    assert score_confirming_insights(5) == 15.0
    assert score_confirming_insights(3) == 12.0
    assert score_confirming_insights(2) == 8.0
    assert score_confirming_insights(1) == 4.0
    assert score_confirming_insights(0) == 0.0


def test_assign_play_tier() -> None:
    """Play tiers: ELITE (>=90), STRONG (>=80), QUALIFIED (>=70), LEAN (>=60), PASS (<60)."""
    assert assign_play_tier(92.0, edge=3.0) == PlayTier.ELITE
    assert assign_play_tier(84.5, edge=2.0) == PlayTier.STRONG
    assert assign_play_tier(73.0, edge=1.5) == PlayTier.QUALIFIED
    assert assign_play_tier(64.0, edge=0.5) == PlayTier.LEAN
    assert assign_play_tier(55.0, edge=0.5) == PlayTier.PASS

    # Non-positive edge automatically forces PASS regardless of high score
    assert assign_play_tier(95.0, edge=0.0) == PlayTier.PASS
    assert assign_play_tier(95.0, edge=-1.5) == PlayTier.PASS


# ---------------------------------------------------------------------------
# 6. Minimum Qualification Gates
# ---------------------------------------------------------------------------


def test_qualification_gates_all_pass() -> None:
    """When all gates clear, qualification status is QUALIFIED."""
    status, reasons = evaluate_qualification_gates(
        edge=3.0,
        ev=0.08,
        n_books=4,
        total_score=75.0,
        data_quality_score=90.0,
        sample_size=25,
        has_price=True,
        has_consensus_prob=True,
    )
    assert status == QualificationStatus.QUALIFIED
    assert len(reasons) == 0


def test_qualification_gates_veto_conditions() -> None:
    """Verify each minimum qualification gate vetoes appropriately."""
    # 1. Non-positive edge
    status_edge, reasons_edge = evaluate_qualification_gates(
        edge=0.0,
        ev=0.05,
        n_books=4,
        total_score=80.0,
    )
    assert status_edge == QualificationStatus.PASS
    assert "non_positive_edge" in reasons_edge

    # 2. Non-positive EV
    status_ev, reasons_ev = evaluate_qualification_gates(
        edge=3.0,
        ev=-0.02,
        n_books=4,
        total_score=80.0,
    )
    assert status_ev == QualificationStatus.PASS
    assert "non_positive_ev" in reasons_ev

    # 3. Thin market: n_books < 3
    status_books, reasons_books = evaluate_qualification_gates(
        edge=3.0,
        ev=0.05,
        n_books=2,
        total_score=80.0,
    )
    assert status_books == QualificationStatus.INSUFFICIENT_DATA
    assert "thin_market" in reasons_books

    # 4. Low data quality < 75.0
    status_dq, reasons_dq = evaluate_qualification_gates(
        edge=3.0,
        ev=0.05,
        n_books=4,
        total_score=80.0,
        data_quality_score=70.0,
    )
    assert status_dq == QualificationStatus.INSUFFICIENT_DATA
    assert "insufficient_model_data" in reasons_dq

    # 5. Small sample size < 5
    status_ss, reasons_ss = evaluate_qualification_gates(
        edge=3.0,
        ev=0.05,
        n_books=4,
        total_score=80.0,
        sample_size=3,
    )
    assert status_ss == QualificationStatus.INSUFFICIENT_DATA
    assert "sample_too_small" in reasons_ss

    # 6. Disqualifying flag (e.g. 'arb', 'invalid', 'placeholder')
    status_flag, reasons_flag = evaluate_qualification_gates(
        edge=3.0,
        ev=0.05,
        n_books=4,
        total_score=80.0,
        flags=["placeholder_price_dropped"],
    )
    assert status_flag == QualificationStatus.INSUFFICIENT_DATA
    assert any("placeholder" in r for r in reasons_flag)


# ---------------------------------------------------------------------------
# 7. Devig Multi-Method & MispricedScanner Orchestration
# ---------------------------------------------------------------------------


def test_devig_two_way_and_method_spread() -> None:
    """MispricedScanner computes Shin, Multiplicative, and method spread."""
    scanner = MispricedScanner()
    # Moderate favorite -200 vs +160
    p_shin, p_mult, method_spread = scanner.devig_two_way(-200, 160)
    assert 0.60 < p_shin < 0.70
    assert 0.60 < p_mult < 0.70
    assert method_spread >= 0.0
    assert math.isclose(method_spread, abs(p_shin - p_mult), abs_tol=1e-4)

    # Extreme favorite -1000 vs +600: Shin and Multiplicative diverge more
    p_shin_ext, p_mult_ext, spread_ext = scanner.devig_two_way(-1000, 600)
    assert spread_ext > method_spread


def test_devig_book_quotes_multi_book() -> None:
    """devig_book_quotes devigs across multiple sportsbooks."""
    scanner = MispricedScanner()
    quotes = {
        "DraftKings": [-110, -110],
        "FanDuel": [-108, -112],
        "BetMGM": [-115, -105],
    }
    p_shin, p_mult, spread = scanner.devig_book_quotes(quotes, side_index=0)
    assert math.isclose(p_shin, 0.50, abs_tol=0.02)
    assert math.isclose(p_mult, 0.50, abs_tol=0.02)
    assert spread < 0.02


def test_mispriced_scanner_reasoning_card_veto_integration() -> None:
    """When ReasoningCard has is_favorite_vetoed=True, candidate is downgraded to AVOID/REVIEW."""
    scanner = MispricedScanner()

    card = ReasoningCard(
        game_id="game_veto_test",
        market="SPREAD",
        side="HOME",
        recommended_play="PASS",
        confidence=2.5,
        tier="AVOID",
        tape_summary="Cupcake-padded offense; honest tape shows 14 PPG ceiling.",
        position_qb_summary="QB1 uncertain, backup has high turnover rate.",
        weather_venue_summary="Standard dome conditions.",
        injuries_trench_summary="Two starting OL out with knee injuries.",
        program_continuity_summary="First year coaching staff.",
        mathematical_edge_summary="Negative true margin edge.",
        contra_indications=["Trench attrition", "Backup QB"],
        is_favorite_vetoed=True,
        counter_thesis="Home favorite is overvalued due to junk FCS blowout tape.",
    )

    opp = scanner.scan_spread(
        game_id="game_veto_test",
        side="HOME",
        line=-14.0,
        projected_margin=18.0,
        posted_price_american=-110,
        consensus_fair_prob=0.5238,
        reasoning_card=card,
    )

    # Opp should be downgraded due to reasoning card counter-thesis
    assert opp.play_tier == PlayTier.AVOID.value
    assert opp.qual_status == QualificationStatus.REVIEW.value
    assert "vetoed_by_negative_gate" in opp.flags
    assert any("counter_thesis" in r for r in opp.rejection_reasons)
    assert opp.is_actionable is False


def test_mispriced_scanner_slate_scan_and_filter() -> None:
    """MispricedScanner.scan_slate correctly scans multiple markets and sorts by edge."""
    scanner = MispricedScanner()
    candidates = [
        # SPREAD candidate with strong edge
        {
            "game_id": "game_1",
            "market_type": "SPREAD",
            "side": "HOME",
            "line": -7.0,
            "projected_margin": 14.0,
            "posted_price_american": -110,
            "consensus_fair_prob": 0.5238,
            "sample_size": 25,
            "historical_hit_rate": 80.0,
        },
        # TOTAL candidate with positive edge
        {
            "game_id": "game_2",
            "market_type": "TOTAL",
            "side": "OVER",
            "line": 48.0,
            "projected_total": 56.5,
            "posted_price_american": -105,
            "consensus_fair_prob": 0.5122,
            "sample_size": 30,
            "historical_hit_rate": 75.0,
        },
        # Player prop (must be silently discarded by scan_slate)
        {
            "game_id": "game_3",
            "market_type": "TEAM_PROP",
            "market": "player_passing_yards",
            "side": "OVER",
            "line": 285.5,
            "projected_value": 310.0,
            "posted_price_american": -110,
            "consensus_fair_prob": 0.5238,
        },
        # TEAM_PROP candidate
        {
            "game_id": "game_4",
            "market_type": "TEAM_PROP",
            "market": "team_rushing_yards",
            "side": "OVER",
            "line": 150.5,
            "projected_value": 185.0,
            "posted_price_american": -110,
            "consensus_fair_prob": 0.5238,
            "sample_size": 25,
            "historical_hit_rate": 78.0,
        },
    ]

    results = scanner.scan_slate(candidates, sort_by="edge_pct")
    assert len(results) == 3  # Player prop was dropped
    assert results[0].edge_pct >= results[1].edge_pct >= results[2].edge_pct

    # Actionable filtering
    actionable = scanner.filter_actionable(results)
    assert all(a.is_actionable for a in actionable)


def test_mispriced_scanner_moneyline() -> None:
    """MispricedScanner.scan_moneyline evaluates moneyline candidates."""
    scanner = MispricedScanner()
    opp = scanner.scan_moneyline(
        game_id="game_ml_1",
        side="HOME",
        posted_price_american=150,
        model_prob=0.48,
        consensus_fair_prob=0.40,
        method_spread=0.015,
        n_books=4,
        sample_size=30,
        historical_hit_rate=72.0,
        confirming_insights_count=2,
    )

    assert opp.market_type == "ML"
    assert opp.side == "HOME"
    assert opp.model_prob == 0.48
    assert opp.consensus_fair_prob == 0.40
    assert opp.edge_pct == 0.08
    # Decimal price for +150 is 2.50; EV = 0.48 * 2.50 - 1.0 = +0.20
    assert math.isclose(opp.ev, 0.20, abs_tol=1e-4)
    assert opp.qual_status == QualificationStatus.QUALIFIED.value
    assert opp.play_score >= 70.0


def test_mispriced_scanner_devig_error_cases() -> None:
    """Devig methods raise DevigError on empty or invalid inputs."""
    scanner = MispricedScanner()
    with pytest.raises(DevigError):
        scanner.devig_book_quotes({})

    with pytest.raises(DevigError):
        scanner.devig_two_way(0, 0)


def test_under_line_movement_scoring() -> None:
    """Verify line movement scoring for UNDER side."""
    # Line rose by 2.0 (50.0 -> 52.0) -> confirming move > 1.5 -> 14 pts
    assert score_line_movement("UNDER", open_line=50.0, current_line=52.0) == 14.0
    # Line rose by 0.5 (50.0 -> 50.5) -> confirming move > 0 -> 10 pts
    assert score_line_movement("UNDER", open_line=50.0, current_line=50.5) == 10.0
    # Line fell by 1.0 (50.0 -> 49.0) -> adverse move <= 1.5 -> 2 pts
    assert score_line_movement("UNDER", open_line=50.0, current_line=49.0) == 2.0
    # Line fell by 2.5 (50.0 -> 47.5) -> adverse move > 1.5 -> 1 pt
    assert score_line_movement("UNDER", open_line=50.0, current_line=47.5) == 1.0


def test_spread_line_movement_scoring() -> None:
    """Verify spread line movement scoring for favorites, underdogs, and pick'em."""
    # 1. Favorite line moving favorably: hurdle gets easier (e.g. laying fewer points: -7.0 -> -5.0)
    # diff = -5.0 - (-7.0) = +2.0 -> confirming move > 1.5 -> 14.0 pts
    assert score_line_movement("HOME", open_line=-7.0, current_line=-5.0) == 14.0
    # Moderate favorable move: -7.0 -> -6.5 (diff = +0.5 -> confirming move > 0 -> 10.0 pts)
    assert score_line_movement("HOME", open_line=-7.0, current_line=-6.5) == 10.0

    # 2. Favorite line moving adversely: hurdle gets harder (e.g. laying more points: -7.0 -> -9.0)
    # diff = -9.0 - (-7.0) = -2.0 -> adverse move < -1.5 -> 1.0 pt
    assert score_line_movement("HOME", open_line=-7.0, current_line=-9.0) == 1.0
    # Moderate adverse move: -7.0 -> -8.0 (diff = -1.0 -> adverse move >= -1.5 -> 2.0 pts)
    assert score_line_movement("HOME", open_line=-7.0, current_line=-8.0) == 2.0

    # 3. Underdog line moving favorably: hurdle gets easier (e.g. getting more points: +7.0 -> +9.0)
    # diff = +9.0 - (+7.0) = +2.0 -> confirming move > 1.5 -> 14.0 pts
    assert score_line_movement("AWAY", open_line=7.0, current_line=9.0) == 14.0
    # Moderate favorable move: +7.0 -> +7.5 (diff = +0.5 -> confirming move > 0 -> 10.0 pts)
    assert score_line_movement("AWAY", open_line=7.0, current_line=7.5) == 10.0

    # 4. Underdog line moving adversely: hurdle gets harder (e.g. getting fewer points: +7.0 -> +5.0)
    # diff = +5.0 - (+7.0) = -2.0 -> adverse move < -1.5 -> 1.0 pt
    assert score_line_movement("AWAY", open_line=7.0, current_line=5.0) == 1.0
    # Moderate adverse move: +7.0 -> +6.0 (diff = -1.0 -> adverse move >= -1.5 -> 2.0 pts)
    assert score_line_movement("AWAY", open_line=7.0, current_line=6.0) == 2.0

    # 5. Flat / pick'em line movements:
    # Flat line on favorite (-7.0 -> -7.0, diff = 0.0 -> neutral -> 4.0 pts)
    assert score_line_movement("HOME", open_line=-7.0, current_line=-7.0) == 4.0
    # Flat line on underdog (+7.0 -> +7.0, diff = 0.0 -> neutral -> 4.0 pts)
    assert score_line_movement("AWAY", open_line=7.0, current_line=7.0) == 4.0
    # Pick'em (0.0 -> 0.0, diff = 0.0 -> neutral -> 4.0 pts)
    assert score_line_movement("HOME", open_line=0.0, current_line=0.0) == 4.0
