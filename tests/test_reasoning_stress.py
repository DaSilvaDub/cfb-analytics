"""Empirical Adversarial Stress Test Suite for Milestone 1 Reasoning Engine.

Tests stress oracles and wild boundary cases:
1. Extreme Weather Oracles: Hurricane (120 kph), Sub-zero (-20°C), Extreme Heat (45°C), Flood Rain (50 mm).
2. Wild Boundary Oracles: 0 completed games, all-cupcake schedule, missing coordinator data, extreme elevation (3000m).
3. Dataclass Immutability & Mutation Leakage Oracles.
4. Volume Redistribution Conservation & Boundary Saturation Oracles.
5. Negative Gate Multi-Contradiction Adversarial Stress.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import pytest

from cfb_analytics.reasoning.engine import MultiFactorReasoningEngine
from cfb_analytics.reasoning.models import (
    SHADOW_MODE_DISCLAIMER,
    NegativeGateResult,
    OpponentTier,
    PlayTier,
    PositionUnitGrades,
    ReasoningCard,
    SituationalContext,
    TalentProfile,
    TapeCategory,
    TapeGame,
    TapeProfile,
    TravelProfile,
    TrenchHealth,
    TrenchMatchupResult,
    VenueProfile,
    VolumeRedistribution,
    WeatherProfile,
)
from cfb_analytics.reasoning.negative_gate import NegativeFavoriteGate
from cfb_analytics.reasoning.roster import (
    build_talent_profile,
    compute_blue_chip_ratio,
    compute_true_talent_composite,
    evaluate_trench_attrition,
    evaluate_trench_matchup,
)
from cfb_analytics.reasoning.tape import (
    analyze_tape,
    classify_opponent,
    classify_tape_game,
    create_tape_game,
)
from cfb_analytics.reasoning.weather import (
    calculate_altitude_fatigue_tax,
    calculate_fg_range_boost,
    calculate_haversine_distance_miles,
    calculate_weather_multipliers,
    evaluate_travel_profile,
    parse_kickoff_et_hour,
    redistribute_pass_to_rush,
)


# ===========================================================================
# 1. Extreme Weather Oracles
# ===========================================================================


class TestExtremeWeatherOracles:
    """Stress tests atmospheric attenuation curves under extreme weather."""

    def test_hurricane_winds_120_kph(self) -> None:
        """Hurricane wind (120 kph, 150 kph gust) must hit floor boundaries."""
        w = calculate_weather_multipliers(
            temp_c=18.0,
            wind_kph=120.0,
            gust_kph=150.0,
            precip_mm=0.0,
            is_dome=False,
        )
        assert w.is_dome is False
        assert w.is_extreme_weather is True
        # Wind pass vol multiplier floor is 0.65
        assert w.pass_vol_mult <= 0.65
        # Scoring multiplier wind floor is 0.75
        assert w.scoring_mult <= 0.75
        # Rush volume multiplier expands by (1.0 - 0.65) * 0.85 = +0.2975 -> 1.2975
        assert w.rush_vol_mult >= 1.29

    def test_sub_zero_cold_minus_20c(self) -> None:
        """Sub-zero cold (-20°C) must trigger cold deficit cap of 25.0°C."""
        w = calculate_weather_multipliers(
            temp_c=-20.0,
            wind_kph=5.0,
            gust_kph=8.0,
            precip_mm=0.0,
            is_dome=False,
        )
        assert w.is_extreme_weather is True
        assert w.cold_deficit_c == 25.0
        # Cold deficit pass vol floor is 0.80
        assert w.pass_vol_mult <= 0.80
        assert w.scoring_mult <= 0.75
        assert w.rush_vol_mult >= 1.17

    def test_extreme_heat_45c(self) -> None:
        """Extreme desert heat (45°C) must trigger heat excess cap of 15.0°C."""
        w = calculate_weather_multipliers(
            temp_c=45.0,
            wind_kph=10.0,
            gust_kph=12.0,
            precip_mm=0.0,
            is_dome=False,
        )
        assert w.heat_excess_c == 15.0
        # Heat pass vol multiplier: 1.0 - 0.003 * 15 = 0.955
        assert w.pass_vol_mult == 0.955
        # Scoring multiplier in extreme heat: max(0.92, 1.0 - 0.006 * 15) = 0.92
        assert w.scoring_mult == 0.92
        # Rush ypc is unaffected by pure heat
        assert w.ypc_mult == 1.0

    def test_flood_rain_50mm(self) -> None:
        """Torrential flood rain (50 mm) must hit excess precipitation cap (15.0 mm)."""
        w = calculate_weather_multipliers(
            temp_c=16.0,
            wind_kph=10.0,
            gust_kph=12.0,
            precip_mm=50.0,
            is_dome=False,
        )
        assert w.is_extreme_weather is True
        assert w.pass_vol_mult <= 0.75
        assert w.scoring_mult <= 0.80
        assert w.rush_vol_mult >= 1.21

    def test_compound_weather_apocalypse(self) -> None:
        """Combined hurricane wind, flood rain, and sub-zero cold must respect floor bounds."""
        w = calculate_weather_multipliers(
            temp_c=-10.0,
            wind_kph=100.0,
            gust_kph=130.0,
            precip_mm=40.0,
            is_dome=False,
        )
        # Composite floors
        assert w.pass_vol_mult >= 0.50
        assert w.scoring_mult >= 0.55
        assert w.ypc_mult >= 0.50
        assert w.comp_prob_mult >= 0.50
        # Rush volume multiplier cannot exceed 1.0 + (1.0 - 0.50) * 0.85 = 1.425
        assert w.rush_vol_mult <= 1.425

    def test_dome_strict_bypass_under_apocalypse(self) -> None:
        """Domes must remain exactly 1.00x under identical apocalyptic outdoor inputs."""
        w = calculate_weather_multipliers(
            temp_c=-10.0,
            wind_kph=100.0,
            gust_kph=130.0,
            precip_mm=40.0,
            is_dome=True,
        )
        assert w.is_dome is True
        assert w.is_extreme_weather is False
        assert w.pass_vol_mult == 1.0
        assert w.rush_vol_mult == 1.0
        assert w.scoring_mult == 1.0
        assert w.effective_wind_kph == 0.0


# ===========================================================================
# 2. Wild Boundary Oracles
# ===========================================================================


class TestWildBoundariesOracles:
    """Stress tests boundary conditions: 0 games, all-cupcake, extreme altitude, etc."""

    def test_zero_completed_games_prior_regression(self) -> None:
        """Zero completed games must regress cleanly to priors without ZeroDivisionError."""
        profile = analyze_tape([], team_id="t_zero", team_name="Zero Games Team")
        assert profile.total_games == 0
        assert profile.honest_games_count == 0
        assert profile.cupcake_games_filtered == 0
        assert profile.has_honest_tape is False
        assert profile.is_cupcake_inflated is True
        assert profile.tape_confidence_penalty == 25.0
        assert profile.offensive_floor_epa == -0.20
        assert profile.offensive_ceiling_epa == 0.10
        assert profile.offensive_floor_points == 13
        assert profile.offensive_ceiling_points == 34
        assert "Zero honest tape" in profile.tape_summary

    def test_all_cupcake_schedule_purges_all_games(self) -> None:
        """A team playing only cupcake blowouts must have all games purged from honest baseline."""
        cupcake_games = [
            create_tape_game(
                game_id=f"g_fcs_{i}",
                season=2026,
                week=i,
                kickoff_utc="2026-09-01T19:00:00Z",
                opponent_team_id=f"fcs_{i}",
                opponent_name=f"FCS Opponent {i}",
                opponent_conference="FCS",
                opponent_classification="fcs",
                is_home=True,
                points_scored=56,
                points_allowed=3,
                offensive_epa=0.55,
                defensive_epa=-0.40,
                offensive_success_rate=0.68,
                defensive_success_rate=0.18,
            )
            for i in range(1, 5)
        ]
        profile = analyze_tape(cupcake_games, team_id="t_fcs", team_name="Paper Tiger")
        assert profile.total_games == 4
        assert profile.cupcake_games_filtered == 4
        assert profile.honest_games_count == 0
        assert profile.has_honest_tape is False
        assert profile.raw_ceiling_points == 56
        assert profile.offensive_ceiling_points == 34  # Prior regressed ceiling
        assert profile.inflation_gap_points == 22.0
        assert profile.is_cupcake_inflated is True
        assert profile.tape_confidence_penalty == 25.0

    def test_extreme_elevation_3000m_caps_fatigue_and_fg_boost(self) -> None:
        """Elevation of 3000m (e.g. high mountain venue) must hit capped taxes."""
        tax = calculate_altitude_fatigue_tax(venue_elevation_m=3000.0, visitor_elevation_m=0.0)
        assert tax == 2.5  # Max cap is 2.5 points
        boost = calculate_fg_range_boost(venue_elevation_m=3000.0)
        assert boost == 4.5  # Max cap is 4.5 yards

    def test_negative_elevation_handled_cleanly(self) -> None:
        """Negative elevation (below sea level venue) must yield zero tax and zero boost."""
        tax = calculate_altitude_fatigue_tax(venue_elevation_m=-86.0, visitor_elevation_m=0.0)
        assert tax == 0.0
        boost = calculate_fg_range_boost(venue_elevation_m=-86.0)
        assert boost == 0.0

    def test_missing_coordinator_data_handled_cleanly(self) -> None:
        """Context with zero coaching continuity rating evaluates without error."""
        tape = TapeProfile()
        weather = WeatherProfile(
            temperature_c=20.0,
            wind_kph=10.0,
            gust_kph=10.0,
            precip_mm=0.0,
            is_dome=True,
            pass_vol_mult=1.0,
            rush_vol_mult=1.0,
            scoring_mult=1.0,
        )
        ctx = SituationalContext(
            game_id="g_coord",
            home_team="Team A",
            away_team="Team B",
            kickoff_utc="2026-10-10T19:00:00Z",
            kickoff_et="7:00 p.m.",
            tape_home=tape,
            tape_away=tape,
            weather=weather,
            qb_home_confirmed=True,
            qb_away_confirmed=True,
            trench_attrition_home=0.0,
            trench_attrition_away=0.0,
            talent_composite_home=750.0,
            talent_composite_away=700.0,
            rest_days_home=7.0,
            rest_days_away=7.0,
            travel_fatigue_tax_away=0.0,
            coaching_continuity_home=0.0,  # Missing/brand-new coordinator staff
            coaching_continuity_away=0.0,
        )
        engine = MultiFactorReasoningEngine()
        card = engine.evaluate_candidate(
            ctx, market="SPREAD", side="HOME", line=-7.0
        )
        assert "Staff continuity rating 0.0/1.0" in card.program_continuity_summary

    def test_kickoff_hour_parser_robustness(self) -> None:
        """Malformed or irregular kickoff strings parse gracefully or yield None."""
        assert parse_kickoff_et_hour("") is None
        assert parse_kickoff_et_hour("TBD") is None
        assert parse_kickoff_et_hour("12:00 PM") == 12.0
        assert parse_kickoff_et_hour("12:00 AM") == 0.0
        assert parse_kickoff_et_hour("11:59 pm") == 23.98


# ===========================================================================
# 3. Dataclass Immutability & Mutation Leakage Oracles
# ===========================================================================


class TestDataclassImmutabilityOracles:
    """Verifies that dataclass attributes cannot be overwritten."""

    def test_direct_assignment_raises_frozen_instance_error(self) -> None:
        pos = PositionUnitGrades(team_id="t1")
        with pytest.raises(FrozenInstanceError):
            pos.team_id = "t2"  # type: ignore[misc]

        tape = TapeProfile()
        with pytest.raises(FrozenInstanceError):
            tape.offensive_floor_epa = 0.5  # type: ignore[misc]

        weather = WeatherProfile(
            temperature_c=20.0,
            wind_kph=0.0,
            gust_kph=0.0,
            precip_mm=0.0,
            is_dome=True,
            pass_vol_mult=1.0,
            rush_vol_mult=1.0,
            scoring_mult=1.0,
        )
        with pytest.raises(FrozenInstanceError):
            weather.temperature_c = 10.0  # type: ignore[misc]

    def test_reasoning_card_contra_indications_collection_type(self) -> None:
        """Adversarial probe: check if ReasoningCard.contra_indications allows inner mutation."""
        card = ReasoningCard(
            game_id="g1",
            market="SPREAD",
            side="HOME",
            recommended_play="HOME -14",
            confidence=8.0,
            tier="STRONG",
            tape_summary="tape",
            position_qb_summary="qb",
            weather_venue_summary="weather",
            injuries_trench_summary="trench",
            program_continuity_summary="prog",
            mathematical_edge_summary="math",
            contra_indications=["Initial warning"],
        )
        # Direct field assignment fails
        with pytest.raises(FrozenInstanceError):
            card.confidence = 9.0  # type: ignore[misc]

        # Inner list mutation: contra_indications is typed as list[str]
        # In a strictly immutable system, this should ideally be an immutable tuple.
        # Here we test that the list reference itself cannot be replaced:
        with pytest.raises(FrozenInstanceError):
            card.contra_indications = ["Replaced list"]  # type: ignore[misc]


# ===========================================================================
# 4. Volume Redistribution Conservation & Boundary Oracles
# ===========================================================================


class TestVolumeRedistributionLawConservation:
    """Verifies pass attenuation translates into rush boost via the 0.85 law."""

    def test_conservation_under_standard_envelope(self) -> None:
        """In standard ranges, rush_boost must equal round(delta_pass * 0.85, 1)."""
        base_pass = 38.0
        base_rush = 32.0
        pass_vol_mult = 0.70  # 30% pass attenuation

        redist = redistribute_pass_to_rush(base_pass, base_rush, pass_vol_mult)

        expected_adj_pass = round(38.0 * 0.70, 1)  # 26.6
        expected_delta_pass = round(base_pass - expected_adj_pass, 1)  # 11.4
        expected_rush_boost = round(expected_delta_pass * 0.85, 1)  # 9.7
        expected_adj_rush = round(base_rush + expected_rush_boost, 1)  # 41.7

        assert redist.adj_pass_attempts == expected_adj_pass
        assert redist.delta_pass == expected_delta_pass
        assert redist.rush_boost == expected_rush_boost
        assert redist.adj_rush_attempts == expected_adj_rush

        # Clock runoff pace contraction: pace should contract by 15% of lost pass attempts
        expected_lost_pace = round(expected_delta_pass * 0.15, 1)
        actual_lost_pace = round((base_pass + base_rush) - redist.adj_pace, 1)
        assert math.isclose(actual_lost_pace, expected_lost_pace, abs_tol=0.2)

    def test_low_pass_option_offense_clamp_boundary(self) -> None:
        """Triple-option offenses with low pass attempts (<10) trigger the max(10.0, ...) clamp."""
        base_pass = 8.0  # Navy / Army style
        base_rush = 58.0
        pass_vol_mult = 0.60

        redist = redistribute_pass_to_rush(base_pass, base_rush, pass_vol_mult)
        # Because of max(10.0, base_pass * mult), adj_pass becomes 10.0
        assert redist.adj_pass_attempts == 10.0
        # delta_pass is max(0.0, 8.0 - 10.0) = 0.0
        assert redist.delta_pass == 0.0
        assert redist.rush_boost == 0.0
        assert redist.adj_rush_attempts == 58.0

    def test_heavy_rush_saturation_clamp_boundary(self) -> None:
        """When base_rush + rush_boost exceeds 75.0, adj_rush is capped at 75.0."""
        base_pass = 40.0
        base_rush = 72.0
        pass_vol_mult = 0.50

        redist = redistribute_pass_to_rush(base_pass, base_rush, pass_vol_mult)
        assert redist.adj_pass_attempts == 20.0
        assert redist.delta_pass == 20.0
        assert redist.rush_boost == 17.0
        # 72.0 + 17.0 = 89.0 -> capped at 75.0
        assert redist.adj_rush_attempts == 75.0


# ===========================================================================
# 5. Negative Gate Multi-Contradiction Stress
# ===========================================================================


class TestNegativeGateAdversarialStress:
    """Stress tests the negative gate under compound adversarial failure modes."""

    def test_multiple_critical_contradictions_escalate_penalty(self) -> None:
        """Favorite with unconfirmed QB AND severe trench attrition receives maximum penalty."""
        bad_tape = TapeProfile()
        weather = WeatherProfile(
            temperature_c=20.0,
            wind_kph=0.0,
            gust_kph=0.0,
            precip_mm=0.0,
            is_dome=True,
            pass_vol_mult=1.0,
            rush_vol_mult=1.0,
            scoring_mult=1.0,
        )
        ctx = SituationalContext(
            game_id="g_multi_fail",
            home_team="Fake Fave",
            away_team="Underdog",
            kickoff_utc="2026-10-10T19:00:00Z",
            kickoff_et="3:30 p.m.",
            tape_home=bad_tape,
            tape_away=bad_tape,
            weather=weather,
            qb_home_confirmed=False,  # Critical 1
            qb_away_confirmed=True,
            trench_attrition_home=0.45,  # Critical 2
            trench_attrition_away=0.0,
            talent_composite_home=750.0,
            talent_composite_away=700.0,
            rest_days_home=7.0,
            rest_days_away=7.0,
            travel_fatigue_tax_away=0.0,
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-14.0)
        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert len(res.disqualifying_reasons) >= 2
        assert res.confidence_penalty >= 5.0
        assert "Unconfirmed quarterback status" in str(res.counter_thesis)
        assert "Critical trench attrition" in str(res.counter_thesis)

    def test_grok_rule_d_vetoes_thin_dog_without_structural_mismatch(self) -> None:
        """Thin dog (+2.5) without opposing QB injury or trench advantage is vetoed."""
        tape = TapeProfile(
            honest_games=[{"opp": "Opp", "margin": 3}],
            offensive_floor_epa=0.05,
            defensive_floor_epa=0.00,
        )
        weather = WeatherProfile(
            temperature_c=20.0,
            wind_kph=0.0,
            gust_kph=0.0,
            precip_mm=0.0,
            is_dome=True,
            pass_vol_mult=1.0,
            rush_vol_mult=1.0,
            scoring_mult=1.0,
        )
        ctx = SituationalContext(
            game_id="g_thin_dog",
            home_team="Favorite",
            away_team="Thin Dog",
            kickoff_utc="2026-10-10T19:00:00Z",
            kickoff_et="3:30 p.m.",
            tape_home=tape,
            tape_away=tape,
            weather=weather,
            qb_home_confirmed=True,  # Opponent QB is healthy!
            qb_away_confirmed=True,
            trench_attrition_home=0.0,
            trench_attrition_away=0.0,
            talent_composite_home=800.0,
            talent_composite_away=750.0,
            rest_days_home=7.0,
            rest_days_away=7.0,
            travel_fatigue_tax_away=0.0,
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="AWAY", line=2.5)
        assert res.is_vetoed is True
        assert "thin_dog_trap_rule_d" in res.disqualifying_reasons
        assert "Rule D" in str(res.counter_thesis)
