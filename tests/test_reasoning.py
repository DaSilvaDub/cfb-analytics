"""Comprehensive unit tests for Multi-Factor Reasoning Engine (Milestone 1).

Tests:
1. Immutability & Contract Invariants of Reasoning Dataclasses.
2. Opponent Classification & Honest vs Junk Cupcake Tape Analysis.
3. Situational Weather, Dome Bypass, Elevation & Travel Fatigue (Rule C).
4. True Talent Composite, Blue-Chip Ratios & Trench Attrition (Rule D).
5. Negative Favorite Gate 5-Dimensional Auditing & Counter-Theses.
6. MultiFactorReasoningEngine Orchestration, Summaries & Epistemic Confidence.
7. Shadow Mode Disclaimer Watermarking.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
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

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_tape() -> TapeProfile:
    return TapeProfile(
        honest_games=[{"opp": "Auburn", "margin": 14}],
        offensive_floor_epa=0.18,
        offensive_ceiling_epa=0.35,
        defensive_floor_epa=-0.12,
        defensive_ceiling_epa=0.08,
        cupcake_games_filtered=0,
        honest_success_rate=0.48,
        honest_yards_per_play=6.2,
    )


@pytest.fixture
def base_weather() -> WeatherProfile:
    return WeatherProfile(
        temperature_c=22.0,
        wind_kph=10.0,
        gust_kph=15.0,
        precip_mm=0.0,
        is_dome=False,
        pass_vol_mult=1.0,
        rush_vol_mult=1.0,
        scoring_mult=1.0,
        effective_wind_kph=11.2,
    )


@pytest.fixture
def base_context(base_tape: TapeProfile, base_weather: WeatherProfile) -> SituationalContext:
    return SituationalContext(
        game_id="cfbd:2026_test_01",
        home_team="Georgia",
        away_team="Vanderbilt",
        kickoff_utc="2026-10-10T19:30:00Z",
        kickoff_et="3:30 p.m.",
        tape_home=base_tape,
        tape_away=base_tape,
        weather=base_weather,
        qb_home_confirmed=True,
        qb_away_confirmed=True,
        qb_home_name="C. Beck",
        qb_away_name="D. Pavia",
        qb_home_attempt_share=0.92,
        qb_away_attempt_share=0.88,
        trench_attrition_home=0.0,
        trench_attrition_away=0.0,
        talent_composite_home=950.0,
        talent_composite_away=680.0,
        rest_days_home=7.0,
        rest_days_away=7.0,
        travel_fatigue_tax_away=0.0,
        coaching_continuity_home=1.0,
        coaching_continuity_away=1.0,
        lookahead_flag_home=False,
        lookahead_flag_away=False,
        venue_elevation_m=200.0,
        is_neutral_site=False,
    )


# ---------------------------------------------------------------------------
# 1. Model Immutability & Contract Tests
# ---------------------------------------------------------------------------


class TestReasoningModelContracts:
    """Verifies dataclass immutability, defaults, and watermarking."""

    def test_reasoning_card_is_immutable(self) -> None:
        card = ReasoningCard(
            game_id="g1",
            market="SPREAD",
            side="HOME",
            recommended_play="HOME -14.0",
            confidence=8.5,
            tier="STRONG",
            tape_summary="tape",
            position_qb_summary="qb",
            weather_venue_summary="weather",
            injuries_trench_summary="trench",
            program_continuity_summary="prog",
            mathematical_edge_summary="math",
            contra_indications=[],
            is_favorite_vetoed=False,
            counter_thesis=None,
        )
        with pytest.raises(FrozenInstanceError):
            card.confidence = 9.0  # type: ignore[misc]

    def test_situational_context_is_immutable(self, base_context: SituationalContext) -> None:
        with pytest.raises(FrozenInstanceError):
            base_context.qb_home_confirmed = False  # type: ignore[misc]

    def test_tape_game_is_immutable(self) -> None:
        game = TapeGame(
            game_id="g1",
            season=2026,
            week=1,
            kickoff_utc="2026-09-05T19:00:00Z",
            opponent_team_id="t2",
            opponent_name="FCS Opponent",
            opponent_conference="FCS",
            opponent_classification="fcs",
            is_home=True,
            points_scored=49,
            points_allowed=3,
            margin=46,
            offensive_epa=0.45,
            defensive_epa=-0.25,
            offensive_success_rate=0.55,
            defensive_success_rate=0.25,
            is_cupcake_blowout=True,
        )
        with pytest.raises(FrozenInstanceError):
            game.points_scored = 50  # type: ignore[misc]

    def test_shadow_mode_disclaimer_constant(self) -> None:
        assert SHADOW_MODE_DISCLAIMER == "UNPROMOTED - shadow output, not decision-grade"
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
        )
        assert card.shadow_mode_disclaimer == SHADOW_MODE_DISCLAIMER

    def test_other_dataclasses_are_immutable(self) -> None:
        pos = PositionUnitGrades(team_id="t1", qb_grade=88.0)
        with pytest.raises(FrozenInstanceError):
            pos.qb_grade = 90.0  # type: ignore[misc]

        venue = VenueProfile(venue_id="v1", name="Sanford Stadium", elevation_m=200.0)
        with pytest.raises(FrozenInstanceError):
            venue.elevation_m = 300.0  # type: ignore[misc]

        travel = TravelProfile(
            distance_miles=500.0,
            timezone_shift_hours=0,
            is_westward=False,
            kickoff_et_hour=15.5,
            is_late_kickoff=False,
            distance_fatigue_tax=0.0,
            timezone_fatigue_tax=0.0,
            late_kickoff_fatigue_tax=0.0,
            altitude_fatigue_tax=0.0,
            total_travel_fatigue_tax=0.0,
            rule_c_triggered=False,
        )
        with pytest.raises(FrozenInstanceError):
            travel.distance_miles = 600.0  # type: ignore[misc]

        redist = VolumeRedistribution(
            orig_pass_attempts=40.0,
            orig_rush_attempts=30.0,
            adj_pass_attempts=32.0,
            adj_rush_attempts=36.8,
            delta_pass=8.0,
            rush_boost=6.8,
            adj_pace=68.8,
        )
        with pytest.raises(FrozenInstanceError):
            redist.adj_pace = 70.0  # type: ignore[misc]

        talent = TalentProfile(
            team_id="t1",
            recruiting_composite=800.0,
            recruiting_rank=15,
            net_portal_composite=5.0,
            true_talent_composite=813.75,
            blue_chip_ratio=0.55,
            is_blue_chip_program=True,
            talent_tier="ELITE",
        )
        with pytest.raises(FrozenInstanceError):
            talent.blue_chip_ratio = 0.60  # type: ignore[misc]

        trench = TrenchHealth(
            team_id="t1",
            ol_attrition=0.20,
            dl_attrition=0.10,
            composite_trench_attrition=0.155,
        )
        with pytest.raises(FrozenInstanceError):
            trench.ol_attrition = 0.30  # type: ignore[misc]

        matchup = TrenchMatchupResult(
            offense_team_id="t1",
            defense_team_id="t2",
            ol_health=trench,
            dl_health=trench,
            projected_line_yards=3.1,
            projected_sack_rate=0.05,
            trench_mismatch_score=0.4,
            is_decisive_mismatch=False,
        )
        with pytest.raises(FrozenInstanceError):
            matchup.trench_mismatch_score = 1.0  # type: ignore[misc]

        gate_res = NegativeGateResult(
            is_vetoed=False,
            gate_status="CLEARED",
            counter_thesis=None,
            contra_indications=(),
            disqualifying_reasons=(),
            confidence_penalty=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            gate_res.is_vetoed = True  # type: ignore[misc]

    def test_play_tier_enum(self) -> None:
        assert PlayTier.ELITE.value == "ELITE"
        assert PlayTier.STRONG.value == "STRONG"
        assert PlayTier.QUALIFIED.value == "QUALIFIED"
        assert PlayTier.LEAN.value == "LEAN"
        assert PlayTier.PASS.value == "PASS"
        assert PlayTier.AVOID.value == "AVOID"


# ---------------------------------------------------------------------------
# 2. Opponent Classification & Honest vs Junk Tape Tests
# ---------------------------------------------------------------------------


class TestTapeReasoning:
    """Verifies opponent tiers, blowout filtering, and honest efficiency bounds."""

    def test_classify_opponent_tiers(self) -> None:
        assert classify_opponent(conference="SEC") == OpponentTier.POWER_CONFERENCE
        assert classify_opponent(conference="Big Ten") == OpponentTier.POWER_CONFERENCE
        assert classify_opponent(school="Notre Dame") == OpponentTier.POWER_CONFERENCE
        assert classify_opponent(classification="fcs") == OpponentTier.CUPCAKE_FCS
        assert classify_opponent(conference="Big Sky") == OpponentTier.CUPCAKE_FCS
        assert classify_opponent(conference="Mountain West") == OpponentTier.HONEST_FBS
        # Bottom G5 with low Elo
        assert (
            classify_opponent(conference="MAC", elo_rating=1280.0)
            == OpponentTier.BOTTOM_G5
        )

    def test_classify_tape_game_blowout_vs_struggle(self) -> None:
        # Blowout against FCS is junk tape
        cat_blowout = classify_tape_game(49, 3, OpponentTier.CUPCAKE_FCS)
        assert cat_blowout == TapeCategory.CUPCAKE_BLOWOUT

        # Close game against FCS establishes honest floor
        cat_struggle = classify_tape_game(17, 14, OpponentTier.CUPCAKE_FCS)
        assert cat_struggle == TapeCategory.HONEST_STRUGGLE

        # Game against Power is always honest
        cat_power = classify_tape_game(42, 10, OpponentTier.POWER_CONFERENCE)
        assert cat_power == TapeCategory.HONEST_POWER

    def test_analyze_tape_filters_cupcake_blowouts(self) -> None:
        g1 = create_tape_game(
            game_id="g1",
            season=2026,
            week=1,
            kickoff_utc="2026-09-01T19:00:00Z",
            opponent_team_id="fcs1",
            opponent_name="Mississippi Valley State",
            opponent_conference="SWAC",
            opponent_classification="fcs",
            is_home=True,
            points_scored=52,
            points_allowed=0,
            offensive_epa=0.45,
            defensive_epa=-0.35,
            offensive_success_rate=0.62,
            defensive_success_rate=0.20,
        )
        g2 = create_tape_game(
            game_id="g2",
            season=2026,
            week=2,
            kickoff_utc="2026-09-08T19:00:00Z",
            opponent_team_id="p4_1",
            opponent_name="Florida",
            opponent_conference="SEC",
            opponent_classification="fbs",
            is_home=False,
            points_scored=24,
            points_allowed=21,
            offensive_epa=0.08,
            defensive_epa=0.02,
            offensive_success_rate=0.44,
            defensive_success_rate=0.42,
        )
        g3 = create_tape_game(
            game_id="g3",
            season=2026,
            week=3,
            kickoff_utc="2026-09-15T19:00:00Z",
            opponent_team_id="p4_2",
            opponent_name="Tennessee",
            opponent_conference="SEC",
            opponent_classification="fbs",
            is_home=True,
            points_scored=13,
            points_allowed=31,
            offensive_epa=-0.12,
            defensive_epa=0.22,
            offensive_success_rate=0.36,
            defensive_success_rate=0.52,
        )

        profile = analyze_tape([g1, g2, g3], team_id="t1", team_name="Georgia")

        assert profile.total_games == 3
        assert profile.cupcake_games_filtered == 1
        assert profile.honest_games_count == 2
        assert profile.has_honest_tape is True
        # Ceilings must NOT be 52 pts or 0.45 EPA!
        assert profile.offensive_ceiling_points == 24
        assert profile.offensive_ceiling_epa == 0.08
        assert profile.offensive_floor_points == 13
        assert profile.offensive_floor_epa == -0.12
        assert profile.raw_ceiling_points == 52
        assert profile.inflation_gap_points == 28.0
        assert profile.is_cupcake_inflated is True
        assert "CUPCAKE INFLATION DETECTED" in profile.tape_summary

    def test_analyze_tape_zero_honest_games_fallback(self) -> None:
        g_cupcake = create_tape_game(
            game_id="g1",
            season=2026,
            week=1,
            kickoff_utc="2026-09-01T19:00:00Z",
            opponent_team_id="fcs1",
            opponent_name="Sacramento State",
            opponent_conference="Big Sky",
            opponent_classification="fcs",
            is_home=True,
            points_scored=55,
            points_allowed=7,
            offensive_epa=0.50,
            defensive_epa=-0.30,
            offensive_success_rate=0.65,
            defensive_success_rate=0.22,
        )
        profile = analyze_tape([g_cupcake], team_id="t1", team_name="Michigan")

        assert profile.has_honest_tape is False
        assert profile.cupcake_games_filtered == 1
        assert profile.offensive_ceiling_points < 50
        assert profile.tape_confidence_penalty == 25.0
        assert "Zero honest tape" in profile.tape_summary


# ---------------------------------------------------------------------------
# 3. Situational Weather, Venue & Travel Fatigue Tests
# ---------------------------------------------------------------------------


class TestWeatherAndTravelReasoning:
    """Verifies atmospheric attenuation, volume redistribution, and elevation/travel tax."""

    def test_outdoor_weather_attenuation(self) -> None:
        w = calculate_weather_multipliers(
            temp_c=2.0,  # Cold
            wind_kph=35.0,  # High wind
            gust_kph=50.0,
            precip_mm=3.0,  # Rain
            is_dome=False,
        )
        assert w.is_dome is False
        assert w.pass_vol_mult < 1.0
        assert w.scoring_mult < 1.0
        assert w.rush_vol_mult > 1.0
        assert w.is_extreme_weather is True

    def test_volume_redistribution_law(self) -> None:
        # Base 40 passes, 30 rushes. With 0.75 pass_vol_mult -> 30 passes (-10 delta).
        # Rush attempts absorb 0.85 * 10 = +8.5 rushes -> 38.5.
        redist = redistribute_pass_to_rush(base_pass=40.0, base_rush=30.0, pass_vol_mult=0.75)
        assert redist.orig_pass_attempts == 40.0
        assert redist.adj_pass_attempts == 30.0
        assert redist.delta_pass == 10.0
        assert redist.rush_boost == 8.5
        assert redist.adj_rush_attempts == 38.5

    def test_dome_strict_bypass(self) -> None:
        w = calculate_weather_multipliers(
            temp_c=-15.0,
            wind_kph=60.0,
            gust_kph=80.0,
            precip_mm=10.0,
            is_dome=True,
        )
        assert w.is_dome is True
        assert w.pass_vol_mult == 1.0
        assert w.rush_vol_mult == 1.0
        assert w.scoring_mult == 1.0
        assert w.effective_wind_kph == 0.0
        assert w.is_extreme_weather is False

    def test_altitude_fatigue_and_fg_boost(self) -> None:
        # Laramie, WY at 2200m
        tax = calculate_altitude_fatigue_tax(venue_elevation_m=2200.0, visitor_elevation_m=200.0)
        assert tax >= 1.5
        boost = calculate_fg_range_boost(2200.0)
        assert boost >= 4.0

    def test_haversine_distance(self) -> None:
        # Atlanta (33.7490, -84.3880) to Los Angeles (34.0522, -118.2437) ~ 1940 miles
        dist = calculate_haversine_distance_miles(33.7490, -84.3880, 34.0522, -118.2437)
        assert 1900.0 <= dist <= 2000.0

    def test_travel_profile_late_kickoff_rule_c(self) -> None:
        tp = evaluate_travel_profile(
            distance_miles=2100.0,
            timezone_shift_hours=3,
            is_westward=True,
            kickoff_et="10:30 p.m.",
            spread_line=-24.5,
        )
        assert tp.is_late_kickoff is True
        assert tp.total_travel_fatigue_tax >= 2.5
        assert tp.rule_c_triggered is True

    def test_parse_kickoff_et_hour(self) -> None:
        assert parse_kickoff_et_hour("10:30 p.m.") == 22.5
        assert parse_kickoff_et_hour("3:30 p.m.") == 15.5
        assert parse_kickoff_et_hour("11:00 am") == 11.0
        assert parse_kickoff_et_hour("10:30PM ET") == 22.5


# ---------------------------------------------------------------------------
# 4. Roster Talent & Trench Attrition Tests
# ---------------------------------------------------------------------------


class TestRosterAndTrenchReasoning:
    """Verifies True Talent Composite, Blue-Chip Ratios, and Trench Attrition."""

    def test_compute_true_talent_composite(self) -> None:
        # 750 recruiting composite + 20 net portal -> 750 + (20 * 2.75) = 805.0
        ttc = compute_true_talent_composite(750.0, 20.0)
        assert ttc == 805.0
        with pytest.raises(ValueError):
            compute_true_talent_composite(-50.0, 0.0)

    def test_compute_blue_chip_ratio(self) -> None:
        bcr = compute_blue_chip_ratio(four_star_count=15, five_star_count=3, total_signees=25)
        assert bcr == 0.72
        assert compute_blue_chip_ratio(0, 0, 0) == 0.0

    def test_build_talent_profile_tiers(self) -> None:
        prof = build_talent_profile(
            team_id="t1",
            recruiting_composite=880.0,
            net_portal=10.0,
            four_star_count=16,
            five_star_count=2,
            total_signees=25,
        )
        assert prof.is_blue_chip_program is True
        assert prof.talent_tier == "ELITE"

    def test_evaluate_trench_attrition_starting_ol(self) -> None:
        # Left tackle and Center out
        injuries = [
            {"position": "LT", "designation": "Out"},
            {"position": "C", "designation": "Out"},
        ]
        trench = evaluate_trench_attrition(team_id="t1", injuries=injuries)
        assert trench.ol_attrition >= 0.40
        assert trench.line_yards_mult < 0.90
        assert trench.sack_rate_allowed_mult > 1.20
        assert trench.is_trench_compromised is True

    def test_evaluate_trench_matchup_decisive(self) -> None:
        ol_bad = TrenchHealth(
            team_id="t1",
            ol_attrition=0.80,
            dl_attrition=0.0,
            composite_trench_attrition=0.44,
            line_yards_mult=0.70,
            sack_rate_allowed_mult=1.60,
        )
        dl_good = TrenchHealth(
            team_id="t2",
            ol_attrition=0.0,
            dl_attrition=0.0,
            composite_trench_attrition=0.0,
            def_stuff_rate_mult=1.35,
            def_havoc_mult=1.40,
            def_sack_rate_mult=1.45,
        )
        matchup = evaluate_trench_matchup("t1", "t2", ol_bad, dl_good)
        assert matchup.is_decisive_mismatch is True
        assert abs(matchup.trench_mismatch_score) >= 1.25
        assert "DECISIVE" in matchup.summary_text
        assert "Trench Matchup" in matchup.summary_text


# ---------------------------------------------------------------------------
# 5. Negative Favorite Gate Tests
# ---------------------------------------------------------------------------


class TestNegativeFavoriteGate:
    """Exhaustive tests for negative_gate.py across all 5 dimensions."""

    def test_cleared_favorite_with_affirmative_edges(self, base_context: SituationalContext) -> None:
        gate = NegativeFavoriteGate()
        res = gate.evaluate(
            base_context,
            market="SPREAD",
            side="HOME",
            line=-14.5,
            consensus_fair_prob=0.55,
        )
        assert res.is_vetoed is False
        assert res.gate_status == "CLEARED"
        assert res.counter_thesis is None
        assert res.confidence_penalty == 0.0

    def test_cupcake_tape_mirage_veto(self, base_context: SituationalContext) -> None:
        bad_tape = TapeProfile(
            honest_games=[{"opp": "Tennessee", "margin": -7}],
            offensive_floor_epa=-0.08,
            offensive_ceiling_epa=0.10,
            defensive_floor_epa=0.15,
            defensive_ceiling_epa=0.25,
            cupcake_games_filtered=2,
            honest_success_rate=0.38,
            honest_yards_per_play=4.6,
        )
        ctx = SituationalContext(**{**base_context.__dict__, "tape_home": bad_tape})
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-17.5)

        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert "cupcake_tape_mirage" in res.disqualifying_reasons
        assert "inflated by cupcake blowout tape" in str(res.counter_thesis)
        assert res.confidence_penalty >= 4.0

    def test_road_fatigue_late_kickoff_rule_c_veto(
        self, base_context: SituationalContext
    ) -> None:
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "kickoff_et": "10:30 p.m.",
                "travel_fatigue_tax_away": 2.2,
            }
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="AWAY", line=-25.5)

        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert "fat_road_dog_rule_c_veto" in res.disqualifying_reasons
        assert "Rule C Veto" in str(res.counter_thesis)

    def test_unconfirmed_qb_disqualification(
        self, base_context: SituationalContext
    ) -> None:
        ctx = SituationalContext(
            **{**base_context.__dict__, "qb_home_confirmed": False}
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-14.0)

        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert "unconfirmed_qb_disqualification" in res.disqualifying_reasons
        assert "Unconfirmed quarterback status" in str(res.counter_thesis)

    def test_severe_trench_attrition_veto(
        self, base_context: SituationalContext
    ) -> None:
        ctx = SituationalContext(
            **{**base_context.__dict__, "trench_attrition_home": 0.40}
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-13.5)

        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert "severe_trench_attrition" in res.disqualifying_reasons
        assert "Critical trench attrition" in str(res.counter_thesis)

    def test_compound_rest_lookahead_trap_veto(
        self, base_context: SituationalContext
    ) -> None:
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "rest_days_home": 6.0,
                "rest_days_away": 14.0,
                "lookahead_flag_home": True,
            }
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-14.5)

        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert "compound_rest_lookahead_trap" in res.disqualifying_reasons
        assert "Classic trap spot" in str(res.counter_thesis)

    def test_brand_name_talent_mismatch_veto(
        self, base_context: SituationalContext
    ) -> None:
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "talent_composite_home": 715.0,
                "talent_composite_away": 700.0,
            }
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-17.5)

        assert res.is_vetoed is True
        assert "brand_name_talent_mismatch" in res.disqualifying_reasons
        assert "Brand-name premium trap" in str(res.counter_thesis)

    def test_thin_dog_trap_rule_d_veto(self, base_context: SituationalContext) -> None:
        gate = NegativeFavoriteGate()
        res = gate.evaluate(base_context, market="SPREAD", side="AWAY", line=3.5)

        assert res.is_vetoed is True
        assert "thin_dog_trap_rule_d" in res.disqualifying_reasons
        assert "Rule D" in str(res.counter_thesis)

    def test_thin_dog_rule_d_exception_clears(
        self, base_context: SituationalContext
    ) -> None:
        # Opponent QB is unconfirmed
        ctx = SituationalContext(
            **{**base_context.__dict__, "qb_home_confirmed": False}
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="AWAY", line=3.5)

        assert res.is_vetoed is False
        assert res.gate_status == "CLEARED"


# ---------------------------------------------------------------------------
# 6. MultiFactorReasoningEngine Orchestration Tests
# ---------------------------------------------------------------------------


class TestMultiFactorReasoningEngine:
    """Verifies engine synthesis, card assembly, and confidence tiering."""

    def test_evaluate_candidate_produces_all_6_summaries(
        self, base_context: SituationalContext
    ) -> None:
        engine = MultiFactorReasoningEngine()
        card = engine.evaluate_candidate(
            base_context,
            market="SPREAD",
            side="HOME",
            line=-14.0,
            model_projected_line=-18.5,
            model_prob=0.62,
            consensus_fair_prob=0.54,
            edge_pct=0.08,
            ev=0.12,
            method_spread=0.004,
            price_american=-110,
        )

        assert card.game_id == "cfbd:2026_test_01"
        assert card.recommended_play == "HOME -14"
        assert "Tape Profile" in card.tape_summary
        assert "Position & QB" in card.position_qb_summary
        assert "Venue & Weather" in card.weather_venue_summary
        assert "Injuries & Trench" in card.injuries_trench_summary
        assert "Program & Schedule" in card.program_continuity_summary
        assert "Mathematical Edge" in card.mathematical_edge_summary
        assert card.is_favorite_vetoed is False
        assert card.tier in ("STRONG", "ELITE")

    def test_vetoed_candidate_caps_tier_at_avoid(
        self, base_context: SituationalContext
    ) -> None:
        ctx = SituationalContext(
            **{**base_context.__dict__, "qb_home_confirmed": False}
        )
        engine = MultiFactorReasoningEngine()
        card = engine.evaluate_candidate(
            ctx,
            market="SPREAD",
            side="HOME",
            line=-14.0,
            edge_pct=0.10,
        )

        assert card.is_favorite_vetoed is True
        assert card.tier == "AVOID"
        assert card.counter_thesis is not None
        assert card.confidence <= 4.5

    def test_extreme_weather_compression_outdoor(
        self, base_context: SituationalContext
    ) -> None:
        windy = WeatherProfile(
            temperature_c=10.0,
            wind_kph=38.0,
            gust_kph=50.0,
            precip_mm=4.0,
            is_dome=False,
            pass_vol_mult=0.72,
            rush_vol_mult=1.12,
            scoring_mult=0.80,
            effective_wind_kph=38.0,
        )
        ctx = SituationalContext(**{**base_context.__dict__, "weather": windy})
        engine = MultiFactorReasoningEngine()
        card = engine.evaluate_candidate(
            ctx, market="SPREAD", side="HOME", line=-20.5
        )

        assert card.is_favorite_vetoed is True
        assert "Weather compression" in str(card.counter_thesis)
        assert any(
            "Weather compression" in warning for warning in card.contra_indications
        )

    def test_dome_bypasses_weather_penalties(
        self, base_context: SituationalContext
    ) -> None:
        dome_weather = WeatherProfile(
            temperature_c=22.0,
            wind_kph=0.0,
            gust_kph=0.0,
            precip_mm=0.0,
            is_dome=True,
            pass_vol_mult=1.0,
            rush_vol_mult=1.0,
            scoring_mult=1.0,
            effective_wind_kph=0.0,
        )
        ctx = SituationalContext(**{**base_context.__dict__, "weather": dome_weather})
        engine = MultiFactorReasoningEngine()
        card = engine.evaluate_candidate(
            ctx, market="SPREAD", side="HOME", line=-20.5
        )

        assert "Indoor/Dome" in card.weather_venue_summary
        assert card.is_favorite_vetoed is False
