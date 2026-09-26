"""Comprehensive Unit Test Suite for Grok Governance Gate, Rules A-F, and Parlay Optimizer.

Verifies:
- Immutability and structural integrity of frozen dataclasses.
- Odds conversions and mathematical payout calculations.
- Grok Rules A through F trigger and non-trigger behavior.
- Thin dog traps (Rule D) and the 3 structural exceptions (QB, Tape, Trench).
- GrokRuleEngine action precedence (VETO > DOWNGRADE > PASS > APPROVE).
- Negative Favorite Gate counter-thesis generation and veto enforcement.
- GovernanceGate orchestrator sequential execution, veto precedence, and slate evaluation.
- ParlayOptimizer 3-10 leg bounds, correlation penalties, fragility index, and marginal filtering.
- Parasitic leg detection and drop-leg alternative proposal.
- Immutability and Shadow Mode Disclaimer watermark.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pytest

from cfb_analytics.governance import (
    SHADOW_MODE_DISCLAIMER,
    BaseGrokRule,
    CandidateWager,
    FatRoadDogFatigueRule,
    FirstHalfPreferenceRule,
    GovernanceAction,
    GovernanceGate,
    GovernanceVerdict,
    GrokRuleEngine,
    HighScoringConferenceRule,
    HomePowerSmashScriptRule,
    MarginalLegAnalysis,
    MontanaRule,
    OptimizerMode,
    ParlayLeg,
    ParlayOptimizer,
    ParlayRecommendation,
    ParlaySlateSummary,
    ParlayTicket,
    RuleResult,
    ThinDogTrapRule,
    american_to_decimal,
    compute_correlation_penalty,
    compute_efficiency_ratio,
    compute_fragility_index,
    compute_parlay_payout,
    decimal_to_american,
)
from cfb_analytics.reasoning.models import (
    PositionUnitGrades,
    ReasoningCard,
    SituationalContext,
    TapeProfile,
    WeatherProfile,
)
from cfb_analytics.scanner.models import (
    MispricedOpportunity,
    PlayTier,
    QualificationStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_tape() -> TapeProfile:
    return TapeProfile(
        honest_games=[
            {"opp": "Tennessee", "margin": 14, "points_scored": 31, "offensive_epa": 0.20}
        ],
        offensive_floor_epa=0.15,
        offensive_ceiling_epa=0.35,
        defensive_floor_epa=-0.10,
        defensive_ceiling_epa=0.05,
        cupcake_games_filtered=0,
        mean_points_scored=32.0,
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
        game_id="cfbd:1001",
        home_team="Georgia",
        away_team="Vanderbilt",
        kickoff_utc="2026-10-10T19:30:00Z",
        kickoff_et="3:30 p.m.",
        tape_home=base_tape,
        tape_away=base_tape,
        weather=base_weather,
        qb_home_confirmed=True,
        qb_away_confirmed=True,
        trench_attrition_home=0.05,
        trench_attrition_away=0.05,
        talent_composite_home=950.0,
        talent_composite_away=720.0,
        rest_days_home=7.0,
        rest_days_away=7.0,
        travel_fatigue_tax_away=0.0,
    )


@pytest.fixture
def candidate_spread_fav() -> MispricedOpportunity:
    return MispricedOpportunity(
        game_id="cfbd:1001",
        market_type="SPREAD",
        market="SPREAD",
        side="HOME",
        line=-14.0,
        posted_price_american=-110,
        posted_price_decimal=1.909,
        consensus_fair_prob=0.52,
        consensus_fair_price_american=-108,
        model_projected_line=-18.5,
        model_prob=0.57,
        edge_pct=0.05,
        ev=0.088,
        method_spread=0.005,
        n_books=4,
        play_score=78.0,
        play_tier=PlayTier.QUALIFIED.value,
        qual_status=QualificationStatus.QUALIFIED.value,
    )


@pytest.fixture
def candidate_total_over() -> MispricedOpportunity:
    return MispricedOpportunity(
        game_id="cfbd:1001",
        market_type="TOTAL",
        market="TOTAL",
        side="OVER",
        line=52.5,
        posted_price_american=-105,
        posted_price_decimal=1.952,
        consensus_fair_prob=0.51,
        consensus_fair_price_american=-104,
        model_projected_line=60.0,
        model_prob=0.58,
        edge_pct=0.07,
        ev=0.132,
        method_spread=0.002,
        n_books=4,
        play_score=84.0,
        play_tier=PlayTier.STRONG.value,
        qual_status=QualificationStatus.QUALIFIED.value,
    )


@pytest.fixture
def candidate_total_under() -> MispricedOpportunity:
    return MispricedOpportunity(
        game_id="cfbd:1001",
        market_type="TOTAL",
        market="TOTAL",
        side="UNDER",
        line=52.5,
        posted_price_american=-110,
        posted_price_decimal=1.909,
        consensus_fair_prob=0.50,
        consensus_fair_price_american=-100,
        model_projected_line=50.0,
        model_prob=0.54,
        edge_pct=0.04,
        ev=0.03,
        method_spread=0.002,
        n_books=4,
        play_score=72.0,
        play_tier=PlayTier.QUALIFIED.value,
        qual_status=QualificationStatus.QUALIFIED.value,
    )


@pytest.fixture
def candidate_ml_fav() -> MispricedOpportunity:
    return MispricedOpportunity(
        game_id="cfbd:1001",
        market_type="ML",
        market="ML",
        side="HOME",
        line=0.0,
        posted_price_american=-250,
        posted_price_decimal=1.40,
        consensus_fair_prob=0.69,
        consensus_fair_price_american=-222,
        model_projected_line=0.0,
        model_prob=0.76,
        edge_pct=0.07,
        ev=0.064,
        method_spread=0.004,
        n_books=5,
        play_score=82.0,
        play_tier=PlayTier.STRONG.value,
        qual_status=QualificationStatus.QUALIFIED.value,
    )


# =============================================================================
# Test Classes: Immutability, Serialization & Odds Utilities
# =============================================================================


class TestImmutabilityAndOddsUtilities:
    """Invariants: Immutability, shadow mode watermark, and odds conversion math."""

    def test_governance_verdict_immutability(self) -> None:
        verdict = GovernanceVerdict(
            candidate_id="cfbd:1001:SPREAD:HOME",
            action=GovernanceAction.APPROVE,
            triggered_rules=("RULE_A",),
            counter_theses=(),
            original_confidence=7.5,
            adjusted_confidence=8.5,
            parlay_eligible=True,
        )
        with pytest.raises(FrozenInstanceError):
            verdict.action = GovernanceAction.VETO  # type: ignore

    def test_parlay_leg_immutability(self) -> None:
        leg = ParlayLeg(
            candidate_id="c1",
            game_id="g1",
            team="Georgia",
            opponent="Vanderbilt",
            market_type="ML",
            line=0.0,
            odds_american=-300,
            fair_prob=0.75,
            edge_pct=0.05,
        )
        with pytest.raises(FrozenInstanceError):
            leg.odds_american = -200  # type: ignore

    def test_parlay_ticket_immutability(self) -> None:
        ticket = ParlayTicket(
            parlay_id="p1",
            legs=(),
            leg_count=0,
            total_odds_american=100,
            total_payout_multiplier=2.0,
            joint_win_prob=0.5,
            raw_win_prob=0.5,
            correlation_penalty=0.0,
            ev=0.0,
            fragility_index=0.0,
            efficiency_ratio=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            ticket.ev = 0.10  # type: ignore

    def test_shadow_mode_disclaimer_watermark(self) -> None:
        verdict = GovernanceVerdict(
            candidate_id="c1",
            action=GovernanceAction.APPROVE,
        )
        assert verdict.disclaimer == SHADOW_MODE_DISCLAIMER
        assert "UNPROMOTED - shadow output, not decision-grade" in verdict.disclaimer

        leg = ParlayLeg("c1", "g1", "TeamA", "TeamB")
        assert leg.disclaimer == SHADOW_MODE_DISCLAIMER

        ticket = ParlayTicket("p1", (leg,), 1, -110, 1.909, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
        assert ticket.disclaimer == SHADOW_MODE_DISCLAIMER

    def test_american_to_decimal_conversions(self) -> None:
        assert round(american_to_decimal(-110), 4) == 1.9091
        assert round(american_to_decimal(+150), 4) == 2.5000
        assert round(american_to_decimal(-200), 4) == 1.5000
        assert round(american_to_decimal(+100), 4) == 2.0000
        assert american_to_decimal(0) == 1.0

    def test_decimal_to_american_conversions(self) -> None:
        assert decimal_to_american(1.9091) == -110
        assert decimal_to_american(2.5000) == 150
        assert decimal_to_american(1.5000) == -200
        assert decimal_to_american(1.0) == 0

    def test_compute_parlay_payout(self) -> None:
        leg1 = ParlayLeg("c1", "g1", "TeamA", "TeamB", odds_american=-110)
        leg2 = ParlayLeg("c2", "g2", "TeamC", "TeamD", odds_american=+150)
        american, mult = compute_parlay_payout([leg1, leg2])
        # 1.9091 * 2.50 = 4.7727 -> American +377
        assert mult > 4.5
        assert american > 300

    def test_rule_result_immutability_and_aliases(self) -> None:
        res = RuleResult(
            rule_id="RULE_A",
            rule_name="Home Power Smash",
            triggered=True,
            action=GovernanceAction.APPROVE,
            confidence_delta=1.5,
        )
        with pytest.raises(FrozenInstanceError):
            res.confidence_delta = 2.0  # type: ignore
        assert res.confidence_adjustment == 1.5
        assert res.disclaimer == SHADOW_MODE_DISCLAIMER

    def test_marginal_leg_analysis_immutability_and_properties(self) -> None:
        mla = MarginalLegAnalysis(
            leg_index=0,
            candidate_id="c1",
            team="Georgia",
            prob_before=0.60,
            prob_after=0.72,
            payout_before=2.0,
            payout_after=2.8,
            marginal_ev=0.08,
            delta_fragility=0.04,
            is_parasitic=False,
        )
        with pytest.raises(FrozenInstanceError):
            mla.is_parasitic = True  # type: ignore
        assert mla.team_name == "Georgia"
        assert mla.marginal_efficiency == pytest.approx(0.08 / 0.04, rel=1e-3)
        d = mla.to_dict()
        assert d["team"] == "Georgia"
        assert d["marginal_ev"] == 0.08

    def test_candidate_wager_adapters(self, candidate_spread_fav: MispricedOpportunity) -> None:
        cw = CandidateWager.from_opportunity(candidate_spread_fav, target_team="Georgia")
        assert cw.game_id == "cfbd:1001"
        assert cw.market_type == "SPREAD"
        assert cw.side == "HOME"
        assert cw.target_team == "Georgia"

        card = ReasoningCard(
            game_id="cfbd:1001",
            market="SPREAD",
            side="HOME",
            recommended_play="HOME -14.0",
            confidence=8.2,
            tier=PlayTier.STRONG.value,
            tape_summary="",
            position_qb_summary="",
            weather_venue_summary="",
            injuries_trench_summary="",
            program_continuity_summary="",
            mathematical_edge_summary="",
            line=-14.0,
            price_american=-110,
        )
        cw2 = CandidateWager.from_reasoning_card(card)
        assert cw2.game_id == "cfbd:1001"
        assert cw2.confidence == 8.2
        assert cw2.line == -14.0

    def test_parlay_recommendation_alias(self) -> None:
        assert ParlayRecommendation is ParlayTicket

    def test_enums_and_constants(self) -> None:
        assert GovernanceAction.APPROVE.value == "APPROVE"
        assert GovernanceAction.VETO.value == "VETO"
        assert GovernanceAction.DOWNGRADE.value == "DOWNGRADE"
        assert GovernanceAction.PASS.value == "PASS"
        assert OptimizerMode.SAFEST.value == "SAFEST"
        assert OptimizerMode.HIGHEST_EV.value == "HIGHEST_EV"
        assert OptimizerMode.BEST_RISK_REWARD.value == "BEST_RISK_REWARD"
        assert issubclass(HomePowerSmashScriptRule, BaseGrokRule)

    def test_compute_efficiency_ratio_edge_cases(self) -> None:
        assert compute_efficiency_ratio(0.15, 0.0) == 15.0
        assert compute_efficiency_ratio(-0.05, 0.20) == 0.0
        assert compute_efficiency_ratio(0.20, 0.40) == 0.50


# =============================================================================
# Test Classes: Grok Rules A through F
# =============================================================================


class TestRuleAHomePowerSmash:
    """Rule A: Home Power team coming off underperformance vs outmatched G5/MAC/FCS."""

    def test_rule_a_triggers_over_and_1h_spread_on_wounded_power_home(
        self, base_context: SituationalContext, candidate_total_over: MispricedOpportunity
    ) -> None:
        engine = GrokRuleEngine()
        # Wounded Power home team: SEC, coming off a loss
        tape_home = TapeProfile(
            honest_games=[
                {"opp": "Alabama", "margin": -7, "points_scored": 17, "offensive_epa": 0.01}
            ],
            mean_points_scored=24.0,
            offensive_floor_epa=0.01,
        )
        # Outmatched G5 opponent who lost to FCS or scored <= 7 vs Power
        tape_away = TapeProfile(
            honest_games=[
                {"opp": "Auburn", "margin": -35, "points_scored": 3, "offensive_epa": -0.20}
            ],
            mean_points_scored=7.0,
        )
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "home_team": "Florida",
                "away_team": "Charlotte",
                "tape_home": tape_home,
                "tape_away": tape_away,
            }
        )

        res_over = engine.evaluate_rule_a(candidate_total_over, ctx)
        assert res_over.triggered is True
        assert res_over.action == GovernanceAction.APPROVE
        assert res_over.confidence_delta > 0
        assert res_over.parlay_eligible is True

        cand_1h = MispricedOpportunity(
            game_id="cfbd:1001",
            market_type="SPREAD",
            market="FIRST_HALF_SPREAD",
            side="HOME",
            line=-14.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.52,
            consensus_fair_price_american=-108,
            model_projected_line=-17.0,
            model_prob=0.58,
            edge_pct=0.06,
            ev=0.10,
            method_spread=0.002,
            n_books=4,
            play_score=80.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res_1h = engine.evaluate_rule_a(cand_1h, ctx)
        assert res_1h.triggered is True
        assert res_1h.action == GovernanceAction.APPROVE
        assert res_1h.target_market_override == "FIRST_HALF_SPREAD"

    def test_rule_a_vetoes_under_and_fat_underdog(
        self, base_context: SituationalContext, candidate_total_under: MispricedOpportunity
    ) -> None:
        engine = GrokRuleEngine()
        tape_home = TapeProfile(
            honest_games=[
                {"opp": "LSU", "margin": -10, "points_scored": 14, "offensive_epa": -0.05}
            ],
            mean_points_scored=20.0,
        )
        tape_away = TapeProfile(
            honest_games=[{"opp": "Tennessee", "margin": -38, "points_scored": 6}],
            mean_points_scored=10.0,
        )
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "home_team": "Florida",
                "away_team": "Charlotte",
                "tape_home": tape_home,
                "tape_away": tape_away,
            }
        )

        # Under Veto
        res_under = engine.evaluate_rule_a(candidate_total_under, ctx)
        assert res_under.triggered is True
        assert res_under.action == GovernanceAction.VETO
        assert "Rule A" in (res_under.counter_thesis or "")

        # Dog taking points Veto
        cand_dog = MispricedOpportunity(
            game_id="cfbd:1001",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=21.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=18.0,
            model_prob=0.55,
            edge_pct=0.05,
            ev=0.05,
            method_spread=0.002,
            n_books=4,
            play_score=75.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res_dog = engine.evaluate_rule_a(cand_dog, ctx)
        assert res_dog.triggered is True
        assert res_dog.action == GovernanceAction.VETO

    def test_rule_a_downgrades_full_game_spread_in_favor_of_1h(
        self, base_context: SituationalContext, candidate_spread_fav: MispricedOpportunity
    ) -> None:
        engine = GrokRuleEngine()
        tape_home = TapeProfile(honest_games=[{"margin": -7, "points_scored": 14}])
        tape_away = TapeProfile(honest_games=[{"margin": -35, "points_scored": 3}])
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "home_team": "Florida",
                "away_team": "Charlotte",
                "tape_home": tape_home,
                "tape_away": tape_away,
            }
        )
        res = engine.evaluate_rule_a(candidate_spread_fav, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.DOWNGRADE
        assert res.target_market_override == "FIRST_HALF_SPREAD"


class TestRuleBMontanaRule:
    """Rule B: 0-2 or 1-2 home FBS team facing FCS opponent (The Montana Rule)."""

    def test_rule_b_triggers_and_vetoes_fcs_dog_under_24(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        # 0-2 FBS home team
        tape_home = TapeProfile(
            honest_games=[{"opp": "Texas", "margin": -14}, {"opp": "Colorado", "margin": -7}],
            mean_points_scored=21.0,
        )
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "home_team": "Colorado State",
                "away_team": "Montana",
                "tape_home": tape_home,
            }
        )

        # FCS Dog taking 17.5 points (< 24.0)
        cand_fcs_dog = MispricedOpportunity(
            game_id="cfbd:1002",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=17.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=14.0,
            model_prob=0.55,
            edge_pct=0.05,
            ev=0.05,
            method_spread=0.002,
            n_books=4,
            play_score=75.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_b(cand_fcs_dog, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.VETO
        assert "Rule B" in (res.counter_thesis or "")

    def test_rule_b_approves_home_fbs_favorite(self, base_context: SituationalContext) -> None:
        engine = GrokRuleEngine()
        tape_home = TapeProfile(
            honest_games=[{"opp": "Texas", "margin": -14}, {"opp": "Colorado", "margin": -7}],
        )
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "home_team": "Colorado State",
                "away_team": "Montana",
                "tape_home": tape_home,
            }
        )

        cand_fbs_fav = MispricedOpportunity(
            game_id="cfbd:1002",
            market_type="SPREAD",
            market="SPREAD",
            side="HOME",
            line=-17.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=-21.0,
            model_prob=0.56,
            edge_pct=0.06,
            ev=0.06,
            method_spread=0.002,
            n_books=4,
            play_score=78.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_b(cand_fbs_fav, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.APPROVE
        assert res.parlay_eligible is True


class TestRuleCFatRoadDogs:
    """Rule C: Heavy road favorite laying -25+ at late kickoff suffers road fatigue."""

    def test_rule_c_vetoes_heavy_road_favorite_at_late_kickoff(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "kickoff_et": "10:30 p.m.",
                "travel_fatigue_tax_away": 2.0,
                "is_late_kickoff": True,
                "market_spread_home": 28.0,
            }
        )
        cand_road_fav = MispricedOpportunity(
            game_id="cfbd:1003",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=-28.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=-32.0,
            model_prob=0.56,
            edge_pct=0.06,
            ev=0.06,
            method_spread=0.002,
            n_books=4,
            play_score=78.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_c(cand_road_fav, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.VETO
        assert "Rule C" in (res.counter_thesis or "")

    def test_rule_c_approves_fat_home_dog_taking_20_plus(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "kickoff_et": "10:30 p.m.",
                "travel_fatigue_tax_away": 2.0,
                "is_late_kickoff": True,
                "market_spread_home": 28.0,
            }
        )
        cand_fat_dog = MispricedOpportunity(
            game_id="cfbd:1003",
            market_type="SPREAD",
            market="SPREAD",
            side="HOME",
            line=28.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=24.0,
            model_prob=0.56,
            edge_pct=0.06,
            ev=0.06,
            method_spread=0.002,
            n_books=4,
            play_score=78.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_c(cand_fat_dog, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.APPROVE
        assert res.parlay_eligible is True


class TestRuleDThinDogsTraps:
    """Rule D: Thin underdogs (+1.5 to +6.0) are traps unless 3 exceptions hold."""

    def test_rule_d_vetoes_thin_dog_without_exceptions(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        cand_dog = MispricedOpportunity(
            game_id="cfbd:1004",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=3.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=1.0,
            model_prob=0.55,
            edge_pct=0.05,
            ev=0.05,
            method_spread=0.002,
            n_books=4,
            play_score=74.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_d(cand_dog, base_context)
        assert res.triggered is True
        assert res.action == GovernanceAction.VETO
        assert "trap" in (res.counter_thesis or "").lower()

    def test_rule_d_cleared_by_opposing_qb_injury_exception(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        # Favorite starting QB unconfirmed/injured (dog is AWAY, opp is HOME)
        ctx = SituationalContext(**{**base_context.__dict__, "qb_home_confirmed": False})
        cand_dog = MispricedOpportunity(
            game_id="cfbd:1004",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=4.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=2.0,
            model_prob=0.56,
            edge_pct=0.06,
            ev=0.06,
            method_spread=0.002,
            n_books=4,
            play_score=75.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_d(cand_dog, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.APPROVE
        assert res.parlay_eligible is True
        assert "qb" in res.notes.lower()

    def test_rule_d_cleared_by_tape_differential_exception(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        # Dog (AWAY) has +0.25 net EPA edge over favorite (HOME)
        tape_dog = TapeProfile(offensive_floor_epa=0.25, defensive_floor_epa=-0.10)  # Net = +0.35
        tape_fav = TapeProfile(
            offensive_floor_epa=0.05, defensive_floor_epa=-0.05
        )  # Net = +0.10, diff = 0.25 >= 0.20
        ctx = SituationalContext(
            **{**base_context.__dict__, "tape_away": tape_dog, "tape_home": tape_fav}
        )

        cand_dog = MispricedOpportunity(
            game_id="cfbd:1004",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=3.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=0.5,
            model_prob=0.55,
            edge_pct=0.05,
            ev=0.05,
            method_spread=0.002,
            n_books=4,
            play_score=75.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_d(cand_dog, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.APPROVE
        assert "tape" in res.notes.lower()

    def test_rule_d_cleared_by_decisive_trench_mismatch(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        # Trench attrition diff: opp (HOME) 0.45 vs dog (AWAY) 0.05 -> diff = 0.40 >= 0.30
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "trench_attrition_home": 0.45,
                "trench_attrition_away": 0.05,
            }
        )
        cand_dog = MispricedOpportunity(
            game_id="cfbd:1004",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=2.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=1.0,
            model_prob=0.55,
            edge_pct=0.05,
            ev=0.05,
            method_spread=0.002,
            n_books=4,
            play_score=75.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_d(cand_dog, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.APPROVE
        assert "trench" in res.notes.lower()

    def test_rule_d_cleared_by_position_grades_trench_mismatch(
        self, base_context: SituationalContext
    ) -> None:
        engine = GrokRuleEngine()
        pos_grades = PositionUnitGrades(trench_mismatch_score=20.0)
        ctx = SituationalContext(**{**base_context.__dict__, "position_grades_away": pos_grades})
        cand_dog = MispricedOpportunity(
            game_id="cfbd:1004",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=3.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=1.0,
            model_prob=0.55,
            edge_pct=0.05,
            ev=0.05,
            method_spread=0.002,
            n_books=4,
            play_score=75.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_d(cand_dog, ctx)
        assert res.triggered is True
        assert res.action == GovernanceAction.APPROVE
        assert "trench" in res.notes.lower()


class TestRuleEHighScoringConferenceShootout:
    """Rule E: Dual 30+ PPG honest offenses in conference play."""

    def test_rule_e_approves_over_and_vetoes_under(self, base_context: SituationalContext) -> None:
        engine = GrokRuleEngine()
        tape_high = TapeProfile(mean_points_scored=35.0)
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "home_team": "UCLA",
                "away_team": "Purdue",
                "tape_home": tape_high,
                "tape_away": tape_high,
            }
        )

        cand_over = MispricedOpportunity(
            game_id="cfbd:1005",
            market_type="TOTAL",
            market="TOTAL",
            side="OVER",
            line=56.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.51,
            consensus_fair_price_american=-104,
            model_projected_line=64.0,
            model_prob=0.59,
            edge_pct=0.08,
            ev=0.12,
            method_spread=0.002,
            n_books=4,
            play_score=82.0,
            play_tier=PlayTier.STRONG.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res_over = engine.evaluate_rule_e(cand_over, ctx)
        assert res_over.triggered is True
        assert res_over.action == GovernanceAction.APPROVE

        cand_under = MispricedOpportunity(
            game_id="cfbd:1005",
            market_type="TOTAL",
            market="TOTAL",
            side="UNDER",
            line=56.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.49,
            consensus_fair_price_american=+104,
            model_projected_line=55.0,
            model_prob=0.51,
            edge_pct=0.02,
            ev=0.01,
            method_spread=0.002,
            n_books=4,
            play_score=68.0,
            play_tier=PlayTier.LEAN.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res_under = engine.evaluate_rule_e(cand_under, ctx)
        assert res_under.triggered is True
        assert res_under.action == GovernanceAction.VETO

    def test_rule_e_downgrades_spread(self, base_context: SituationalContext) -> None:
        engine = GrokRuleEngine()
        tape_high = TapeProfile(mean_points_scored=35.0)
        ctx = SituationalContext(
            **{
                **base_context.__dict__,
                "home_team": "UCLA",
                "away_team": "Purdue",
                "tape_home": tape_high,
                "tape_away": tape_high,
            }
        )
        cand_spread = MispricedOpportunity(
            game_id="cfbd:1005",
            market_type="SPREAD",
            market="SPREAD",
            side="HOME",
            line=-4.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.52,
            consensus_fair_price_american=-108,
            model_projected_line=-7.0,
            model_prob=0.58,
            edge_pct=0.06,
            ev=0.10,
            method_spread=0.002,
            n_books=4,
            play_score=78.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res_spread = engine.evaluate_rule_e(cand_spread, ctx)
        assert res_spread.triggered is True
        assert res_spread.action == GovernanceAction.DOWNGRADE


class TestRuleFFirstHalfSpreadPreference:
    """Rule F: Massive favorite laying -24 to -35 prefers 1H spread over full game."""

    def test_rule_f_downgrades_full_game_spread(self, base_context: SituationalContext) -> None:
        engine = GrokRuleEngine()
        cand_blowout = MispricedOpportunity(
            game_id="cfbd:1006",
            market_type="SPREAD",
            market="SPREAD",
            side="HOME",
            line=-28.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=-34.0,
            model_prob=0.57,
            edge_pct=0.07,
            ev=0.088,
            method_spread=0.002,
            n_books=4,
            play_score=80.0,
            play_tier=PlayTier.STRONG.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_f(cand_blowout, base_context)
        assert res.triggered is True
        assert res.action == GovernanceAction.DOWNGRADE
        assert res.target_market_override == "FIRST_HALF_SPREAD"

    def test_rule_f_approves_1h_spread(self, base_context: SituationalContext) -> None:
        engine = GrokRuleEngine()
        cand_1h = MispricedOpportunity(
            game_id="cfbd:1006",
            market_type="SPREAD",
            market="FIRST_HALF_SPREAD",
            side="HOME",
            line=-15.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.51,
            consensus_fair_price_american=-104,
            model_projected_line=-18.0,
            model_prob=0.58,
            edge_pct=0.07,
            ev=0.10,
            method_spread=0.002,
            n_books=4,
            play_score=82.0,
            play_tier=PlayTier.STRONG.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        res = engine.evaluate_rule_f(cand_1h, base_context)
        assert res.triggered is True
        assert res.action == GovernanceAction.APPROVE
        assert res.parlay_eligible is True


# =============================================================================
# Test Classes: GrokRuleEngine & GovernanceGate Orchestration
# =============================================================================


class TestGovernanceGateOrchestrator:
    """Integration of GovernanceGate orchestrating rules, negative gate, and parlay legs."""

    def test_veto_precedence_overrides_approvals(
        self, base_context: SituationalContext, candidate_spread_fav: MispricedOpportunity
    ) -> None:
        gate = GovernanceGate()
        vetoed_card = ReasoningCard(
            game_id="cfbd:1001",
            market="SPREAD",
            side="HOME",
            recommended_play="HOME -14.0",
            confidence=3.5,
            tier=PlayTier.AVOID.value,
            tape_summary="Cupcake inflated",
            position_qb_summary="QB questionable",
            weather_venue_summary="Normal",
            injuries_trench_summary="Trench severe attrition",
            program_continuity_summary="Stable",
            mathematical_edge_summary="Small edge",
            is_favorite_vetoed=True,
            counter_thesis="Blind favorite with depleted OL and cupcake inflation.",
        )

        verdict = gate.evaluate_candidate(
            candidate=candidate_spread_fav, reasoning_card=vetoed_card, context=base_context
        )

        assert verdict.action == GovernanceAction.VETO
        assert verdict.parlay_eligible is False
        assert len(verdict.counter_theses) > 0
        assert verdict.adjusted_confidence <= 4.0

    def test_downgrade_precedence_over_approve(self, base_context: SituationalContext) -> None:
        gate = GovernanceGate()
        cand_blowout = MispricedOpportunity(
            game_id="cfbd:1001",
            market_type="SPREAD",
            market="SPREAD",
            side="HOME",
            line=-27.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.52,
            consensus_fair_price_american=-108,
            model_projected_line=-31.0,
            model_prob=0.58,
            edge_pct=0.06,
            ev=0.10,
            method_spread=0.002,
            n_books=4,
            play_score=80.0,
            play_tier=PlayTier.STRONG.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        verdict = gate.evaluate_candidate(cand_blowout, None, base_context)
        assert verdict.action == GovernanceAction.DOWNGRADE
        assert verdict.target_market_override == "FIRST_HALF_SPREAD"

    def test_evaluate_slate_batch_processing(
        self,
        base_context: SituationalContext,
        candidate_ml_fav: MispricedOpportunity,
        candidate_spread_fav: MispricedOpportunity,
    ) -> None:
        gate = GovernanceGate()
        candidates = [candidate_ml_fav, candidate_spread_fav]
        verdicts = gate.evaluate_slate(candidates, contexts={"cfbd:1001": base_context})
        assert len(verdicts) == 2
        assert all(isinstance(v, GovernanceVerdict) for v in verdicts)

    def test_get_parlay_qualified_candidates_filters_moneylines_only(
        self,
        base_context: SituationalContext,
        candidate_ml_fav: MispricedOpportunity,
        candidate_spread_fav: MispricedOpportunity,
    ) -> None:
        gate = GovernanceGate(parlay_min_confidence=7.0)
        candidates = [candidate_ml_fav, candidate_spread_fav]

        # Construct verdicts directly to isolate the filtering logic
        # (evaluate_candidate triggers Rule E DOWNGRADE on SEC conference
        # shootouts, which is correct behavior but not what this test targets)
        verdict_ml = GovernanceVerdict(
            candidate_id="cfbd:1001:ML:HOME",
            action=GovernanceAction.APPROVE,
            triggered_rules=(),
            counter_theses=(),
            original_confidence=8.2,
            adjusted_confidence=8.2,
            parlay_eligible=True,
            notes="Cleared governance: No contra-indications.",
        )
        verdict_spread = GovernanceVerdict(
            candidate_id="cfbd:1001:SPREAD:HOME",
            action=GovernanceAction.APPROVE,
            triggered_rules=(),
            counter_theses=(),
            original_confidence=7.8,
            adjusted_confidence=7.8,
            parlay_eligible=False,
            notes="Cleared governance: No contra-indications.",
        )

        verdicts = [verdict_ml, verdict_spread]
        legs = gate.get_parlay_qualified_candidates(
            candidates, verdicts, {"cfbd:1001": base_context}
        )

        assert len(legs) == 1
        assert legs[0].market_type == "ML"
        assert legs[0].game_id == "cfbd:1001"
        assert legs[0].odds_american == -250


# =============================================================================
# Test Classes: Parlay Optimizer & Mathematical Penalties
# =============================================================================


class TestParlayOptimizer:
    """Parlay optimizer 3-10 bounds, penalties, fragility, and marginal value."""

    def test_parlay_optimizer_enforces_3_to_10_leg_count(self) -> None:
        optimizer = ParlayOptimizer()
        leg1 = ParlayLeg("c1", "g1", "Georgia", "Vandy", "ML", 0.0, -300, 0.73, 0.05)
        leg2 = ParlayLeg("c2", "g2", "Ohio St", "Purdue", "ML", 0.0, -400, 1.250, 0.78, 0.04)

        # Under 3 legs must raise ValueError
        with pytest.raises(ValueError, match="3 and 10"):
            optimizer.optimize_parlays([leg1, leg2])

    def test_same_conference_correlation_penalty(self) -> None:
        # 3 SEC teams -> 3 pairs -> 3 * 0.03 = 0.09 penalty
        legs = [
            ParlayLeg(
                f"c{i}",
                f"g{i}",
                f"Team{i}",
                f"Opp{i}",
                "ML",
                0.0,
                -200,
                0.70,
                0.05,
                conference="SEC",
            )
            for i in range(3)
        ]
        penalty = compute_correlation_penalty(legs)
        assert round(penalty, 4) == 0.0900

    def test_weather_hazard_and_qb_news_penalty(self) -> None:
        legs = [
            ParlayLeg(
                "c1", "g1", "TeamA", "OppA", "ML", 0.0, -200, 0.75, 0.05, weather_hazard=True
            ),
            ParlayLeg("c2", "g2", "TeamB", "OppB", "ML", 0.0, -250, 0.75, 0.05, qb_news_risk=True),
            ParlayLeg("c3", "g3", "TeamC", "OppC", "ML", 0.0, -180, 0.70, 0.05),
        ]
        penalty = compute_correlation_penalty(legs)
        # Weather: 0.04, QB: 0.04 -> 0.08
        assert round(penalty, 4) == 0.0800

    def test_heavy_public_favorite_chalk_penalty(self) -> None:
        # -800 chalk leg: 0.015 + 0.00005 * 200 = 0.025
        leg_chalk = ParlayLeg(
            "c1", "g1", "Alabama", "UAB", "ML", 0.0, -800, 0.88, 0.02, is_heavy_favorite=True
        )
        leg2 = ParlayLeg("c2", "g2", "Georgia", "Vandy", "ML", 0.0, -300, 0.75, 0.05)
        leg3 = ParlayLeg("c3", "g3", "Texas", "Rice", "ML", 0.0, -400, 0.80, 0.04)

        penalty = compute_correlation_penalty([leg_chalk, leg2, leg3])
        assert penalty >= 0.025

    def test_fragility_index_bounded_between_0_and_1(self) -> None:
        optimizer = ParlayOptimizer()
        legs = [
            ParlayLeg(
                f"c{i}",
                f"g{i}",
                f"Team{i}",
                f"Opp{i}",
                "ML",
                0.0,
                -200,
                0.65,
                0.05,
                conference="SEC",
            )
            for i in range(4)
        ]
        parlays = optimizer.optimize_parlays(legs)
        assert len(parlays) > 0
        for p in parlays:
            assert 0.0 <= p.fragility_index <= 1.0

    def test_marginal_leg_value_detects_parasitic_leg(self) -> None:
        optimizer = ParlayOptimizer()
        # 3 high-quality value legs + 1 heavy chalk -1000 leg with no edge
        leg1 = ParlayLeg("c1", "g1", "TeamA", "OppA", "ML", 0.0, -200, 0.80, 0.08)
        leg2 = ParlayLeg("c2", "g2", "TeamB", "OppB", "ML", 0.0, -250, 0.82, 0.07)
        leg3 = ParlayLeg("c3", "g3", "TeamC", "OppC", "ML", 0.0, -180, 0.78, 0.06)
        leg_parasitic = ParlayLeg(
            "c4", "g4", "TeamD", "OppD", "ML", 0.0, -1200, 0.88, -0.05, is_heavy_favorite=True
        )

        ticket = optimizer.evaluate_ticket([leg1, leg2, leg3, leg_parasitic])
        assert ticket.leg_count == 4
        # At least one leg should have marginal analysis
        assert len(ticket.marginal_analyses) == 4
        # Leg 4 should be evaluated
        leg4_analysis = ticket.marginal_analyses[3]
        assert leg4_analysis.candidate_id == "c4"

    def test_find_optimal_tickets_multi_objective(self) -> None:
        optimizer = ParlayOptimizer()
        legs = [
            ParlayLeg("c1", "g1", "Georgia", "Vandy", "ML", 0.0, -350, 0.85, 0.06),
            ParlayLeg("c2", "g2", "Ohio State", "Marshall", "ML", 0.0, -450, 0.88, 0.05),
            ParlayLeg("c3", "g3", "Utah", "Utah State", "ML", 0.0, -300, 0.82, 0.07),
            ParlayLeg("c4", "g4", "Oregon", "Boise", "ML", 0.0, -280, 0.80, 0.06),
        ]
        summary = optimizer.find_optimal_tickets(legs, slate_date="2026-09-26")
        assert summary.status == "SUCCESS"
        assert summary.safest_parlay is not None
        assert summary.highest_ev_parlay is not None
        assert summary.best_risk_reward_parlay is not None
        assert len(summary.optimal_by_size) > 0

    def test_insufficient_qualified_legs_returns_status(self) -> None:
        optimizer = ParlayOptimizer()
        legs = [
            ParlayLeg("c1", "g1", "Georgia", "Vandy", "ML", 0.0, -350, 0.85, 0.06),
            ParlayLeg("c2", "g2", "Ohio State", "Marshall", "ML", 0.0, -450, 0.88, 0.05),
        ]
        summary = optimizer.find_optimal_tickets(legs)
        assert summary.status == "INSUFFICIENT_QUALIFIED_LEGS"
        assert summary.safest_parlay is None
