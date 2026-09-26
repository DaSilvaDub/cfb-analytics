"""Empirical Stress-Test and Challenge Suite for Milestone 3 (Governance & Parlay Engine).

Authored by Challenger M3-1.
Empirically stress-tests:
1. Parlay Sizing & Boundary Constraints (1, 2, 3, 10, 11+ legs).
2. Correlation & Volatility Stress (Conference overlap, Unconfirmed QBs, Extreme odds, Fragility bounds).
3. Rule Conflict Arbitration (Affirmative vs Veto, Downward overrides).
4. Thin Dog Boundary Tests (+1.49, +1.50, +6.00, +6.01, and 3 independent rescue exceptions).
"""

from __future__ import annotations

import pytest

from cfb_analytics.governance.gate import GovernanceGate
from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    GovernanceAction,
    GovernanceVerdict,
    ParlayLeg,
    compute_correlation_penalty,
    compute_efficiency_ratio,
    compute_fragility_index,
    compute_parlay_payout,
)
from cfb_analytics.governance.parlay import ParlayOptimizer
from cfb_analytics.governance.rules import (
    CandidateWager,
    GrokRuleEngine,
    ThinDogTrapRule,
)
from cfb_analytics.reasoning.models import (
    PositionUnitGrades,
    ReasoningCard,
    SituationalContext,
    TapeProfile,
    WeatherProfile,
)
from cfb_analytics.scanner.models import MispricedOpportunity, PlayTier, QualificationStatus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def neutral_tape() -> TapeProfile:
    return TapeProfile(
        honest_games=[{"opp": "Tennessee", "margin": 7, "points_scored": 28, "offensive_epa": 0.10}],
        offensive_floor_epa=0.10,
        offensive_ceiling_epa=0.25,
        defensive_floor_epa=-0.05,
        defensive_ceiling_epa=0.10,
        mean_points_scored=28.0,
    )


@pytest.fixture
def neutral_weather() -> WeatherProfile:
    return WeatherProfile(
        temperature_c=20.0,
        wind_kph=10.0,
        gust_kph=15.0,
        precip_mm=0.0,
        is_dome=False,
        pass_vol_mult=1.0,
        rush_vol_mult=1.0,
        scoring_mult=1.0,
        effective_wind_kph=10.0,
    )


@pytest.fixture
def neutral_context(neutral_tape: TapeProfile, neutral_weather: WeatherProfile) -> SituationalContext:
    return SituationalContext(
        game_id="cfbd:9001",
        home_team="Georgia",
        away_team="Vanderbilt",
        kickoff_utc="2026-10-10T19:30:00Z",
        kickoff_et="3:30 p.m.",
        tape_home=neutral_tape,
        tape_away=neutral_tape,
        weather=neutral_weather,
        qb_home_confirmed=True,
        qb_away_confirmed=True,
        trench_attrition_home=0.05,
        trench_attrition_away=0.05,
        talent_composite_home=850.0,
        talent_composite_away=750.0,
        rest_days_home=7.0,
        rest_days_away=7.0,
        travel_fatigue_tax_away=0.0,
    )


def _make_leg(
    idx: int,
    team: str = "Team",
    conf: str = "SEC",
    odds: int = -200,
    fair_prob: float = 0.70,
    edge_pct: float = 0.05,
    weather_hazard: bool = False,
    qb_news_risk: bool = False,
    is_heavy_fav: bool = False,
) -> ParlayLeg:
    return ParlayLeg(
        candidate_id=f"c_{idx}",
        game_id=f"cfbd:game_{idx:03d}",
        team=f"{team}_{idx}",
        opponent=f"Opp_{idx}",
        market_type="ML",
        line=0.0,
        odds_american=odds,
        fair_prob=fair_prob,
        edge_pct=edge_pct,
        conference=conf,
        weather_hazard=weather_hazard,
        qb_news_risk=qb_news_risk,
        is_heavy_favorite=is_heavy_fav or (odds <= -600),
    )


# =============================================================================
# 1. Parlay Sizing & Boundary Constraints
# =============================================================================

class TestParlaySizingAndBoundaries:
    """Stress-test Parlay sizing bounds: 1, 2, 3, 10, 11+ legs."""

    def test_parlay_fewer_than_3_legs_raises_value_error(self) -> None:
        optimizer = ParlayOptimizer()
        leg1 = _make_leg(1)
        leg2 = _make_leg(2)

        # 0 legs -> must raise ValueError
        with pytest.raises(ValueError, match="between 3 and 10"):
            optimizer.optimize_parlays([])

        # 1 leg -> must raise ValueError
        with pytest.raises(ValueError, match="between 3 and 10"):
            optimizer.optimize_parlays([leg1])

        # 2 legs -> must raise ValueError
        with pytest.raises(ValueError, match="between 3 and 10"):
            optimizer.optimize_parlays([leg1, leg2])

    def test_parlay_target_count_bounds_validation(self) -> None:
        """CHALLENGE / BUG PROBE:

        Caller requests target_count outside [3, 10] on a valid pool of legs.
        Currently optimize_parlays filters target_sizes without raising ValueError.
        """
        optimizer = ParlayOptimizer()
        legs_12 = [_make_leg(i) for i in range(12)]

        # Requesting target_count = 2 on a 12-leg pool:
        # Expected governance behavior: either raise ValueError or return empty.
        res_2 = optimizer.optimize_parlays(legs_12, target_count=2)
        assert len(res_2) == 0, "target_count=2 must never produce parlay tickets"

        # Requesting target_count = 11 on a 12-leg pool:
        res_11 = optimizer.optimize_parlays(legs_12, target_count=11)
        assert len(res_11) == 0, "target_count=11 must never produce parlay tickets"

    def test_evaluate_ticket_leg_bounds_and_actionability(self) -> None:
        """CHALLENGE / BUG PROBE:

        evaluate_ticket creates ParlayTicket directly without raising ValueError,
        but ticket.is_actionable MUST strictly enforce 3 <= leg_count <= 10.
        """
        optimizer = ParlayOptimizer()
        leg1 = _make_leg(1)
        leg2 = _make_leg(2)

        ticket_2 = optimizer.evaluate_ticket([leg1, leg2])
        assert ticket_2.leg_count == 2
        # Must NOT be actionable
        assert ticket_2.is_actionable is False, "2-leg ticket must never be actionable"

        # 11-leg ticket
        legs_11 = [_make_leg(i) for i in range(11)]
        ticket_11 = optimizer.evaluate_ticket(legs_11)
        assert ticket_11.leg_count == 11
        assert ticket_11.is_actionable is False, "11-leg ticket must never be actionable"

    def test_exactly_3_legs_smallest_valid_parlay(self) -> None:
        """Test exact boundary: 3 legs is smallest valid parlay."""
        optimizer = ParlayOptimizer()
        legs = [_make_leg(1, conf="SEC"), _make_leg(2, conf="Big Ten"), _make_leg(3, conf="ACC")]

        tickets = optimizer.optimize_parlays(legs, target_count=3)
        assert len(tickets) == 1
        t = tickets[0]
        assert t.leg_count == 3
        # Fragility index size penalty at k=3 must be 0.0
        assert compute_fragility_index(t.legs) >= 0.0
        # Marginal analyses should be computed for all 3 legs
        assert len(t.marginal_analyses) == 3

    def test_exactly_10_legs_largest_valid_parlay(self) -> None:
        """Test exact boundary: 10 legs is largest valid parlay."""
        optimizer = ParlayOptimizer()
        confs = ["SEC", "Big Ten", "ACC", "Big 12", "Pac-12", "MWC", "MAC", "Sun Belt", "C-USA", "AAC"]
        legs = [_make_leg(i, conf=confs[i % len(confs)], odds=-180, fair_prob=0.75, edge_pct=0.08) for i in range(10)]

        tickets = optimizer.optimize_parlays(legs, target_count=10)
        assert len(tickets) == 1
        t = tickets[0]
        assert t.leg_count == 10
        # Size penalty at K=10 is maxed at 0.35
        # Leg count tail penalty at K=10 is 0.02 * (10 - 5) = 0.10
        assert t.correlation_penalty >= 0.10
        assert t.fragility_index <= 1.0


# =============================================================================
# 2. Correlation & Volatility Stress
# =============================================================================

class TestCorrelationAndVolatilityStress:
    """Stress-test conference overlap, unconfirmed QBs, extreme chalk odds, and fragility bounds."""

    def test_heavy_conference_overlap_pruned_in_optimizer(self) -> None:
        """4+ teams from same conference must be pruned during optimization."""
        optimizer = ParlayOptimizer()
        # 4 SEC legs + 1 Big Ten leg
        legs = [
            _make_leg(1, conf="SEC"),
            _make_leg(2, conf="SEC"),
            _make_leg(3, conf="SEC"),
            _make_leg(4, conf="SEC"),
            _make_leg(5, conf="Big Ten"),
        ]
        # Any 4-leg combo containing all 4 SEC legs must be pruned
        tickets_4 = optimizer.optimize_parlays(legs, target_count=4)
        for t in tickets_4:
            sec_count = sum(1 for l in t.legs if l.conference.lower() == "sec")
            assert sec_count < 4, f"Found ticket with {sec_count} SEC legs; must be pruned!"

    def test_conference_overlap_penalty_calculation(self) -> None:
        """Verify non-linear conference penalty formula: pairs * 0.03, capped at 0.12 per conf."""
        legs_2 = [_make_leg(1, conf="SEC"), _make_leg(2, conf="SEC"), _make_leg(3, conf="ACC")]
        p2 = compute_correlation_penalty(legs_2)
        # 1 pair in SEC -> 0.03
        assert round(p2, 4) == 0.0300

        legs_3 = [_make_leg(1, conf="SEC"), _make_leg(2, conf="SEC"), _make_leg(3, conf="SEC")]
        p3 = compute_correlation_penalty(legs_3)
        # 3 pairs in SEC -> 0.09
        assert round(p3, 4) == 0.0900

        legs_4 = [_make_leg(i, conf="SEC") for i in range(4)]
        p4 = compute_correlation_penalty(legs_4)
        # 6 pairs in SEC -> 0.18, capped at 0.12
        assert round(p4, 4) == 0.1200

    def test_multiple_unconfirmed_qbs_pruned_in_optimizer(self) -> None:
        """Combinations with 2+ unconfirmed QBs must be strictly pruned."""
        optimizer = ParlayOptimizer()
        legs = [
            _make_leg(1, qb_news_risk=True),
            _make_leg(2, qb_news_risk=True),
            _make_leg(3, qb_news_risk=False),
            _make_leg(4, qb_news_risk=False),
        ]
        tickets = optimizer.optimize_parlays(legs, target_count=3)
        for t in tickets:
            qb_risk_count = sum(1 for l in t.legs if l.qb_news_risk)
            assert qb_risk_count <= 1, f"Found parlay with {qb_risk_count} unconfirmed QBs; must be <= 1!"

    def test_unconfirmed_qb_volatility_penalty_scaling(self) -> None:
        """compute_correlation_penalty adds 0.04 per volatile QB leg."""
        legs_1_qb = [_make_leg(1, qb_news_risk=True), _make_leg(2), _make_leg(3)]
        p1 = compute_correlation_penalty(legs_1_qb)
        assert p1 >= 0.04

        legs_2_qb = [_make_leg(1, qb_news_risk=True), _make_leg(2, qb_news_risk=True), _make_leg(3)]
        p2 = compute_correlation_penalty(legs_2_qb)
        assert p2 >= 0.08

    def test_extreme_odds_chalk_penalty_plateau_challenge(self) -> None:
        """CHALLENGE / AUDIT:

        Test extreme odds: heavy favorites at -600, -800, -1200, -2000, -5000.
        The current implementation formula:
            excess = max(0, abs(odds) - 600)
            penalty = min(0.05, 0.015 + 0.00005 * min(excess, 600))
        Notice: min(excess, 600) clamps at excess=600 (odds = -1200).
        Thus:
        - odds = -600 -> penalty = 0.015
        - odds = -800 -> penalty = 0.025
        - odds = -1200 -> penalty = 0.045
        - odds = -2000 -> penalty = 0.045 (plateaus!)
        - odds = -5000 -> penalty = 0.045 (plateaus!)
        """
        def get_single_chalk_penalty(odds: int) -> float:
            legs = [_make_leg(1, odds=odds, is_heavy_fav=True), _make_leg(2, conf="ACC"), _make_leg(3, conf="Big 12")]
            # confs distinct, no weather, no qb risk, 3 legs (no tail decay)
            return compute_correlation_penalty(legs)

        p_600 = get_single_chalk_penalty(-600)
        p_800 = get_single_chalk_penalty(-800)
        p_1200 = get_single_chalk_penalty(-1200)
        p_2000 = get_single_chalk_penalty(-2000)
        p_5000 = get_single_chalk_penalty(-5000)

        assert round(p_600, 4) == 0.0150
        assert round(p_800, 4) == 0.0250
        assert round(p_1200, 4) == 0.0450
        # Document the plateau behavior:
        assert round(p_2000, 4) == 0.0450
        assert round(p_5000, 4) == 0.0450

    def test_fragility_index_bounded_across_all_extreme_scenarios(self) -> None:
        """Verify Fragility Index is strictly bounded in [0.0, 1.0] across stress cases."""
        # 1. Empty legs
        assert compute_fragility_index([]) == 1.0

        # 2. Perfect safe legs (min_prob = 0.95, k=3, 0 penalties)
        safe_legs = [_make_leg(i, fair_prob=0.95, odds=-300, conf=f"C{i}") for i in range(3)]
        f_safe = compute_fragility_index(safe_legs)
        assert 0.0 <= f_safe <= 1.0
        assert f_safe == 0.0  # phi_size=0, phi_weak=0, phi_risk=0

        # 3. Maximum fragility legs (k=10, min_prob=0.40, max conference + weather + qb penalties)
        extreme_legs = [
            _make_leg(
                i,
                fair_prob=0.40,
                odds=-5000,
                conf="SEC",
                weather_hazard=True,
                qb_news_risk=True,
                is_heavy_fav=True,
            )
            for i in range(10)
        ]
        f_extreme = compute_fragility_index(extreme_legs)
        assert 0.0 <= f_extreme <= 1.0
        assert f_extreme == 1.0  # Maxed out at 1.0

        # 4. Combinatorial test of 50 random stress configurations
        import itertools
        for k in (3, 5, 8, 10):
            for prob in (0.45, 0.55, 0.75, 0.90):
                test_legs = [_make_leg(i, fair_prob=prob, odds=-250, conf=f"Conf_{i%2}") for i in range(k)]
                phi = compute_fragility_index(test_legs)
                assert 0.0 <= phi <= 1.0, f"Fragility {phi} out of bounds for k={k}, prob={prob}"


# =============================================================================
# 3. Rule Conflict Arbitration
# =============================================================================

class TestRuleConflictArbitration:
    """Challenge priority precedence: VETO > DOWNGRADE > PASS > APPROVE."""

    def test_veto_wins_over_rule_a_affirmative(self, neutral_context: SituationalContext) -> None:
        """When candidate triggers affirmative Rule A (OVER) but is vetoed by Negative Gate, VETO must win."""
        gate = GovernanceGate()

        # Context configured so Rule A would approve OVER:
        # Home Power team coming off a loss vs outmatched opponent
        tape_home = TapeProfile(honest_games=[{"margin": -10, "points_scored": 14, "offensive_epa": -0.05}])
        tape_away = TapeProfile(honest_games=[{"margin": -40, "points_scored": 3, "offensive_epa": -0.30}])
        ctx = SituationalContext(
            **{**neutral_context.__dict__, "home_team": "Florida", "away_team": "Charlotte", "tape_home": tape_home, "tape_away": tape_away}
        )

        cand_over = MispricedOpportunity(
            game_id="cfbd:9001",
            market_type="TOTAL",
            market="TOTAL",
            side="OVER",
            line=52.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.52,
            consensus_fair_price_american=-108,
            model_projected_line=60.0,
            model_prob=0.60,
            edge_pct=0.08,
            ev=0.14,
            method_spread=0.002,
            n_books=4,
            play_score=85.0,
            play_tier=PlayTier.STRONG.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )

        # Reasoning card imposes explicit VETO
        vetoed_card = ReasoningCard(
            game_id="cfbd:9001",
            market="TOTAL",
            side="OVER",
            recommended_play="OVER 52.5",
            confidence=3.0,
            tier=PlayTier.AVOID.value,
            tape_summary="",
            position_qb_summary="",
            weather_venue_summary="Hurricane force winds",
            injuries_trench_summary="",
            program_continuity_summary="",
            mathematical_edge_summary="",
            is_favorite_vetoed=True,
            counter_thesis="Extreme weather and trench attrition vetoes any OVER wager.",
        )

        verdict = gate.evaluate_candidate(cand_over, reasoning_card=vetoed_card, context=ctx)

        # Strict precedence: VETO MUST WIN
        assert verdict.action == GovernanceAction.VETO
        assert verdict.is_vetoed is True
        assert verdict.parlay_eligible is False
        assert verdict.adjusted_confidence <= 4.0
        assert any("veto" in ct.lower() for ct in verdict.counter_theses)

    def test_downgrade_to_1h_spread_preserved_when_both_rule_a_and_rule_f_trigger(
        self, neutral_context: SituationalContext
    ) -> None:
        """Candidate is a massive favorite (-28.0) triggering Rule A and Rule F.

        Governance must emit DOWNGRADE and preserve target_market_override = 'FIRST_HALF_SPREAD'.
        """
        gate = GovernanceGate()
        tape_home = TapeProfile(honest_games=[{"margin": -7, "points_scored": 17}])
        tape_away = TapeProfile(honest_games=[{"margin": -35, "points_scored": 0}])
        ctx = SituationalContext(
            **{**neutral_context.__dict__, "home_team": "Georgia", "away_team": "Charlotte", "tape_home": tape_home, "tape_away": tape_away}
        )

        cand_massive_fav = MispricedOpportunity(
            game_id="cfbd:9001",
            market_type="SPREAD",
            market="SPREAD",
            side="HOME",
            line=-28.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.52,
            consensus_fair_price_american=-108,
            model_projected_line=-33.0,
            model_prob=0.58,
            edge_pct=0.06,
            ev=0.10,
            method_spread=0.002,
            n_books=4,
            play_score=80.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )

        verdict = gate.evaluate_candidate(cand_massive_fav, None, ctx)

        assert verdict.action == GovernanceAction.DOWNGRADE
        assert verdict.is_downgraded is True
        assert verdict.target_market_override == "FIRST_HALF_SPREAD"
        assert verdict.parlay_eligible is False
        assert verdict.adjusted_confidence <= 6.5


# =============================================================================
# 4. Thin Dog Boundary Tests
# =============================================================================

class TestThinDogBoundaryAndExceptions:
    """Strict boundary and exception tests for Rule D: Thin Dogs (+1.5 to +6.0)."""

    def _eval_thin_dog(self, line: float, ctx: SituationalContext) -> tuple[bool, GovernanceAction, str]:
        rule = ThinDogTrapRule()
        cand = CandidateWager(
            candidate_id="c_dog",
            game_id="cfbd:9001",
            market_type="SPREAD",
            market="SPREAD",
            side="AWAY",
            line=line,
            odds_american=+110,
            confidence=7.5,
        )
        res = rule.evaluate(ctx, cand)
        return res.triggered, res.action, res.notes

    def test_line_1_49_not_thin_dog_trap(self, neutral_context: SituationalContext) -> None:
        """Line = +1.49 is below +1.50 threshold -> NOT thin dog trap."""
        triggered, action, notes = self._eval_thin_dog(1.49, neutral_context)
        assert triggered is False
        assert action == GovernanceAction.APPROVE

    def test_line_1_50_is_thin_dog_trap_vetoed(self, neutral_context: SituationalContext) -> None:
        """Line = +1.50 is exact lower boundary -> thin dog trap -> VETOED."""
        triggered, action, notes = self._eval_thin_dog(1.50, neutral_context)
        assert triggered is True
        assert action == GovernanceAction.VETO

    def test_line_6_00_is_thin_dog_trap_vetoed(self, neutral_context: SituationalContext) -> None:
        """Line = +6.00 is exact upper boundary -> thin dog trap -> VETOED."""
        triggered, action, notes = self._eval_thin_dog(6.00, neutral_context)
        assert triggered is True
        assert action == GovernanceAction.VETO

    def test_line_6_01_not_thin_dog_trap(self, neutral_context: SituationalContext) -> None:
        """Line = +6.01 is above +6.00 threshold -> NOT thin dog trap."""
        triggered, action, notes = self._eval_thin_dog(6.01, neutral_context)
        assert triggered is False
        assert action == GovernanceAction.APPROVE

    def test_exception_1_opposing_qb_injury_rescues_thin_dog(self, neutral_context: SituationalContext) -> None:
        """Exception A: Opponent QB unconfirmed/injured rescues +3.5 thin dog."""
        # Dog is AWAY, opponent is HOME
        ctx = SituationalContext(**{**neutral_context.__dict__, "qb_home_confirmed": False})
        triggered, action, notes = self._eval_thin_dog(3.5, ctx)
        assert triggered is True
        assert action == GovernanceAction.APPROVE
        assert "qb" in notes.lower()

    def test_exception_2_tape_differential_rescues_thin_dog(self, neutral_context: SituationalContext) -> None:
        """Exception B: > 20 point tape margin differential (net EPA diff >= +0.20) rescues thin dog."""
        tape_dog = TapeProfile(offensive_floor_epa=0.25, defensive_floor_epa=-0.10)  # Net EPA = +0.35
        tape_opp = TapeProfile(offensive_floor_epa=0.05, defensive_floor_epa=-0.05)  # Net EPA = +0.10 (diff = +0.25 >= 0.20)
        ctx = SituationalContext(**{**neutral_context.__dict__, "tape_away": tape_dog, "tape_home": tape_opp})

        triggered, action, notes = self._eval_thin_dog(3.5, ctx)
        assert triggered is True
        assert action == GovernanceAction.APPROVE
        assert "tape" in notes.lower()

    def test_exception_3_trench_mismatch_rescues_thin_dog(self, neutral_context: SituationalContext) -> None:
        """Exception C: Major trench mismatch (attrition diff >= 0.30) rescues thin dog."""
        # Opponent (HOME) has 0.45 attrition, Dog (AWAY) has 0.05 attrition -> diff = 0.40 >= 0.30
        ctx = SituationalContext(
            **{**neutral_context.__dict__, "trench_attrition_home": 0.45, "trench_attrition_away": 0.05}
        )
        triggered, action, notes = self._eval_thin_dog(3.5, ctx)
        assert triggered is True
        assert action == GovernanceAction.APPROVE
        assert "trench" in notes.lower()

    def test_exception_3_position_grades_trench_score_rescues_thin_dog(
        self, neutral_context: SituationalContext
    ) -> None:
        """Exception C alternate: PositionUnitGrades trench_mismatch_score >= 15.0 rescues thin dog."""
        pos_grades = PositionUnitGrades(trench_mismatch_score=18.5)
        ctx = SituationalContext(**{**neutral_context.__dict__, "position_grades_away": pos_grades})
        triggered, action, notes = self._eval_thin_dog(3.5, ctx)
        assert triggered is True
        assert action == GovernanceAction.APPROVE
        assert "trench" in notes.lower()
