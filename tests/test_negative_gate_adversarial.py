"""Adversarial stress-test suite for NegativeFavoriteGate and MultiFactorReasoningEngine.

Constructed by Challenger M1-2 to empirically prove critical bugs and vulnerabilities
in the Milestone 1 negative gate implementation.
"""

from __future__ import annotations

import pytest

from cfb_analytics.reasoning.engine import MultiFactorReasoningEngine
from cfb_analytics.reasoning.models import (
    SituationalContext,
    TapeProfile,
    WeatherProfile,
)
from cfb_analytics.reasoning.negative_gate import NegativeFavoriteGate


@pytest.fixture
def baseline_context() -> SituationalContext:
    tape = TapeProfile(
        honest_games=[{"opp": "Auburn", "margin": 14}],
        offensive_floor_epa=0.10,
        offensive_ceiling_epa=0.30,
        defensive_floor_epa=-0.10,
        defensive_ceiling_epa=0.05,
        cupcake_games_filtered=0,
        honest_success_rate=0.45,
        honest_yards_per_play=5.8,
    )
    weather = WeatherProfile(
        temperature_c=20.0,
        wind_kph=10.0,
        gust_kph=15.0,
        precip_mm=0.0,
        is_dome=False,
        pass_vol_mult=1.0,
        rush_vol_mult=1.0,
        scoring_mult=1.0,
    )
    return SituationalContext(
        game_id="cfbd:adversarial_test",
        home_team="Georgia",
        away_team="Vanderbilt",
        kickoff_utc="2026-10-10T19:30:00Z",
        kickoff_et="3:30 p.m.",
        tape_home=tape,
        tape_away=tape,
        weather=weather,
        qb_home_confirmed=True,
        qb_away_confirmed=True,
        qb_home_attempt_share=0.90,
        qb_away_attempt_share=0.90,
        trench_attrition_home=0.0,
        trench_attrition_away=0.0,
        talent_composite_home=950.0,
        talent_composite_away=700.0,
        rest_days_home=7.0,
        rest_days_away=7.0,
        travel_fatigue_tax_away=0.0,
    )


class TestNegativeGateAdversarialVulnerabilities:
    """Empirical proof of vulnerabilities identified during Milestone 1 challenge."""

    def test_defensive_epa_polarity_in_rule_d(self, baseline_context: SituationalContext) -> None:
        """Verify Rule D tape_edge compares net efficiency margins with correct EPA polarity."""
        gate = NegativeFavoriteGate()

        # Underdog Vanderbilt (+3.5) with mediocre offense (floor EPA = +0.05, def floor EPA = +0.05 -> net = 0.00)
        dog_tape = TapeProfile(
            honest_games=[{"opp": "LSU", "margin": -7}],
            offensive_floor_epa=0.05,
            offensive_ceiling_epa=0.15,
            defensive_floor_epa=0.05,
            defensive_ceiling_epa=0.20,
        )

        # Case 1: Georgia has an ELITE defense allowing -0.25 EPA/play (off floor = 0.20 -> net = 0.45)
        elite_def_tape = TapeProfile(
            honest_games=[{"opp": "Texas", "margin": 14}],
            offensive_floor_epa=0.20,
            offensive_ceiling_epa=0.35,
            defensive_floor_epa=-0.25,
            defensive_ceiling_epa=-0.10,
        )
        ctx_vs_elite = SituationalContext(
            **{
                **baseline_context.__dict__,
                "tape_home": elite_def_tape,  # Georgia home
                "tape_away": dog_tape,  # Vanderbilt away
            }
        )
        # Vanderbilt is +3.5 thin dog away against elite Georgia defense
        res_elite = gate.evaluate(ctx_vs_elite, market="SPREAD", side="AWAY", line=3.5)
        # dog_net (0.00) - opp_net (0.45) = -0.45 < 0.20 -> no tape edge -> VETOED
        assert res_elite.is_vetoed is True, (
            "Mediocre offense vs elite defense must NOT clear Rule D tape exception"
        )

        # Case 2: Opponent has a TERRIBLE defense allowing +0.15 EPA/play (off floor = -0.10 -> net = -0.25)
        porous_def_tape = TapeProfile(
            honest_games=[{"opp": "Akron", "margin": 3}],
            offensive_floor_epa=-0.10,
            offensive_ceiling_epa=0.20,
            defensive_floor_epa=0.15,
            defensive_ceiling_epa=0.30,
        )
        ctx_vs_porous = SituationalContext(
            **{
                **baseline_context.__dict__,
                "tape_home": porous_def_tape,
                "tape_away": dog_tape,
            }
        )
        res_porous = gate.evaluate(ctx_vs_porous, market="SPREAD", side="AWAY", line=3.5)
        # dog_net (0.00) - opp_net (-0.25) = +0.25 >= 0.20 -> tape edge -> CLEARED
        assert res_porous.is_vetoed is False, (
            "Decisive net efficiency edge vs porous defense must clear Rule D tape exception"
        )

    def test_team_prop_over_correctly_evaluates_home_team(
        self, baseline_context: SituationalContext
    ) -> None:
        """Verify TEAM_PROP OVER evaluates the home team rather than defaulting to away team."""
        # Home team (Georgia) has unconfirmed backup QB and 45% trench attrition
        # Away team (Vanderbilt) is 100% healthy
        ctx = SituationalContext(
            **{
                **baseline_context.__dict__,
                "qb_home_confirmed": False,
                "trench_attrition_home": 0.45,
                "qb_away_confirmed": True,
                "trench_attrition_away": 0.0,
            }
        )
        gate = NegativeFavoriteGate()
        # Evaluating Home team prop: OVER on points
        res = gate.evaluate(ctx, market="TEAM_PROP", side="OVER", line=42.5)

        # Verified: Home team's missing QB and trench attrition are evaluated and vetoed
        assert res.is_vetoed is True, "Home team prop OVER must evaluate and veto on home team injuries"
        assert "unconfirmed_qb_disqualification" in res.disqualifying_reasons
        assert res.gate_status == "VETOED"

    def test_sub_touchdown_favorite_with_unconfirmed_qb_is_vetoed(
        self, baseline_context: SituationalContext
    ) -> None:
        """Verify unconfirmed QB vetoes any favorite, including sub-TD spreads like -6.0."""
        ctx = SituationalContext(**{**baseline_context.__dict__, "qb_home_confirmed": False})
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-6.0)

        # Verified: Vetoed despite abs(-6.0) < 7.0
        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert res.counter_thesis is not None
        assert "unconfirmed_qb_disqualification" in res.disqualifying_reasons

        # Verify engine strictly assigns AVOID tier and caps confidence at <= 4.5
        engine = MultiFactorReasoningEngine(gate=gate)
        card = engine.evaluate_candidate(
            ctx, market="SPREAD", side="HOME", line=-6.0, edge_pct=0.08
        )
        assert card.tier == "AVOID", f"Candidate with unconfirmed QB must receive AVOID, got {card.tier}"
        assert card.confidence <= 4.5

    def test_grok_rule_c_parses_unspaced_timestamp(
        self, baseline_context: SituationalContext
    ) -> None:
        """Verify _extract_et_hour handles '10:30PM ET' (no space) correctly."""
        ctx = SituationalContext(
            **{
                **baseline_context.__dict__,
                "kickoff_et": "10:30PM ET",  # Standard book feed formatting without space
                "travel_fatigue_tax_away": 0.0,
            }
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="AWAY", line=-25.0)

        # Gate should veto: Rule C (late kickoff) + talent mismatch both fire
        assert res.is_vetoed is True
        assert "fat_road_dog_rule_c_veto" in res.disqualifying_reasons, (
            "Rule C must trigger on '10:30PM ET' format"
        )

    def test_moneyline_favorite_without_fair_prob_is_vetoed_on_unconfirmed_qb(
        self, baseline_context: SituationalContext
    ) -> None:
        """Verify ML favorites with negative American price are audited and vetoed even if fair_prob is None."""
        ctx = SituationalContext(**{**baseline_context.__dict__, "qb_home_confirmed": False})
        gate = NegativeFavoriteGate()
        res = gate.evaluate(
            ctx,
            market="ML",
            side="HOME",
            line=0.0,
            price_american=-450,
            consensus_fair_prob=None,
        )

        assert res.is_vetoed is True
        assert res.gate_status == "VETOED"
        assert "unconfirmed_qb_disqualification" in res.disqualifying_reasons

    def test_flagged_status_emits_non_empty_counter_thesis(
        self, baseline_context: SituationalContext
    ) -> None:
        """Verify FLAGGED status generates a descriptive counter-thesis string."""
        # Unsettled QB attempt share (< 0.70) creates a minor warning without critical disqualification
        ctx = SituationalContext(
            **{
                **baseline_context.__dict__,
                "qb_home_attempt_share": 0.60,
            }
        )
        gate = NegativeFavoriteGate()
        res = gate.evaluate(ctx, market="SPREAD", side="HOME", line=-14.0)

        assert res.gate_status == "FLAGGED"
        assert res.is_vetoed is False
        assert res.counter_thesis is not None
        assert "Negative gate flagged" in res.counter_thesis
        assert "Unsettled QB room" in res.counter_thesis
