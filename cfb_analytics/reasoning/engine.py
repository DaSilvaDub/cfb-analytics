"""Multi-Factor Contextual Game Reasoning Engine for NCAA College Football.

Synthesizes tape, weather, injuries, roster talent, program structure, and
mathematical edge into structured, auditable ReasoningCard records with
negative gate interlocks.
"""

from __future__ import annotations

from cfb_analytics.reasoning.models import (
    NegativeGateResult,
    ReasoningCard,
    SituationalContext,
)
from cfb_analytics.reasoning.negative_gate import NegativeFavoriteGate


class MultiFactorReasoningEngine:
    """Core reasoning engine orchestrating all 6 analytical dimensions."""

    def __init__(self, gate: NegativeFavoriteGate | None = None) -> None:
        self._gate = gate or NegativeFavoriteGate()

    def evaluate_candidate(
        self,
        context: SituationalContext,
        *,
        market: str,
        side: str,
        line: float,
        model_projected_line: float | None = None,
        model_prob: float | None = None,
        consensus_fair_prob: float | None = None,
        edge_pct: float | None = None,
        ev: float | None = None,
        method_spread: float | None = None,
        price_american: int | None = None,
    ) -> ReasoningCard:
        """Synthesize all dimensions and assemble an auditable ReasoningCard."""
        # 1. Negative Favorite Gate evaluation
        gate_result = self._gate.evaluate(
            context,
            market=market,
            side=side,
            line=line,
            model_projected_line=model_projected_line,
            model_prob=model_prob,
            consensus_fair_prob=consensus_fair_prob,
            price_american=price_american,
        )

        # 2. Build 6 Dimension Summaries
        tape_summary = self._build_tape_summary(context, side)
        position_qb_summary = self._build_position_qb_summary(context, side)
        weather_venue_summary = self._build_weather_venue_summary(context)
        injuries_trench_summary = self._build_injuries_trench_summary(context, side)
        program_continuity_summary = self._build_program_continuity_summary(context, side)
        mathematical_edge_summary = self._build_mathematical_edge_summary(
            market=market,
            side=side,
            line=line,
            model_projected_line=model_projected_line,
            model_prob=model_prob,
            consensus_fair_prob=consensus_fair_prob,
            edge_pct=edge_pct,
            ev=ev,
            method_spread=method_spread,
            price_american=price_american,
        )

        # 3. Compile all contra-indications
        contra_indications = list(gate_result.contra_indications)

        # 4. Compute epistemic confidence (0.0 to 10.0 scale)
        confidence = self._compute_confidence(
            context=context,
            side=side,
            edge_pct=edge_pct,
            gate_result=gate_result,
            n_contra=len(contra_indications),
        )
        if gate_result.is_vetoed:
            confidence = min(confidence, 4.5)

        # 5. Determine Play Recommendation Label & Tier
        play_label = self._format_play_label(market, side, line)
        tier = self._assign_tier(confidence, gate_result.is_vetoed)

        return ReasoningCard(
            game_id=context.game_id,
            market=market,
            side=side,
            recommended_play=play_label,
            confidence=confidence,
            tier=tier,
            tape_summary=tape_summary,
            position_qb_summary=position_qb_summary,
            weather_venue_summary=weather_venue_summary,
            injuries_trench_summary=injuries_trench_summary,
            program_continuity_summary=program_continuity_summary,
            mathematical_edge_summary=mathematical_edge_summary,
            contra_indications=contra_indications,
            is_favorite_vetoed=gate_result.is_vetoed,
            counter_thesis=gate_result.counter_thesis,
            matchup_label=f"{context.away_team} at {context.home_team}",
            kickoff_et=context.kickoff_et,
            line=line,
            price_american=price_american,
        )

    # -------------------------------------------------------------------------
    # Dimension Summarizers
    # -------------------------------------------------------------------------

    def _build_tape_summary(self, context: SituationalContext, side: str) -> str:
        is_home = side == "HOME" or context.home_team in side
        tape = context.tape_home if is_home else context.tape_away
        opp_tape = context.tape_away if is_home else context.tape_home
        team_name = context.home_team if is_home else context.away_team

        n_honest = len(tape.honest_games)
        n_cupcakes = tape.cupcake_games_filtered

        return (
            f"Tape Profile ({team_name}): {n_honest} honest FBS games analyzed "
            f"({n_cupcakes} cupcake blowouts stripped). Offensive EPA: floor {tape.offensive_floor_epa:+.2f}, "
            f"ceiling {tape.offensive_ceiling_epa:+.2f} (success rate {tape.honest_success_rate:.1%}). "
            f"Defensive EPA allowed: floor {tape.defensive_floor_epa:+.2f}, ceiling {tape.defensive_ceiling_epa:+.2f}. "
            f"Opponent honest defensive floor EPA: {opp_tape.defensive_floor_epa:+.2f}."
        )

    def _build_position_qb_summary(self, context: SituationalContext, side: str) -> str:
        is_home = side == "HOME" or context.home_team in side
        qb_confirmed = context.qb_home_confirmed if is_home else context.qb_away_confirmed
        qb_name = context.qb_home_name if is_home else context.qb_away_name
        qb_share = context.qb_home_attempt_share if is_home else context.qb_away_attempt_share
        team_name = context.home_team if is_home else context.away_team

        status_str = "Confirmed" if qb_confirmed else "UNRESOLVED/BACKUP"
        qb_display = qb_name or "Presumptive Starter"

        return (
            f"Position & QB ({team_name}): Starting QB {status_str} ({qb_display}, "
            f"{qb_share:.1%} pass attempt share). Trench line push supported by starting "
            f"personnel continuity."
        )

    def _build_weather_venue_summary(self, context: SituationalContext) -> str:
        w = context.weather
        if w.is_dome:
            return (
                f"Venue & Weather: Indoor/Dome stadium. Climate controlled; "
                f"atmospheric multipliers neutral (pass 1.00x, rush 1.00x, scoring 1.00x). "
                f"Elevation: {context.venue_elevation_m:.0f}m."
            )

        temp_f = (w.temperature_c * 9.0 / 5.0) + 32.0
        eff_wind = w.effective_wind_kph if w.effective_wind_kph > 0 else w.wind_kph
        return (
            f"Venue & Weather: Kickoff {temp_f:.1f}°F ({w.temperature_c:.1f}°C), outdoor. "
            f"Wind: {w.wind_kph:.1f} kph sustained (gusts {w.gust_kph:.1f} kph, effective {eff_wind:.1f} kph). "
            f"Precipitation: {w.precip_mm:.1f} mm. Multipliers: pass {w.pass_vol_mult:.2f}x, "
            f"rush {w.rush_vol_mult:.2f}x, scoring {w.scoring_mult:.2f}x. "
            f"Away travel fatigue tax: {context.travel_fatigue_tax_away:.1f} pts."
        )

    def _build_injuries_trench_summary(self, context: SituationalContext, side: str) -> str:
        is_home = side == "HOME" or context.home_team in side
        trench_self = context.trench_attrition_home if is_home else context.trench_attrition_away
        trench_opp = context.trench_attrition_away if is_home else context.trench_attrition_home
        team_self = context.home_team if is_home else context.away_team
        team_opp = context.away_team if is_home else context.home_team

        return (
            f"Injuries & Trench Health: {team_self} starting front attrition {trench_self:.0%}; "
            f"{team_opp} front attrition {trench_opp:.0%}. Net trench differential: "
            f"{(trench_opp - trench_self):+.0%} in favor of {team_self}."
        )

    def _build_program_continuity_summary(self, context: SituationalContext, side: str) -> str:
        is_home = side == "HOME" or context.home_team in side
        rest_self = context.rest_days_home if is_home else context.rest_days_away
        rest_opp = context.rest_days_away if is_home else context.rest_days_home
        continuity = (
            context.coaching_continuity_home if is_home else context.coaching_continuity_away
        )
        lookahead = context.lookahead_flag_home if is_home else context.lookahead_flag_away
        team_name = context.home_team if is_home else context.away_team

        lookahead_str = (
            "WARNING: Marquee matchup on deck next week" if lookahead else "Clean situational spot"
        )
        return (
            f"Program & Schedule ({team_name}): Staff continuity rating {continuity:.1f}/1.0. "
            f"Rest schedule: {rest_self:.0f} days rest vs opponent {rest_opp:.0f} days rest. "
            f"Situational spot: {lookahead_str}."
        )

    def _build_mathematical_edge_summary(
        self,
        *,
        market: str,
        side: str,
        line: float,
        model_projected_line: float | None,
        model_prob: float | None,
        consensus_fair_prob: float | None,
        edge_pct: float | None,
        ev: float | None,
        method_spread: float | None,
        price_american: int | None,
    ) -> str:
        price_str = f"{price_american:+d}" if price_american is not None else "-110"
        proj_str = f"{model_projected_line:+.1f}" if model_projected_line is not None else "N/A"
        prob_str = f"{model_prob:.1%}" if model_prob is not None else "N/A"
        fair_str = f"{consensus_fair_prob:.1%}" if consensus_fair_prob is not None else "N/A"
        edge_str = f"{edge_pct:+.1%}" if edge_pct is not None else "+0.0%"
        ev_str = f"{ev:+.1%}" if ev is not None else "+0.0%"
        spread_str = f"{method_spread * 100:.1f} pp" if method_spread is not None else "0.0 pp"

        return (
            f"Mathematical Edge vs Consensus: Market {market} {side} {line:+g} ({price_str}). "
            f"Model projected line: {proj_str} (Model Prob: {prob_str}). "
            f"Consensus vig-free fair probability: {fair_str} (Method Spread: {spread_str}). "
            f"Quantified edge: {edge_str}, Expected Value (EV): {ev_str}."
        )

    # -------------------------------------------------------------------------
    # Calibration & Formatting
    # -------------------------------------------------------------------------

    def _compute_confidence(
        self,
        context: SituationalContext,
        side: str,
        edge_pct: float | None,
        gate_result: NegativeGateResult,
        n_contra: int,
    ) -> float:
        """Compute calibrated epistemic confidence score on 0.0 to 10.0 scale."""
        edge = edge_pct or 0.0
        # Base confidence from edge
        if edge >= 0.06:
            base = 8.5
        elif edge >= 0.03:
            base = 7.5
        elif edge >= 0.01:
            base = 6.5
        elif edge > 0.0:
            base = 5.5
        else:
            base = 3.5

        # Affirmative situational adjustments
        is_home = side == "HOME" or context.home_team in side
        tape = context.tape_home if is_home else context.tape_away
        trench = context.trench_attrition_home if is_home else context.trench_attrition_away
        qb_confirmed = context.qb_home_confirmed if is_home else context.qb_away_confirmed

        bonus = 0.0
        if tape.offensive_floor_epa >= 0.15:
            bonus += 0.5
        if trench == 0.0:
            bonus += 0.5
        attempt_share = context.qb_home_attempt_share if is_home else context.qb_away_attempt_share
        if qb_confirmed and attempt_share >= 0.85:
            bonus += 0.5

        # Contra penalties
        penalty = gate_result.confidence_penalty + (0.5 * n_contra)
        score = base + min(1.5, bonus) - penalty
        # Hard cap: vetoed candidates never exceed 4.0 confidence
        if gate_result.is_vetoed:
            score = min(score, 4.0)
        return round(max(0.1, min(9.9, score)), 1)

    def _assign_tier(self, confidence: float, is_vetoed: bool) -> str:
        if is_vetoed:
            return "AVOID"
        if confidence >= 9.0:
            return "ELITE"
        if confidence >= 8.0:
            return "STRONG"
        if confidence >= 7.0:
            return "QUALIFIED"
        if confidence >= 6.0:
            return "LEAN"
        if confidence >= 4.0:
            return "PASS"
        return "AVOID"

    def _format_play_label(self, market: str, side: str, line: float) -> str:
        if market == "SPREAD":
            return f"{side} {line:+g}"
        if market in ("TOTAL", "POINTS"):
            return f"{side} {line:.1f}"
        if market == "ML":
            return f"{side} ML"
        return f"{market} {side} {line:.1f}"
