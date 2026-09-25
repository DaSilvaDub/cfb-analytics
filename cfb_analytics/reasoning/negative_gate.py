"""Negative gate against blind favorites and situational contradiction detector.

Implements the 'No Blind Favorites' governing philosophy:
Never endorse a favorite based purely on market price, program brand, or AP ranking
without affirmative verification across honest tape, trench health, weather,
roster talent composite, and program schedule context.

Also enforces:
- Grok Rule C: Fatigue risk for heavy road favorites (-20+ laying points at >=10:30 PM ET).
- Grok Rule D: Thin dogs (+1.5 to +6.0) are traps unless structural exceptions are satisfied.
"""

from __future__ import annotations

from cfb_analytics.reasoning.models import NegativeGateResult, SituationalContext
from cfb_analytics.reasoning.weather import parse_kickoff_et_hour

# Rule C thresholds
FAT_ROAD_DOG_MIN_SPREAD: float = 20.0  # Favorite laying -20.0 or more
LATE_KICKOFF_HOUR_ET: float = 22.5  # 10:30 PM ET or later
ROAD_FATIGUE_TAX_THRESHOLD: float = 1.5  # Significant travel penalty

# Weather compression thresholds
WIND_COMPRESSION_KPH: float = 30.0
WIND_GUST_COMPRESSION_KPH: float = 45.0
PRECIP_COMPRESSION_MM: float = 2.5
SCORING_MULT_COMPRESSION: float = 0.85

# Trench & QB thresholds
TRENCH_ATTRITION_SEVERE: float = 0.35
TRENCH_ATTRITION_NET_DEFICIT: float = 0.25
QB_SHARE_MIN_STABLE: float = 0.70

# Rest & Lookahead thresholds
REST_DAYS_BYE: float = 12.0
REST_DAYS_SHORT: float = 6.5
TALENT_POINT_RATIO: float = 0.055  # Spread points per talent composite point diff


class NegativeFavoriteGate:
    """Evaluates candidate recommendations against the 5 situational dimensions."""

    def evaluate(
        self,
        context: SituationalContext,
        *,
        market: str,
        side: str,
        line: float,
        model_projected_line: float | None = None,
        model_prob: float | None = None,
        consensus_fair_prob: float | None = None,
        price_american: int | None = None,
        team_id: str | None = None,
    ) -> NegativeGateResult:
        """Evaluate candidate pick for blind favorite risks or situational contradictions."""
        is_fav, fav_team = self._is_favorite_play(
            context,
            market,
            side,
            line,
            consensus_fair_prob,
            price_american=price_american,
            team_id=team_id,
        )

        # Check thin underdog trap (Rule D) if play is on a thin dog
        if self._is_thin_underdog_play(context, market, side, line):
            return self._evaluate_thin_dog_trap(context, side, line)

        if not is_fav:
            return NegativeGateResult(
                is_vetoed=False,
                gate_status="CLEARED",
                counter_thesis=None,
                contra_indications=(),
                disqualifying_reasons=(),
                confidence_penalty=0.0,
            )

        # Candidate is a favorite: execute 5-dimensional audit
        critical_contradictions: list[str] = []
        minor_contra_indications: list[str] = []
        theses_parts: list[str] = []

        # 1. Tape Dimension: Cupcake inflation & Honest EPA
        tape_veto, tape_thesis, tape_warnings = self._audit_tape_dimension(context, fav_team, line)
        if tape_veto:
            critical_contradictions.append(tape_veto)
            theses_parts.append(tape_thesis)
        minor_contra_indications.extend(tape_warnings)

        # 2. Weather & Venue Dimension: Road fatigue (Rule C) & Weather compression
        weather_veto, weather_thesis, weather_warnings = self._audit_weather_venue_dimension(
            context, fav_team, line
        )
        if weather_veto:
            critical_contradictions.append(weather_veto)
            theses_parts.append(weather_thesis)
        minor_contra_indications.extend(weather_warnings)

        # 3. Injuries & Trench Health Dimension: QB stability & Trench attrition
        trench_criticals, trench_theses, trench_warnings = self._audit_injuries_trench_dimension(
            context, fav_team, line
        )
        critical_contradictions.extend(trench_criticals)
        theses_parts.extend(trench_theses)
        minor_contra_indications.extend(trench_warnings)

        # 4. Roster Talent Dimension: Brand name premium vs Talent Composite
        talent_veto, talent_thesis, talent_warnings = self._audit_roster_talent_dimension(
            context, fav_team, line
        )
        if talent_veto:
            critical_contradictions.append(talent_veto)
            theses_parts.append(talent_thesis)
        minor_contra_indications.extend(talent_warnings)

        # 5. Program Structure & Schedule Dimension: Rest disparity & Lookahead trap
        schedule_veto, schedule_thesis, schedule_warnings = self._audit_schedule_spot_dimension(
            context, fav_team, line
        )
        if schedule_veto:
            critical_contradictions.append(schedule_veto)
            theses_parts.append(schedule_thesis)
        minor_contra_indications.extend(schedule_warnings)

        # Final arbitration: VETOED vs FLAGGED vs CLEARED
        if critical_contradictions or len(minor_contra_indications) >= 3:
            combined_thesis = (
                " ".join(theses_parts)
                if theses_parts
                else (
                    f"Negative gate vetoed {fav_team} ({line:+g}): Multiple situational contradictions "
                    f"({len(minor_contra_indications)}) undermine the market line."
                )
            )
            penalty = 5.0 if len(critical_contradictions) >= 2 else 4.0
            return NegativeGateResult(
                is_vetoed=True,
                gate_status="VETOED",
                counter_thesis=combined_thesis,
                contra_indications=tuple(minor_contra_indications),
                disqualifying_reasons=tuple(critical_contradictions),
                confidence_penalty=penalty,
            )

        if minor_contra_indications:
            flagged_thesis = (
                f"Negative gate flagged {fav_team} ({line:+g}) with situational concerns: "
                + "; ".join(minor_contra_indications)
            )
            return NegativeGateResult(
                is_vetoed=False,
                gate_status="FLAGGED",
                counter_thesis=flagged_thesis,
                contra_indications=tuple(minor_contra_indications),
                disqualifying_reasons=(),
                confidence_penalty=1.5 * len(minor_contra_indications),
            )

        return NegativeGateResult(
            is_vetoed=False,
            gate_status="CLEARED",
            counter_thesis=None,
            contra_indications=(),
            disqualifying_reasons=(),
            confidence_penalty=0.0,
        )

    # -------------------------------------------------------------------------
    # Helper & Dimensional Audit Methods
    # -------------------------------------------------------------------------

    def _is_favorite_play(
        self,
        context: SituationalContext,
        market: str,
        side: str,
        line: float,
        fair_prob: float | None,
        price_american: int | None = None,
        team_id: str | None = None,
    ) -> tuple[bool, str]:
        """Determine if this candidate play backs a favorite."""
        m_upper = market.upper()
        s_upper = side.upper()

        if m_upper == "SPREAD":
            if s_upper == "HOME" and line < 0:
                return True, context.home_team
            if s_upper == "AWAY" and line < 0:
                return True, context.away_team
        elif m_upper == "ML":
            # Moneyline favorites: if price_american < 0 or consensus_fair_prob > 0.50 (or line < 0)
            is_fav_price = (price_american is not None and price_american < 0) or (line < 0)
            is_fav_prob = (fair_prob is not None and fair_prob > 0.50)
            if is_fav_price or is_fav_prob:
                if s_upper == "HOME":
                    return True, context.home_team
                if s_upper == "AWAY":
                    return True, context.away_team
        elif (
            m_upper == "TEAM_PROP"
            or "TEAM" in m_upper
            or m_upper in ("POINTS", "RUSHING_YARDS", "RECEIVING_YARDS", "OFFENSIVE_YARDS")
        ):
            # Team Prop Attribution: inspect market string or team_id for home vs away attribution
            if team_id:
                is_home = (team_id == context.home_team or team_id == context.home_team_id)
            elif context.away_team.lower() in market.lower() or "away" in market.lower():
                is_home = False
            elif context.home_team.lower() in market.lower() or "home" in market.lower():
                is_home = True
            else:
                # Default to home team when attributing generic team props unless away is indicated
                is_home = (s_upper in ("HOME", "OVER"))
            return True, context.home_team if is_home else context.away_team

        return False, ""

    def _is_thin_underdog_play(
        self,
        context: SituationalContext,
        market: str,
        side: str,
        line: float,
    ) -> bool:
        """Check if candidate is backing a thin underdog (+1.5 to +6.0)."""
        if market == "SPREAD":
            if (side in ("HOME", "AWAY")) and 1.5 <= line <= 6.0:
                return True
        return False

    def _evaluate_thin_dog_trap(
        self,
        context: SituationalContext,
        side: str,
        line: float,
    ) -> NegativeGateResult:
        """Enforce Grok Rule D: Thin dogs are traps unless specific criteria met."""
        # Check exception criteria:
        # 1. Opposing QB injury/unconfirmed
        # 2. Honest EPA differential > 20 points equivalent (~0.20 net EPA)
        # 3. Decisive trench mismatch in favor of dog
        is_dog_home = side == "HOME"
        qb_opp_missing = not (
            context.qb_away_confirmed if is_dog_home else context.qb_home_confirmed
        )
        tape_opp = context.tape_away if is_dog_home else context.tape_home
        tape_dog = context.tape_home if is_dog_home else context.tape_away

        # Compare net efficiency margins: defensive_floor_epa is EPA allowed per play (negative is elite).
        # Net efficiency margin = offensive floor EPA minus defensive floor EPA allowed.
        dog_net = tape_dog.offensive_floor_epa - tape_dog.defensive_floor_epa
        opp_net = tape_opp.offensive_floor_epa - tape_opp.defensive_floor_epa
        tape_edge = (dog_net - opp_net) >= 0.20
        trench_dog = context.trench_attrition_home if is_dog_home else context.trench_attrition_away
        trench_opp = context.trench_attrition_away if is_dog_home else context.trench_attrition_home
        trench_edge = (trench_opp - trench_dog) >= 0.30

        if qb_opp_missing or tape_edge or trench_edge:
            return NegativeGateResult(
                is_vetoed=False,
                gate_status="CLEARED",
                counter_thesis=None,
                contra_indications=("Thin dog qualified via Rule D exception criteria",),
                disqualifying_reasons=(),
                confidence_penalty=0.0,
            )

        return NegativeGateResult(
            is_vetoed=True,
            gate_status="VETOED",
            counter_thesis=(
                f"Thin underdog ({line:+g}) vetoed under Rule D: Thin dogs fail at extreme rates "
                f"without an opposing QB injury, decisive tape margin, or dominant trench mismatch."
            ),
            contra_indications=(
                "Rule D: Thin dogs are traps without affirmative structural mismatch",
            ),
            disqualifying_reasons=("thin_dog_trap_rule_d",),
            confidence_penalty=4.5,
        )

    def _audit_tape_dimension(
        self,
        context: SituationalContext,
        fav_team: str,
        line: float,
    ) -> tuple[str | None, str, list[str]]:
        is_home = fav_team == context.home_team
        tape = context.tape_home if is_home else context.tape_away
        opp_tape = context.tape_away if is_home else context.tape_home
        warnings: list[str] = []

        # Check cupcake inflation
        if tape.cupcake_games_filtered > 0:
            warnings.append(
                f"Cupcake inflation: {tape.cupcake_games_filtered} junk game(s) stripped from baseline"
            )
            if tape.offensive_floor_epa < -0.02 and abs(line) >= 10.0:
                thesis = (
                    f"Favorite {fav_team} line ({line:+g}) is inflated by cupcake blowout tape. "
                    f"Against honest FBS competition, offensive floor EPA drops to "
                    f"{tape.offensive_floor_epa:+.2f}, failing to demonstrate multi-score cover separation."
                )
                return "cupcake_tape_mirage", thesis, warnings

        # Stagnation against capable defense: offensive floor EPA suppressed by opponent defensive floor EPA allowed
        if (tape.offensive_floor_epa + opp_tape.defensive_floor_epa) < 0.0 and abs(line) >= 14.0:
            warnings.append(
                f"Tape mismatch: Offensive floor EPA ({tape.offensive_floor_epa:+.2f}) "
                f"trails opponent defensive resistance ({opp_tape.defensive_floor_epa:+.2f} allowed)"
            )

        return None, "", warnings

    def _audit_weather_venue_dimension(
        self,
        context: SituationalContext,
        fav_team: str,
        line: float,
    ) -> tuple[str | None, str, list[str]]:
        warnings: list[str] = []
        is_away = fav_team == context.away_team

        # Grok Rule C: Fat Road Dogs vs Road Fatigue
        if is_away and abs(line) >= FAT_ROAD_DOG_MIN_SPREAD:
            kickoff_hour = self._extract_et_hour(context.kickoff_et)
            is_late_kickoff = kickoff_hour is not None and kickoff_hour >= LATE_KICKOFF_HOUR_ET
            has_fatigue_tax = context.travel_fatigue_tax_away >= ROAD_FATIGUE_TAX_THRESHOLD

            if is_late_kickoff or has_fatigue_tax:
                thesis = (
                    f"Grok Rule C Veto: Heavy road favorite {fav_team} laying {line:+g} at "
                    f"{context.kickoff_et} under road fatigue tax ({context.travel_fatigue_tax_away:.1f} pts). "
                    f"Late-night cross-country favorites pull starters early, bleed clock, and "
                    f"concede fourth-quarter backdoor covers."
                )
                warnings.append(
                    f"Rule C road fatigue: Late kickoff {context.kickoff_et} with travel tax"
                )
                return "fat_road_dog_rule_c_veto", thesis, warnings

        # Atmospheric variance compression
        if not context.weather.is_dome:
            w = context.weather
            is_windy = w.wind_kph >= WIND_COMPRESSION_KPH or w.gust_kph >= WIND_GUST_COMPRESSION_KPH
            is_wet = w.precip_mm >= PRECIP_COMPRESSION_MM
            is_low_scoring = w.scoring_mult <= SCORING_MULT_COMPRESSION

            if is_windy:
                warnings.append(
                    f"Weather compression: Sustained wind {w.wind_kph:.1f} kph (gusts {w.gust_kph:.1f} kph)"
                )
            if is_wet:
                warnings.append(f"Weather compression: Precipitation {w.precip_mm:.1f} mm")

            if (is_windy or is_wet or is_low_scoring) and abs(line) >= 17.5:
                thesis = (
                    f"Weather compression veto: Adverse conditions ({w.wind_kph:.0f} kph wind, "
                    f"scoring multiplier {w.scoring_mult:.2f}x) compress total possessions, "
                    f"forcing ground-bound clock bleed and eliminating the scoring margin required for {line:+g}."
                )
                return "weather_possession_compression", thesis, warnings

        return None, "", warnings

    def _audit_injuries_trench_dimension(
        self,
        context: SituationalContext,
        fav_team: str,
        line: float,
    ) -> tuple[list[str], list[str], list[str]]:
        """Returns (criticals, theses, warnings) — checks both OL and DL injuries."""
        is_home = fav_team == context.home_team
        qb_confirmed = context.qb_home_confirmed if is_home else context.qb_away_confirmed
        qb_share = context.qb_home_attempt_share if is_home else context.qb_away_attempt_share
        trench_fav = context.trench_attrition_home if is_home else context.trench_attrition_away
        trench_opp = context.trench_attrition_away if is_home else context.trench_attrition_home
        ol_out = context.ol_starters_out_home if is_home else context.ol_starters_out_away
        dl_out = context.dl_starters_out_home if is_home else context.dl_starters_out_away
        criticals: list[str] = []
        theses: list[str] = []
        warnings: list[str] = []

        # QB Uncertainty — any favorite with unconfirmed QB1 must be vetoed
        if not qb_confirmed:
            warnings.append("Quarterback uncertainty: Starting QB is unconfirmed")
            criticals.append("unconfirmed_qb_disqualification")
            line_str = f" ({line:+g})" if line != 0 else ""
            theses.append(
                f"Unconfirmed quarterback status for {fav_team}: Endorsing a favorite{line_str} "
                f"without confirmed QB1 availability introduces unacceptable structural variance."
            )

        if qb_share < QB_SHARE_MIN_STABLE:
            warnings.append(f"Unsettled QB room: Starter attempt share is only {qb_share:.1%}")

        # Trench Attrition — check both OL and DL injuries
        if trench_fav >= TRENCH_ATTRITION_SEVERE:
            warnings.append(
                f"Severe trench attrition: {trench_fav:.0%} of starting front unavailable"
            )
            if abs(line) >= 10.0 and "severe_trench_attrition" not in criticals:
                criticals.append("severe_trench_attrition")
                theses.append(
                    f"Critical trench attrition ({trench_fav:.0%}) for {fav_team}: Severe OL/DL personnel losses "
                    f"compromise line-of-scrimmage push, elevating sack risk and stalling drives."
                )

        if ol_out >= 2 or dl_out >= 2:
            warnings.append(f"Compound trench attrition: {ol_out} OL and {dl_out} DL starters unavailable")
            if abs(line) >= 10.0 and "severe_trench_attrition" not in criticals:
                criticals.append("severe_trench_attrition")
                theses.append(
                    f"Compound trench attrition ({ol_out} OL, {dl_out} DL out) for {fav_team}: "
                    f"Multiple line-of-scrimmage starters unavailable, degrading pass protection and run fits."
                )
        elif ol_out > 0:
            warnings.append(f"Offensive line attrition: {ol_out} starting OL unavailable")
        elif dl_out > 0:
            warnings.append(f"Defensive line attrition: {dl_out} starting DL unavailable")

        if (trench_fav - trench_opp) >= TRENCH_ATTRITION_NET_DEFICIT:
            warnings.append(
                f"Net trench deficit: {fav_team} attrition ({trench_fav:.0%}) vs "
                f"opponent ({trench_opp:.0%})"
            )

        return criticals, theses, warnings

    def _audit_roster_talent_dimension(
        self,
        context: SituationalContext,
        fav_team: str,
        line: float,
    ) -> tuple[str | None, str, list[str]]:
        is_home = fav_team == context.home_team
        talent_fav = context.talent_composite_home if is_home else context.talent_composite_away
        talent_opp = context.talent_composite_away if is_home else context.talent_composite_home
        talent_delta = talent_fav - talent_opp
        warnings: list[str] = []

        # Check if spread is radically larger than talent supports
        expected_spread_from_talent = talent_delta * TALENT_POINT_RATIO
        spread_gap = abs(line) - expected_spread_from_talent

        if spread_gap >= 12.0 and talent_delta < 50.0:
            warnings.append(
                f"Brand name inflation: Talent delta ({talent_delta:+.1f}) does not support {line:+g} spread"
            )
            if abs(line) >= 14.0:
                thesis = (
                    f"Brand-name premium trap: {fav_team} carries public prestige pricing ({line:+g}), "
                    f"but roster talent composite differential ({talent_delta:+.1f}) shows near-parity "
                    f"in baseline physical caliber."
                )
                return "brand_name_talent_mismatch", thesis, warnings

        return None, "", warnings

    def _audit_schedule_spot_dimension(
        self,
        context: SituationalContext,
        fav_team: str,
        line: float,
    ) -> tuple[str | None, str, list[str]]:
        is_home = fav_team == context.home_team
        rest_fav = context.rest_days_home if is_home else context.rest_days_away
        rest_opp = context.rest_days_away if is_home else context.rest_days_home
        lookahead = context.lookahead_flag_home if is_home else context.lookahead_flag_away
        warnings: list[str] = []

        # Rest Disparity
        has_rest_deficit = (rest_opp >= REST_DAYS_BYE) and (rest_fav <= REST_DAYS_SHORT)
        if has_rest_deficit:
            warnings.append(
                f"Rest disparity: Opponent off bye ({rest_opp:.0f}d) vs favorite short turnaround ({rest_fav:.0f}d)"
            )

        # Lookahead Spot
        if lookahead:
            warnings.append("Lookahead trap: Marquee conference/rivalry matchup on deck next week")

        # Compound trap
        if has_rest_deficit and lookahead and abs(line) >= 10.0:
            thesis = (
                f"Classic trap spot: {fav_team} faces a major rest deficit ({rest_fav:.0f}d vs {rest_opp:.0f}d) "
                f"combined with a lookahead distraction ahead of a marquee game, making {line:+g} highly fragile."
            )
            return "compound_rest_lookahead_trap", thesis, warnings

        return None, "", warnings

    def _extract_et_hour(self, kickoff_et: str) -> float | None:
        """Parse string kickoff time into float hours using weather module parser."""
        return parse_kickoff_et_hour(kickoff_et)
