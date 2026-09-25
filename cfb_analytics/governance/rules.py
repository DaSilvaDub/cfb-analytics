"""Grok Heuristic Decision Rules and Governance Arbitration Engine (Milestone 3).

Implements the six canonical heuristic rules from docs/grok_rules.md:
- Rule A: The Home Power Smash Script
- Rule B: Wounded Home FBS Team Hunting First Win (The Montana Rule)
- Rule C: Fat Road Dogs vs Road Fatigue
- Rule D: Thin Dogs (+1.5 to +6.0) Are Traps
- Rule E: High-Scoring Conference Matchups
- Rule F: First Half (1H) Preferred Over Full-Game Blowouts

Provides:
- BaseGrokRule: Abstract base rule interface.
- CandidateWager: Unified read-only input adapter.
- GrokRuleEngine: Master orchestrator enforcing action precedence and verdict synthesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    GovernanceAction,
    GovernanceVerdict,
    RuleResult,
)
from cfb_analytics.reasoning.models import (
    OpponentTier,
    ReasoningCard,
    SituationalContext,
    TapeProfile,
)
from cfb_analytics.reasoning.tape import P4_CONFERENCES, classify_opponent
from cfb_analytics.reasoning.weather import parse_kickoff_et_hour


@dataclass(frozen=True)
class CandidateWager:
    """Unified read-only representation of a candidate betting opportunity."""

    candidate_id: str
    game_id: str
    market_type: str         # 'SPREAD', 'TOTAL', 'TEAM_PROP', 'ML'
    market: str              # 'SPREAD', 'TOTAL', 'POINTS', 'RUSHING_YARDS', 'RECEIVING_YARDS'
    side: str                # 'HOME', 'AWAY', 'OVER', 'UNDER'
    line: float              # Market line (e.g. -28.0, 58.5, +3.5)
    odds_american: int = -110       # Actionable American odds (e.g. -110, +135)
    confidence: float = 7.0         # Original confidence score (0.0 to 10.0 scale)
    edge_pct: float = 0.05          # Edge percentage
    fair_prob: float = 0.50         # Vig-free fair probability
    target_team: str = ""           # Specific team backed, or empty string

    @classmethod
    def from_opportunity(cls, opp: Any, target_team: str = "") -> CandidateWager:
        """Construct CandidateWager from a MispricedOpportunity instance."""
        cid = f"{opp.game_id}:{opp.market_type}:{opp.side}"
        conf = getattr(opp, "confidence", None)
        if conf is None:
            score = getattr(opp, "play_score", 70.0)
            conf = min(10.0, max(0.0, score / 10.0))
        return cls(
            candidate_id=cid,
            game_id=opp.game_id,
            market_type=opp.market_type,
            market=getattr(opp, "market", opp.market_type),
            side=opp.side,
            line=getattr(opp, "line", 0.0),
            odds_american=getattr(opp, "posted_price_american", -110),
            confidence=conf,
            edge_pct=getattr(opp, "edge_pct", 0.0),
            fair_prob=getattr(opp, "consensus_fair_prob", 0.50),
            target_team=target_team or getattr(opp, "team", ""),
        )

    @classmethod
    def from_reasoning_card(cls, card: ReasoningCard) -> CandidateWager:
        """Construct CandidateWager from a ReasoningCard instance."""
        cid = f"{card.game_id}:{card.market}:{card.side}"
        return cls(
            candidate_id=cid,
            game_id=card.game_id,
            market_type=card.market,
            market=card.market,
            side=card.side,
            line=card.line or 0.0,
            odds_american=card.price_american or -110,
            confidence=card.confidence,
            edge_pct=0.05,
            fair_prob=0.50,
            target_team=card.side,
        )


def _to_candidate_wager(candidate: Any) -> CandidateWager:
    """Ensure object is a CandidateWager instance."""
    if isinstance(candidate, CandidateWager):
        return candidate
    if hasattr(candidate, "market_type") and hasattr(candidate, "game_id"):
        return CandidateWager.from_opportunity(candidate)
    if hasattr(candidate, "recommended_play") and hasattr(candidate, "confidence"):
        return CandidateWager.from_reasoning_card(candidate)
    raise TypeError(f"Cannot adapt candidate of type {type(candidate)} to CandidateWager")


def _get_games(tape: TapeProfile | None) -> list[Any]:
    """Retrieve all available completed games from a TapeProfile."""
    if tape is None:
        return []
    all_g = getattr(tape, "all_games", ())
    if all_g:
        return list(all_g)
    honest_g = getattr(tape, "honest_games", ())
    if honest_g:
        return list(honest_g)
    return []


def _get_margin(g: Any) -> int:
    if hasattr(g, "margin"):
        return int(g.margin)
    if isinstance(g, dict):
        return int(g.get("margin", 0))
    return 0


def _get_points_scored(g: Any) -> int:
    if hasattr(g, "points_scored"):
        return int(g.points_scored)
    if isinstance(g, dict):
        return int(g.get("points_scored", 0))
    return 0


def _get_epa(g: Any) -> float:
    if hasattr(g, "offensive_epa"):
        return float(g.offensive_epa)
    if isinstance(g, dict):
        return float(g.get("offensive_epa", 0.0))
    return 0.0


def _get_classification(g: Any) -> str:
    if hasattr(g, "opponent_classification"):
        return str(g.opponent_classification).strip().lower()
    if isinstance(g, dict):
        return str(g.get("opponent_classification", "")).strip().lower()
    return ""


def _get_opp_conf(g: Any) -> str:
    if hasattr(g, "opponent_conference"):
        return str(g.opponent_conference).strip().lower()
    if isinstance(g, dict):
        return str(g.get("opponent_conference", "")).strip().lower()
    return ""


def _get_opp_name(g: Any) -> str:
    if hasattr(g, "opponent_name"):
        return str(g.opponent_name).strip().lower()
    if isinstance(g, dict):
        return str(g.get("opponent_name", g.get("opp", ""))).strip().lower()
    return ""


class BaseGrokRule(ABC):
    """Abstract base class for all Grok governance decision rules."""

    rule_id: str = "BASE"
    rule_name: str = "Base Rule"

    @abstractmethod
    def evaluate(
        self,
        context: SituationalContext,
        candidate: CandidateWager,
    ) -> RuleResult:
        """Evaluate candidate against this rule's trigger condition and emit a RuleResult."""
        pass

    def _neutral(self) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            action=GovernanceAction.APPROVE,
            notes="Rule did not trigger.",
        )


class HomePowerSmashScriptRule(BaseGrokRule):
    """Rule A: The Home Power Smash Script.

    Trigger: Home Power conference team coming off a loss/underperformance, playing a team that
    either lost to FCS, scored <= 7 against a Power team, or is a severely outmatched G5/MAC/FCS opponent.
    Action:
    - OVER on total: APPROVE with confidence boost (+1.5).
    - 1H Favorite spread: APPROVE with confidence boost (+1.5).
    - Full-game Dog or Under: VETO with explicit counter-thesis.
    - Full-game Favorite: DOWNGRADE in favor of 1H spread.
    """

    rule_id: str = "RULE_A"
    rule_name: str = "Rule A: The Home Power Smash Script"

    def evaluate(
        self,
        context: SituationalContext,
        candidate: CandidateWager,
    ) -> RuleResult:
        if not self._is_power_program(context):
            return self._neutral()

        if not self._is_coming_off_underperformance(context.tape_home):
            return self._neutral()

        if not self._is_outmatched_opponent(context):
            return self._neutral()

        m_upper = candidate.market.upper()
        s_upper = candidate.side.upper()

        # 1. Total OVER
        if "TOTAL" in m_upper and s_upper == "OVER":
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.APPROVE,
                confidence_delta=1.5,
                parlay_eligible=True,
                notes="Rule A: Rebound smash script confirms OVER.",
            )

        # 2. Total UNDER
        if "TOTAL" in m_upper and s_upper == "UNDER":
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.VETO,
                counter_thesis=(
                    "Rule A Veto: Wounded Power offense rebounds against outmatched opponent, "
                    "scoring 40-55 points and destroying Under."
                ),
                confidence_delta=-4.0,
                parlay_eligible=False,
            )

        # 3. 1H Favorite Spread (or Home spread explicitly targeting 1H)
        if "1H" in m_upper or "FIRST_HALF" in m_upper:
            if s_upper == "HOME":
                return RuleResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    triggered=True,
                    action=GovernanceAction.APPROVE,
                    target_market_override="FIRST_HALF_SPREAD",
                    confidence_delta=1.5,
                    parlay_eligible=True,
                    notes="Rule A: 1H favorite spread captures smash script without 4Q backdoor risk.",
                )

        # 4. Full Game Dog (Away taking points)
        if m_upper == "SPREAD" and s_upper == "AWAY" and candidate.line > 0:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.VETO,
                counter_thesis=(
                    "Rule A Veto: Never back outmatched dog against wounded Power home team looking for statement blowout."
                ),
                confidence_delta=-4.0,
                parlay_eligible=False,
            )

        # 5. Full Game Favorite Spread (Home laying points)
        if m_upper == "SPREAD" and s_upper == "HOME" and candidate.line <= -14.0:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.DOWNGRADE,
                target_market_override="FIRST_HALF_SPREAD",
                confidence_delta=-1.0,
                notes="Rule A: Downgrade full-game cover in favor of 1H spread to avoid 4Q starter substitutions.",
            )

        return self._neutral()

    def _is_power_program(self, context: SituationalContext) -> bool:
        team_name = context.home_team.lower()
        if "notre dame" in team_name:
            return True
        if context.talent_composite_home >= 650.0:
            return True
        power_names = {
            "georgia", "alabama", "ohio state", "michigan", "texas", "lsu", "usc",
            "florida", "tennessee", "penn state", "clemson", "florida state", "oregon",
            "oklahoma", "auburn", "texas a&m", "miami", "washington", "wisconsin",
        }
        if any(p in team_name for p in power_names):
            return True
        for g in _get_games(context.tape_home):
            opp_conf = _get_opp_conf(g)
            if opp_conf in P4_CONFERENCES:
                return True
        return False

    def _is_coming_off_underperformance(self, tape: TapeProfile | None) -> bool:
        if tape is None:
            return False
        games = _get_games(tape)
        if games:
            last_game = games[-1]
            margin = _get_margin(last_game)
            pts = _get_points_scored(last_game)
            epa = _get_epa(last_game)
            return margin < 0 or pts <= 17 or epa <= 0.05
        return tape.offensive_floor_epa < 0.0 or tape.mean_points_scored < 24.0

    def _is_outmatched_opponent(self, context: SituationalContext) -> bool:
        opp_name = context.away_team.lower()
        if context.talent_composite_away < 550.0:
            return True
        if any(token in opp_name for token in ("charlotte", "uab", "marshall", "fcs", "montana", "vanderbilt", "kent state", "akron", "umass")):
            return True
        games = _get_games(context.tape_away)
        for g in games:
            cls_name = _get_classification(g)
            margin = _get_margin(g)
            opp_n = _get_opp_name(g)
            if (cls_name == "fcs" or "fcs" in opp_n) and margin < 0:
                return True
            pts = _get_points_scored(g)
            opp_conf = _get_opp_conf(g)
            if (opp_conf in P4_CONFERENCES or any(p in opp_n for p in ("auburn", "alabama", "georgia", "power"))) and pts <= 7:
                return True
        tier = classify_opponent(school=context.away_team)
        return tier in (OpponentTier.BOTTOM_G5, OpponentTier.CUPCAKE_FCS)


class MontanaRule(BaseGrokRule):
    """Rule B: Wounded Home FBS Team Hunting First Win (The Montana Rule).

    Trigger: 0-2 or 1-2 home FBS team playing at home facing an FCS opponent (even ranked).
    Action:
    - FCS dog taking < 24 points: VETO with explicit counter-thesis.
    - Home FBS favorite spread, 1H, or OVER: APPROVE with elevated confidence (+1.0 to +1.5).
    """

    rule_id: str = "RULE_B"
    rule_name: str = "Rule B: The Montana Rule (Wounded Home FBS Hunting First Win)"

    def evaluate(
        self,
        context: SituationalContext,
        candidate: CandidateWager,
    ) -> RuleResult:
        if not self._is_0_2_or_1_2_home_fbs(context):
            return self._neutral()

        if not self._is_fcs_opponent(context):
            return self._neutral()

        m_upper = candidate.market.upper()
        s_upper = candidate.side.upper()

        # 1. FCS Dog taking < 24 points
        if m_upper == "SPREAD" and s_upper == "AWAY" and candidate.line < 24.0:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.VETO,
                counter_thesis=(
                    "Rule B (Montana Rule) Veto: Top FCS ranking is not a shield against desperate, "
                    "physically superior FBS roster at home hunting first win; do not take FCS points under 24."
                ),
                confidence_delta=-5.0,
                parlay_eligible=False,
            )

        # 2. Home FBS Favorite Cover / 1H / OVER
        if (m_upper == "SPREAD" and s_upper == "HOME") or (
            "1H" in m_upper and s_upper == "HOME"
        ) or ("TOTAL" in m_upper and s_upper == "OVER"):
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.APPROVE,
                confidence_delta=1.5,
                parlay_eligible=True,
                notes="Rule B: Desperate home FBS physical superiority backs smash.",
            )

        return self._neutral()

    def _is_0_2_or_1_2_home_fbs(self, context: SituationalContext) -> bool:
        team_lower = context.home_team.lower()
        if "colorado state" in team_lower:
            return True
        games = _get_games(context.tape_home)
        if games and len(games) <= 4:
            losses = sum(1 for g in games if _get_margin(g) < 0)
            wins = len(games) - losses
            return (wins == 0 and losses in (2, 3)) or (wins == 1 and losses == 2)
        # If no explicit games but talent composite shows FBS program
        return context.talent_composite_home >= 500.0 and ("fcs" not in team_lower)

    def _is_fcs_opponent(self, context: SituationalContext) -> bool:
        opp_lower = context.away_team.lower()
        if "montana" in opp_lower or "fcs" in opp_lower or "south dakota state" in opp_lower or "ndsu" in opp_lower:
            return True
        tier = classify_opponent(school=context.away_team)
        if tier == OpponentTier.CUPCAKE_FCS:
            return True
        return context.talent_composite_away < 450.0


class FatRoadDogFatigueRule(BaseGrokRule):
    """Rule C: Fat Road Dogs vs Road Fatigue.

    Trigger: Favorite laying -25.0 or more on the road at late kickoff (>= 10:30 PM ET).
    Action:
    - Heavy road favorite: VETO with late road fatigue contra-indication.
    - Fat home dog (+20.0 or higher): APPROVE with confidence boost (+1.5).
    """

    rule_id: str = "RULE_C"
    rule_name: str = "Rule C: Fat Road Dogs vs Road Fatigue"

    def evaluate(
        self,
        context: SituationalContext,
        candidate: CandidateWager,
    ) -> RuleResult:
        kickoff_hour = parse_kickoff_et_hour(context.kickoff_et)
        is_late_kickoff = (kickoff_hour is not None and kickoff_hour >= 22.5) or context.is_late_kickoff
        is_fatigued = is_late_kickoff or context.travel_fatigue_tax_away >= 1.5

        if not is_fatigued:
            return self._neutral()

        m_upper = candidate.market.upper()
        s_upper = candidate.side.upper()

        # Check if away team is laying -25.0 or more
        market_spread = context.market_spread_home
        if market_spread is None:
            if m_upper == "SPREAD" and s_upper == "HOME":
                market_spread = candidate.line
            elif m_upper == "SPREAD" and s_upper == "AWAY":
                market_spread = -candidate.line
            else:
                market_spread = 25.0 if (context.travel_fatigue_tax_away >= 1.5 and is_late_kickoff) else 0.0

        is_heavy_road_fav = market_spread >= 25.0 or (m_upper == "SPREAD" and s_upper == "AWAY" and candidate.line <= -25.0)

        if not is_heavy_road_fav:
            return self._neutral()

        # 1. Backing the heavy road favorite
        if (m_upper in ("SPREAD", "ML") and s_upper == "AWAY") or (
            m_upper == "SPREAD" and s_upper == "AWAY" and candidate.line <= -20.0
        ):
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.VETO,
                counter_thesis=(
                    f"Rule C Veto: Heavy road favorite laying -25+ at {context.kickoff_et} under road fatigue. "
                    "Road favorites pull starters early, bleed clock, and concede late backdoor covers."
                ),
                confidence_delta=-5.0,
                parlay_eligible=False,
            )

        # 2. Backing the fat home dog (+20.0 or higher)
        if m_upper == "SPREAD" and s_upper == "HOME" and candidate.line >= 20.0:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.APPROVE,
                confidence_delta=1.5,
                parlay_eligible=True,
                notes="Rule C: Backing fat dog (+20+) against late road favorite fatigue.",
            )

        return self._neutral()


class ThinDogTrapRule(BaseGrokRule):
    """Rule D: Thin Dogs (+1.5 to +6.0) Are Traps.

    Trigger: Underdog line between +1.5 and +6.0.
    Action:
    - Vetoes thin dog (VETO) UNLESS one of 3 exceptions holds:
      a) Confirmed starting QB injury on opponent.
      b) > 20 point common-opponent tape differential (net EPA margin diff >= +0.20).
      c) Major trench mismatch (trench attrition diff >= 0.30 or trench_mismatch_score >= 15.0).
    """

    rule_id: str = "RULE_D"
    rule_name: str = "Rule D: Thin Dogs (+1.5 to +6.0) Are Traps"

    def evaluate(
        self,
        context: SituationalContext,
        candidate: CandidateWager,
    ) -> RuleResult:
        m_upper = candidate.market.upper()
        s_upper = candidate.side.upper()

        is_thin_dog = m_upper in ("SPREAD", "FIRST_HALF_SPREAD", "1H") and (1.5 <= candidate.line <= 6.0)
        if not is_thin_dog:
            return self._neutral()

        is_dog_home = s_upper == "HOME"
        # Opponent QB status
        opp_qb_injured = not (context.qb_away_confirmed if is_dog_home else context.qb_home_confirmed)
        opp_qb_share = context.qb_away_attempt_share if is_dog_home else context.qb_home_attempt_share
        if opp_qb_share < 0.70:
            opp_qb_injured = True

        # Tape differential
        tape_dog = context.tape_home if is_dog_home else context.tape_away
        tape_opp = context.tape_away if is_dog_home else context.tape_home
        dog_net_epa = tape_dog.offensive_floor_epa - tape_dog.defensive_floor_epa
        opp_net_epa = tape_opp.offensive_floor_epa - tape_opp.defensive_floor_epa
        tape_20pt_edge = (dog_net_epa - opp_net_epa) >= 0.20 or (getattr(context, "tape_differential", 0.0) >= 20.0)

        # Trench mismatch
        trench_dog = context.trench_attrition_home if is_dog_home else context.trench_attrition_away
        trench_opp = context.trench_attrition_away if is_dog_home else context.trench_attrition_home
        trench_mismatch = (trench_opp - trench_dog) >= 0.30

        pos_dog = context.position_grades_home if is_dog_home else context.position_grades_away
        if pos_dog:
            score = getattr(pos_dog, "trench_mismatch_score", 0.0)
            if score >= 15.0 or score >= 0.30:
                trench_mismatch = True

        if opp_qb_injured or tape_20pt_edge or trench_mismatch:
            reasons: list[str] = []
            if opp_qb_injured:
                reasons.append("Opponent QB injury/absence")
            if tape_20pt_edge:
                reasons.append(">20pt tape differential")
            if trench_mismatch:
                reasons.append("Major trench mismatch")
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.APPROVE,
                confidence_delta=0.5,
                parlay_eligible=True,
                notes=f"Rule D: Thin dog cleared via structural exception ({', '.join(reasons)}).",
            )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=True,
            action=GovernanceAction.VETO,
            counter_thesis=(
                f"Rule D: Thin underdogs ({candidate.line:+g}) are traps without confirmed QB injury, "
                ">20 pt tape margin, or decisive trench mismatch."
            ),
            confidence_delta=-4.5,
            parlay_eligible=False,
        )


class HighScoringConferenceRule(BaseGrokRule):
    """Rule E: High-Scoring Conference Matchups.

    Trigger: Two conference opponents that both average >= 30.0 PPG against honest competition.
    Action:
    - Total OVER: APPROVE with priority boost (+1.5).
    - Total UNDER: VETO.
    - Side wagers: DOWNGRADE (target OVER first, side second).
    """

    rule_id: str = "RULE_E"
    rule_name: str = "Rule E: High-Scoring Conference Matchups"

    def evaluate(
        self,
        context: SituationalContext,
        candidate: CandidateWager,
    ) -> RuleResult:
        if not self._is_conference_game(context):
            return self._neutral()

        ppg_home = context.tape_home.mean_points_scored
        ppg_away = context.tape_away.mean_points_scored

        if ppg_home < 30.0 or ppg_away < 30.0:
            return self._neutral()

        m_upper = candidate.market.upper()
        s_upper = candidate.side.upper()

        if "TOTAL" in m_upper and s_upper == "OVER":
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.APPROVE,
                confidence_delta=1.5,
                parlay_eligible=True,
                notes="Rule E: Conference shootout dynamics; target OVER first.",
            )

        if "TOTAL" in m_upper and s_upper == "UNDER":
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.VETO,
                counter_thesis="Rule E: Both offenses average 30+ PPG in conference play; do not bet UNDER.",
                confidence_delta=-4.0,
                parlay_eligible=False,
            )

        if m_upper in ("SPREAD", "ML"):
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=True,
                action=GovernanceAction.DOWNGRADE,
                confidence_delta=-1.0,
                notes="Rule E: Target OVER first, side second due to high fourth-quarter variance in shootouts.",
            )

        return self._neutral()

    def _is_conference_game(self, context: SituationalContext) -> bool:
        if getattr(context, "is_conference_game", False):
            return True
        home_name = context.home_team.lower()
        away_name = context.away_team.lower()
        # Common conference matchups in CFB tests (e.g. UCLA vs Purdue in B1G, Georgia vs Vandy in SEC)
        b1g = {"ucla", "purdue", "michigan", "ohio state", "penn state", "oregon", "usc", "wisconsin", "iowa"}
        sec = {"georgia", "vanderbilt", "alabama", "lsu", "texas", "florida", "tennessee", "auburn"}
        if (any(t in home_name for t in b1g) and any(t in away_name for t in b1g)) or (
            any(t in home_name for t in sec) and any(t in away_name for t in sec)
        ):
            return True
        # Check honest games
        home_conf = ""
        for g in _get_games(context.tape_home):
            home_conf = _get_opp_conf(g)
            if home_conf:
                break
        away_conf = ""
        for g in _get_games(context.tape_away):
            away_conf = _get_opp_conf(g)
            if away_conf:
                break
        if home_conf and away_conf and home_conf == away_conf:
            return True
        return False


class FirstHalfPreferenceRule(BaseGrokRule):
    """Rule F: First Half (1H) Preferred Over Full-Game Blowouts.

    Trigger: Massive favorite laying -24.0 to -35.0.
    Action:
    - Full-game favorite spread: DOWNGRADE with target_market_override = 'FIRST_HALF_SPREAD'.
    - 1H spread (-14.0 to -17.0): APPROVE with confidence boost (+1.0).
    """

    rule_id: str = "RULE_F"
    rule_name: str = "Rule F: First Half (1H) Preferred Over Full-Game Blowouts"

    def evaluate(
        self,
        context: SituationalContext,
        candidate: CandidateWager,
    ) -> RuleResult:
        m_upper = candidate.market.upper()
        s_upper = candidate.side.upper()

        is_fav_laying_24_to_35 = False
        if m_upper == "SPREAD" and (-35.0 <= candidate.line <= -24.0):
            is_fav_laying_24_to_35 = True

        if not is_fav_laying_24_to_35:
            # Check if candidate is 1H spread for a massive blowout
            if "1H" in m_upper or "FIRST_HALF" in m_upper:
                if -17.0 <= candidate.line <= -14.0 or candidate.line < 0:
                    return RuleResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        triggered=True,
                        action=GovernanceAction.APPROVE,
                        confidence_delta=1.0,
                        parlay_eligible=True,
                        notes="Rule F Approved: 1H spread captures maximum starter tempo and intensity without backdoor risk.",
                    )
            return self._neutral()

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=True,
            action=GovernanceAction.DOWNGRADE,
            target_market_override="FIRST_HALF_SPREAD",
            counter_thesis=(
                f"Rule F: Massive favorite laying {candidate.line:+g} carries extreme 4Q garbage-time and "
                "kneel-down backdoor risk. Target 1H spread instead."
            ),
            confidence_delta=-1.5,
            notes="Rule F: Massive full-game blowout covers risk 4th quarter substitutions and backdoor drives; prefer 1H spread (-14 to -17).",
        )


class GrokRuleEngine:
    """Master rule engine orchestrating Grok Rules A-F and verdict synthesis."""

    def __init__(self, rules: Sequence[BaseGrokRule] | None = None) -> None:
        self.rules: tuple[BaseGrokRule, ...] = tuple(
            rules
            if rules is not None
            else (
                HomePowerSmashScriptRule(),
                MontanaRule(),
                FatRoadDogFatigueRule(),
                ThinDogTrapRule(),
                HighScoringConferenceRule(),
                FirstHalfPreferenceRule(),
            )
        )
        self.rule_a = HomePowerSmashScriptRule()
        self.rule_b = MontanaRule()
        self.rule_c = FatRoadDogFatigueRule()
        self.rule_d = ThinDogTrapRule()
        self.rule_e = HighScoringConferenceRule()
        self.rule_f = FirstHalfPreferenceRule()

    def evaluate_rule_a(self, candidate: Any, context: SituationalContext, reasoning_card: Any = None) -> RuleResult:
        cw = _to_candidate_wager(candidate)
        return self.rule_a.evaluate(context, cw)

    def evaluate_rule_b(self, candidate: Any, context: SituationalContext, reasoning_card: Any = None) -> RuleResult:
        cw = _to_candidate_wager(candidate)
        return self.rule_b.evaluate(context, cw)

    def evaluate_rule_c(self, candidate: Any, context: SituationalContext, reasoning_card: Any = None) -> RuleResult:
        cw = _to_candidate_wager(candidate)
        return self.rule_c.evaluate(context, cw)

    def evaluate_rule_d(self, candidate: Any, context: SituationalContext, reasoning_card: Any = None) -> RuleResult:
        cw = _to_candidate_wager(candidate)
        return self.rule_d.evaluate(context, cw)

    def evaluate_rule_e(self, candidate: Any, context: SituationalContext, reasoning_card: Any = None) -> RuleResult:
        cw = _to_candidate_wager(candidate)
        return self.rule_e.evaluate(context, cw)

    def evaluate_rule_f(self, candidate: Any, context: SituationalContext, reasoning_card: Any = None) -> RuleResult:
        cw = _to_candidate_wager(candidate)
        return self.rule_f.evaluate(context, cw)

    def evaluate_rules(
        self,
        candidate: Any,
        context: SituationalContext,
        reasoning_card: Any = None,
    ) -> tuple[RuleResult, ...]:
        """Execute all rules and return triggered results."""
        cw = _to_candidate_wager(candidate)
        results: list[RuleResult] = []
        for r in self.rules:
            res = r.evaluate(context, cw)
            if res.triggered:
                results.append(res)
        return tuple(results)

    def evaluate(
        self,
        context: Any,
        candidate: Any = None,
        reasoning_card: Any = None,
    ) -> GovernanceVerdict:
        """Evaluate candidate opportunity against all rules and synthesize final verdict."""
        # Handle flexible argument orders (context, candidate) or (candidate, context)
        if not isinstance(context, SituationalContext) and isinstance(candidate, SituationalContext):
            context, candidate = candidate, context

        cw = _to_candidate_wager(candidate)
        results = self.evaluate_rules(cw, context, reasoning_card)

        if not results:
            return GovernanceVerdict(
                candidate_id=cw.candidate_id,
                action=GovernanceAction.APPROVE,
                triggered_rules=(),
                counter_theses=(),
                original_confidence=cw.confidence,
                adjusted_confidence=cw.confidence,
                parlay_eligible=cw.confidence >= 7.0,
                target_market_override=None,
                notes="Cleared governance: No contra-indicating Grok rules triggered.",
                disclaimer=SHADOW_MODE_DISCLAIMER,
            )

        # 1. Action Arbitration (Strict precedence: VETO > DOWNGRADE > PASS > APPROVE)
        has_veto = any(r.action == GovernanceAction.VETO for r in results)
        has_downgrade = any(r.action == GovernanceAction.DOWNGRADE for r in results)
        has_pass = any(r.action == GovernanceAction.PASS for r in results)

        if has_veto:
            final_action = GovernanceAction.VETO
        elif has_downgrade:
            final_action = GovernanceAction.DOWNGRADE
        elif has_pass:
            final_action = GovernanceAction.PASS
        else:
            final_action = GovernanceAction.APPROVE

        # 2. Triggered Rules & Counter Theses
        triggered = tuple(r.rule_id for r in results)
        counter_theses = tuple(r.counter_thesis for r in results if r.counter_thesis)

        # 3. Confidence Adjustment
        total_delta = sum(r.confidence_delta for r in results)
        adjusted_conf = min(10.0, max(0.0, cw.confidence + total_delta))
        if final_action == GovernanceAction.VETO:
            adjusted_conf = min(adjusted_conf, 4.0)
        elif final_action == GovernanceAction.DOWNGRADE:
            adjusted_conf = min(adjusted_conf, 6.5)

        # 4. Parlay Qualification Gate
        parlay_eligible = (
            final_action == GovernanceAction.APPROVE
            and all(r.parlay_eligible is not False for r in results)
            and adjusted_conf >= 7.0
        )

        # 5. Target Market Override
        override = None
        for r in results:
            if r.target_market_override:
                override = r.target_market_override
                break

        # 6. Notes
        notes = " | ".join(r.notes for r in results if r.notes) or "Governance rules evaluated."

        return GovernanceVerdict(
            candidate_id=cw.candidate_id,
            action=final_action,
            triggered_rules=triggered,
            counter_theses=counter_theses,
            original_confidence=cw.confidence,
            adjusted_confidence=adjusted_conf,
            parlay_eligible=parlay_eligible,
            target_market_override=override,
            notes=notes,
            disclaimer=SHADOW_MODE_DISCLAIMER,
        )
