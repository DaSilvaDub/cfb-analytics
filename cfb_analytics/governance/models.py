"""Data models and immutable contracts for Grok Decision Governance (Milestone 3).

Provides frozen dataclasses, action enums, and mathematical formulas for:
1. Governance Verdicts: Pre-bet gating, downgrading, vetoing, and target market overrides.
2. Parlay Leg Records: Qualified individual wagers admitting conference, weather, and QB risk attributes.
3. Parlay Recommendations: Optimized multi-leg wagers with correlation penalties, payout multipliers,
   and fragility index quantification.

All models strictly enforce:
- Pure Python standard library primitives (no pandas, numpy, or external AI).
- Absolute immutability via `@dataclass(frozen=True)` and tuple sequences.
- Mandatory Shadow Mode disclaimer watermark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

# Mandatory shadow mode disclaimer for unpromoted research output
SHADOW_MODE_DISCLAIMER: str = "UNPROMOTED - shadow output, not decision-grade"


class GovernanceAction(str, Enum):
    """Actionable decision emitted by the Grok Governance Engine."""

    APPROVE = "APPROVE"      # Candidate wager passes all gates and is endorsed for execution/parlays
    DOWNGRADE = "DOWNGRADE"  # Candidate wager value is reduced; recommended stake cut or market redirect
    VETO = "VETO"            # Candidate wager is strictly forbidden due to fatal situational contradictions
    PASS = "PASS"            # Candidate wager lacks sufficient edge, fails structural test, or is a trap


@dataclass(frozen=True)
class RuleResult:
    """Atomic evaluation output emitted by an individual Grok rule."""

    rule_id: str
    rule_name: str
    triggered: bool
    action: GovernanceAction = GovernanceAction.APPROVE
    counter_thesis: str | None = None
    confidence_delta: float = 0.0
    parlay_eligible: bool | None = None
    target_market_override: str | None = None
    notes: str = ""
    disclaimer: str = SHADOW_MODE_DISCLAIMER

    @property
    def confidence_adjustment(self) -> float:
        """Alias for confidence_delta for backwards/cross-module compatibility."""
        return self.confidence_delta


@dataclass(frozen=True)
class GovernanceVerdict:
    """Authoritative gating decision emitted for a single candidate betting opportunity.

    Attributes:
        candidate_id: Unique deterministic identifier for candidate (e.g., 'cfbd:401520182:SPREAD:AWAY').
        action: Final governance action (APPROVE, DOWNGRADE, VETO, PASS).
        triggered_rules: Immutable sequence of rule identifiers that fired (e.g., ('RULE_A', 'RULE_F')).
        counter_theses: Concrete explanations of situational risks or contradictions identified.
        original_confidence: Confidence score before governance evaluation (scale 0.0 to 10.0).
        adjusted_confidence: Confidence score after rule adjustments and penalties (scale 0.0 to 10.0).
        parlay_eligible: Whether candidate qualifies for inclusion in multi-leg parlay construction.
        target_market_override: Redirected market recommendation (e.g., 'FIRST_HALF_SPREAD'), or None.
        notes: Synthesized analytical notes describing the governance rationale.
        disclaimer: Mandatory shadow mode disclaimer.
    """

    candidate_id: str
    action: GovernanceAction
    triggered_rules: tuple[str, ...] = ()
    counter_theses: tuple[str, ...] = ()
    original_confidence: float = 0.0
    adjusted_confidence: float = 0.0
    parlay_eligible: bool = False
    target_market_override: str | None = None
    notes: str = ""
    disclaimer: str = SHADOW_MODE_DISCLAIMER

    @property
    def is_approved(self) -> bool:
        """True if the candidate was unconditionally approved."""
        return self.action == GovernanceAction.APPROVE

    @property
    def is_vetoed(self) -> bool:
        """True if the candidate was vetoed."""
        return self.action == GovernanceAction.VETO

    @property
    def is_downgraded(self) -> bool:
        """True if the candidate was downgraded."""
        return self.action == GovernanceAction.DOWNGRADE

    @property
    def is_pass(self) -> bool:
        """True if the candidate was designated as a pass."""
        return self.action == GovernanceAction.PASS

    def to_dict(self) -> dict[str, Any]:
        """Serialize verdict to a standard JSON-compatible dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "action": self.action.value if isinstance(self.action, GovernanceAction) else str(self.action),
            "triggered_rules": list(self.triggered_rules),
            "counter_theses": list(self.counter_theses),
            "original_confidence": round(self.original_confidence, 2),
            "adjusted_confidence": round(self.adjusted_confidence, 2),
            "parlay_eligible": self.parlay_eligible,
            "target_market_override": self.target_market_override,
            "notes": self.notes,
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class ParlayLeg:
    """Individual qualified wager admitted for multi-leg parlay construction.

    Attributes:
        candidate_id: Unique candidate identifier.
        game_id: Canonical CFBD game identifier ('cfbd:<id>').
        team: Team being backed in this leg.
        opponent: Opposing team.
        market_type: High-level market family (strictly 'ML' for moneyline parlays).
        line: Handicap line or total points (0.0 for ML).
        odds_american: American odds price for this specific leg (e.g. -110, +135).
        fair_prob: Vig-free consensus or model fair win probability (0.0 to 1.0).
        edge_pct: Mathematical edge percentage (model_prob - consensus_fair_prob).
        conference: Primary conference of the backed team (e.g. 'SEC', 'Big Ten').
        weather_hazard: True if game is exposed to high wind (>= 18 mph / 29 kph) or precip (>= 2.5 mm).
        qb_news_risk: True if starting QB is unconfirmed or returning from injury.
        is_heavy_favorite: True if odds_american <= -600 (chalk trap risk).
        disclaimer: Mandatory shadow mode disclaimer.
    """

    candidate_id: str
    game_id: str
    team: str
    opponent: str
    market_type: str = "ML"
    line: float = 0.0
    odds_american: int = -110
    fair_prob: float = 0.50
    edge_pct: float = 0.0
    conference: str = ""
    weather_hazard: bool = False
    qb_news_risk: bool = False
    is_heavy_favorite: bool = False
    disclaimer: str = SHADOW_MODE_DISCLAIMER

    def __post_init__(self) -> None:
        if self.odds_american <= -600 and not self.is_heavy_favorite:
            # Enforce is_heavy_favorite invariant
            object.__setattr__(self, "is_heavy_favorite", True)

    @property
    def decimal_odds(self) -> float:
        """Convert American odds to standard European decimal odds multiplier."""
        return american_to_decimal(self.odds_american)

    @property
    def odds_decimal(self) -> float:
        """Alias for decimal_odds."""
        return self.decimal_odds

    def to_dict(self) -> dict[str, Any]:
        """Serialize parlay leg to a dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "team": self.team,
            "opponent": self.opponent,
            "market_type": self.market_type,
            "line": self.line,
            "odds_american": self.odds_american,
            "decimal_odds": round(self.decimal_odds, 4),
            "fair_prob": round(self.fair_prob, 4),
            "edge_pct": round(self.edge_pct, 4),
            "conference": self.conference,
            "weather_hazard": self.weather_hazard,
            "qb_news_risk": self.qb_news_risk,
            "is_heavy_favorite": self.is_heavy_favorite,
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class MarginalLegAnalysis:
    """Audit metrics for a single leg within a specific multi-leg parlay ticket.

    Attributes:
        leg_index: 0-indexed position of leg in parlay.
        candidate_id: Candidate identifier.
        team: Team backed.
        prob_before: Joint win probability without this leg.
        prob_after: Joint win probability with this leg included.
        payout_before: Payout multiplier without this leg.
        payout_after: Payout multiplier with this leg included.
        marginal_ev: Marginal change in expected value (EV_with - EV_without).
        delta_fragility: Marginal change in fragility index (Phi_with - Phi_without).
        is_parasitic: True if adding this leg degrades or fails to improve expected value.
    """

    leg_index: int
    candidate_id: str
    team: str
    prob_before: float
    prob_after: float
    payout_before: float
    payout_after: float
    marginal_ev: float
    delta_fragility: float
    is_parasitic: bool = False

    @property
    def team_name(self) -> str:
        """Alias for team."""
        return self.team

    @property
    def marginal_efficiency(self) -> float:
        """Marginal EV generated per unit of additional fragility."""
        return self.marginal_ev / max(0.001, self.delta_fragility)

    def to_dict(self) -> dict[str, Any]:
        """Serialize marginal leg analysis to dictionary."""
        return {
            "leg_index": self.leg_index,
            "candidate_id": self.candidate_id,
            "team": self.team,
            "prob_before": round(self.prob_before, 4),
            "prob_after": round(self.prob_after, 4),
            "payout_before": round(self.payout_before, 4),
            "payout_after": round(self.payout_after, 4),
            "marginal_ev": round(self.marginal_ev, 4),
            "delta_fragility": round(self.delta_fragility, 4),
            "marginal_efficiency": round(self.marginal_efficiency, 4),
            "is_parasitic": self.is_parasitic,
        }


@dataclass(frozen=True)
class ParlayTicket:
    """Optimized multi-leg parlay construction with correlation adjustments and fragility metrics.

    Attributes:
        parlay_id: Unique deterministic parlay identifier (e.g. 'parlay:2026-09-26:3leg:01').
        legs: Immutable tuple of individual component ParlayLeg instances.
        leg_count: Number of legs in this parlay (strictly between 3 and 10).
        total_odds_american: Combined American sportsbook odds (e.g. +595).
        total_payout_multiplier: Decimal payout multiplier including original stake (e.g. 6.95).
        joint_win_prob: True joint win probability after correlation/risk penalty discount.
        raw_win_prob: Raw independent joint win probability (product of fair_prob_i).
        correlation_penalty: Probability discount due to intra-slate shared variance and risk.
        ev: Expected value: (joint_win_prob * total_payout_multiplier) - 1.0.
        fragility_index: Normalized 0.0 to 1.0 metric of vulnerability to weakest leg failure.
        efficiency_ratio: EV / fragility_index (risk/reward density).
        marginal_analyses: Tuple of MarginalLegAnalysis records per leg.
        alternate_pruned_ticket: Optional sub-ticket with parasitic leg dropped.
        disclaimer: Mandatory shadow mode disclaimer.
    """

    parlay_id: str
    legs: tuple[ParlayLeg, ...]
    leg_count: int
    total_odds_american: int
    total_payout_multiplier: float
    joint_win_prob: float
    raw_win_prob: float
    correlation_penalty: float
    ev: float
    fragility_index: float
    efficiency_ratio: float
    marginal_analyses: tuple[MarginalLegAnalysis, ...] = field(default_factory=tuple)
    alternate_pruned_ticket: ParlayTicket | None = None
    disclaimer: str = SHADOW_MODE_DISCLAIMER

    @property
    def adjusted_prob(self) -> float:
        """Alias for joint_win_prob for backwards/cross-module compatibility."""
        return self.joint_win_prob

    @property
    def weakest_leg(self) -> ParlayLeg | None:
        """Identifies the single leg with the lowest individual fair probability."""
        if not self.legs:
            return None
        return min(self.legs, key=lambda leg: leg.fair_prob)

    @property
    def weakest_leg_name(self) -> str:
        """Name of team in weakest leg."""
        wl = self.weakest_leg
        return wl.team if wl else ""

    @property
    def weakest_leg_prob(self) -> float:
        """Probability of weakest leg."""
        wl = self.weakest_leg
        return wl.fair_prob if wl else 0.0

    @property
    def has_parasitic_leg(self) -> bool:
        """True if any marginal analysis flagged a parasitic leg."""
        return any(m.is_parasitic for m in self.marginal_analyses)

    @property
    def is_actionable(self) -> bool:
        """True if the parlay delivers positive EV and clears safety thresholds."""
        return (
            3 <= self.leg_count <= 10
            and self.ev > 0.0
            and self.joint_win_prob > 0.05
            and self.fragility_index < 0.65
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize parlay recommendation to a dictionary."""
        weakest = self.weakest_leg
        return {
            "parlay_id": self.parlay_id,
            "leg_count": self.leg_count,
            "total_odds_american": self.total_odds_american,
            "total_payout_multiplier": round(self.total_payout_multiplier, 4),
            "joint_win_prob": round(self.joint_win_prob, 4),
            "raw_win_prob": round(self.raw_win_prob, 4),
            "correlation_penalty": round(self.correlation_penalty, 4),
            "adjusted_prob": round(self.joint_win_prob, 4),
            "ev": round(self.ev, 4),
            "fragility_index": round(self.fragility_index, 4),
            "efficiency_ratio": round(self.efficiency_ratio, 4),
            "weakest_leg_team": weakest.team if weakest else None,
            "weakest_leg_prob": round(weakest.fair_prob, 4) if weakest else None,
            "has_parasitic_leg": self.has_parasitic_leg,
            "legs": [leg.to_dict() for leg in self.legs],
            "marginal_analyses": [m.to_dict() for m in self.marginal_analyses],
            "alternate_pruned_ticket": self.alternate_pruned_ticket.to_dict() if self.alternate_pruned_ticket else None,
            "disclaimer": self.disclaimer,
        }


# Alias ParlayRecommendation to ParlayTicket for cross-module compatibility
ParlayRecommendation = ParlayTicket


@dataclass(frozen=True)
class ParlaySlateSummary:
    """Summary of parlay optimization search across an entire slate."""

    slate_date: str
    candidates_evaluated: int
    qualified_legs_count: int
    total_combinations_screened: int
    safest_parlay: ParlayTicket | None
    highest_ev_parlay: ParlayTicket | None
    best_risk_reward_parlay: ParlayTicket | None
    optimal_by_size: tuple[ParlayTicket, ...] = ()
    unjustified_sizes: tuple[int, ...] = ()
    status: str = "SUCCESS"  # SUCCESS, INSUFFICIENT_QUALIFIED_LEGS, NO_POSITIVE_EV
    disclaimer: str = SHADOW_MODE_DISCLAIMER


# =============================================================================
# Mathematical Odds & Conversion Utilities
# =============================================================================

def american_to_decimal(american: int) -> float:
    """Convert American odds to standard European decimal odds multiplier.

    Examples:
        -110 -> 1.9091
        +150 -> 2.5000
        -200 -> 1.5000
    """
    if american == 0:
        return 1.0
    if american > 0:
        return 1.0 + (american / 100.0)
    return 1.0 + (100.0 / abs(american))


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal payout multiplier to standard American odds integer.

    Examples:
        1.9091 -> -110
        2.5000 -> +150
        1.5000 -> -200
    """
    if decimal_odds <= 1.0:
        return 0
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def compute_parlay_payout(legs: Sequence[ParlayLeg]) -> tuple[int, float]:
    """Calculate combined decimal payout multiplier and American odds for a series of legs.

    Returns:
        tuple[int, float]: (total_odds_american, total_payout_multiplier)
    """
    if not legs:
        return 0, 1.0
    mult = 1.0
    for leg in legs:
        mult *= leg.decimal_odds
    mult = round(mult, 4)
    american = decimal_to_american(mult)
    return american, mult


def compute_correlation_penalty(legs: Sequence[ParlayLeg]) -> float:
    """Quantify correlation and risk discount across multiple parlay legs.

    Evaluates:
    1. Same-Conference Overlap: pairs(c) * 0.03 per duplicate leg in the same conference (capped at 0.12).
    2. Weather Hazard Overlap: High wind (>= 18 mph / 29 kph) or precip (>= 2.5 mm) adds 0.04 per leg (capped at 0.16).
    3. QB Volatility Overlap: Unconfirmed starting QB adds 0.04 per unconfirmed leg.
    4. Heavy Public Favorite Chalk Penalty: Odds <= -600 incur progressive 0.015 to 0.05 penalty.
    5. Leg Count Decay: 0.02 per leg above 5 legs.

    Returns:
        Penalty discount factor bounded strictly between 0.0 and 0.35.
    """
    if len(legs) <= 1:
        return 0.0

    conf_counts: dict[str, int] = {}
    weather_hazard_count = 0
    qb_risk_count = 0
    heavy_fav_penalty = 0.0

    for leg in legs:
        c = leg.conference.strip().lower()
        if c and c not in ("independent", "fcs", "unknown", ""):
            conf_counts[c] = conf_counts.get(c, 0) + 1
        if leg.weather_hazard:
            weather_hazard_count += 1
        if leg.qb_news_risk:
            qb_risk_count += 1
        if leg.odds_american <= -600 or leg.is_heavy_favorite:
            excess = max(0, abs(leg.odds_american) - 600)
            heavy_fav_penalty += min(0.05, 0.015 + 0.00005 * min(excess, 600))

    # 1. Conference clustering penalty: pairs * 0.03, capped at 0.12 per conference
    conf_penalty = 0.0
    for cnt in conf_counts.values():
        if cnt > 1:
            pairs = (cnt * (cnt - 1)) // 2
            conf_penalty += min(0.12, pairs * 0.03)

    # 2. Weather hazard penalty: 0.04 per hazard leg beyond the first (or 0.04 per hazard leg, max 0.16)
    weather_penalty = min(0.16, 0.04 * weather_hazard_count)

    # 3. QB news penalty: 0.04 per volatile QB leg
    qb_penalty = 0.04 * qb_risk_count

    # 4. Heavy public favorite penalty: capped at 0.12
    heavy_fav_penalty = min(0.12, heavy_fav_penalty)

    # 5. Long parlay tail decay: 0.02 per leg above 5 legs
    count_penalty = 0.02 * max(0, len(legs) - 5)

    total_penalty = conf_penalty + weather_penalty + qb_penalty + heavy_fav_penalty + count_penalty
    return min(0.35, max(0.0, round(total_penalty, 4)))


def compute_fragility_index(legs: Sequence[ParlayLeg], joint_prob: float = 0.0) -> float:
    """Calculate the normalized fragility index (0.0 to 1.0) of a parlay.

    Unifies three structural vulnerabilities:
    1. Size Fragility: Scaling from K=3 to K=10 (up to 0.35).
    2. Weakest Leg Hazard: Vulnerability to lowest individual probability (up to 0.35).
    3. Risk & Volatility Burden: Accumulated correlation/weather/QB penalty (up to 0.30).

    Returns:
        float strictly in [0.0, 1.0].
    """
    if not legs:
        return 1.0

    k = len(legs)
    # Size penalty: 0.0 at K=3, 0.35 at K=10
    phi_size = ((max(3, min(10, k)) - 3) / 7.0) * 0.35

    # Weakest leg hazard: 0.0 if min_prob >= 0.85, 0.35 if min_prob <= 0.50
    min_prob = min(leg.fair_prob for leg in legs)
    if min_prob >= 0.85:
        phi_weak = 0.0
    elif min_prob <= 0.50:
        phi_weak = 0.35
    else:
        phi_weak = ((0.85 - min_prob) / (0.85 - 0.50)) * 0.35

    # Risk penalty burden:
    corr_penalty = compute_correlation_penalty(legs)
    phi_risk = min(0.30, corr_penalty)

    raw_fragility = phi_size + phi_weak + phi_risk
    return min(1.0, max(0.0, round(raw_fragility, 4)))


def compute_efficiency_ratio(ev: float, fragility_index: float) -> float:
    """Calculate the efficiency ratio rho = EV / Fragility.

    Rewards parlays with positive expected value per unit of fragility hazard.
    """
    if ev <= 0.0:
        return 0.0
    if fragility_index <= 0.0001:
        return round(ev * 100.0, 4)
    return round(ev / fragility_index, 4)
