"""Grok Decision Governance & Moneyline Parlay Optimization (Milestone 3).

Central heuristic arbitration, situational rule execution, and multi-leg parlay
optimization layer for NCAA College Football quantitative handicapping.

Operates strictly in Shadow Mode:
UNPROMOTED - shadow output, not decision-grade.
"""

from __future__ import annotations

from cfb_analytics.governance.gate import GovernanceGate
from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    GovernanceAction,
    GovernanceVerdict,
    MarginalLegAnalysis,
    ParlayLeg,
    ParlayRecommendation,
    ParlaySlateSummary,
    ParlayTicket,
    RuleResult,
    american_to_decimal,
    compute_correlation_penalty,
    compute_efficiency_ratio,
    compute_fragility_index,
    compute_parlay_payout,
    decimal_to_american,
)
from cfb_analytics.governance.parlay import (
    OptimizerMode,
    ParlayOptimizer,
)
from cfb_analytics.governance.rules import (
    BaseGrokRule,
    CandidateWager,
    FatRoadDogFatigueRule,
    FirstHalfPreferenceRule,
    GrokRuleEngine,
    HighScoringConferenceRule,
    HomePowerSmashScriptRule,
    MontanaRule,
    ThinDogTrapRule,
)

__all__ = [
    "SHADOW_MODE_DISCLAIMER",
    "GovernanceAction",
    "RuleResult",
    "GovernanceVerdict",
    "ParlayLeg",
    "MarginalLegAnalysis",
    "ParlayTicket",
    "ParlayRecommendation",
    "ParlaySlateSummary",
    "CandidateWager",
    "BaseGrokRule",
    "HomePowerSmashScriptRule",
    "MontanaRule",
    "FatRoadDogFatigueRule",
    "ThinDogTrapRule",
    "HighScoringConferenceRule",
    "FirstHalfPreferenceRule",
    "GrokRuleEngine",
    "OptimizerMode",
    "ParlayOptimizer",
    "GovernanceGate",
    "american_to_decimal",
    "decimal_to_american",
    "compute_parlay_payout",
    "compute_correlation_penalty",
    "compute_fragility_index",
    "compute_efficiency_ratio",
]
