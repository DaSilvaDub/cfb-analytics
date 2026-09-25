"""NCAA College Football Comprehensive Mispriced Line Scanner (Milestone 2).

Submodules:
- models: Frozen dataclasses and enums (MispricedOpportunity, PlayScore, PlayTier, QualificationStatus).
- spreads: Model 2 spread margin vs devigged consensus spread, cover prob via normal CDF erf, EV.
- totals: Tempo and weather-adjusted scoring projections vs consensus game totals, normal CDF (sigma=10.75).
- props: Model 3 team points, rushing yards, and receiving yards vs posted book lines (strictly team props).
- scoring: Candidate Play Scoring (0-100 scale) and minimum qualification gates.
- engine: MispricedScanner orchestrating multi-market scanning and ReasoningCard integration.
"""

from __future__ import annotations

from cfb_analytics.scanner.engine import MispricedScanner
from cfb_analytics.scanner.models import (
    SHADOW_MODE_DISCLAIMER,
    MispricedOpportunity,
    PlayScore,
    PlayTier,
    QualificationStatus,
)
from cfb_analytics.scanner.props import (
    BLOWOUT_GAME_YARDS,
    DOG_REC_BLOWOUT,
    PROB_CAP,
    SIT_QB_SPREAD,
    PlayerPropProhibitedError,
    calibrated_prop_prob,
    calculate_prop_edge,
    evaluate_team_prop_candidate,
    is_sit_qb_candidate,
    validate_team_prop,
)
from cfb_analytics.scanner.scoring import (
    assign_play_tier,
    calculate_play_score,
    evaluate_qualification_gates,
    score_confirming_insights,
    score_edge,
    score_hit_rate,
    score_line_movement,
    score_price_value,
    score_sample_size,
)
from cfb_analytics.scanner.spreads import (
    SPREAD_SIGMA_BASE,
    calculate_spread_edge,
    convert_spread_line_to_hurdle,
    cover_probability,
    evaluate_spread_candidate,
)
from cfb_analytics.scanner.totals import (
    TOTAL_SIGMA_BASE,
    calculate_total_edge,
    evaluate_total_candidate,
    project_game_total_adjusted,
    totals_cover_probability,
)

__all__ = [
    # Disclaimer
    "SHADOW_MODE_DISCLAIMER",
    # Data models
    "MispricedOpportunity",
    "PlayScore",
    "PlayTier",
    "QualificationStatus",
    # Main scanner engine
    "MispricedScanner",
    # Spreads
    "SPREAD_SIGMA_BASE",
    "cover_probability",
    "calculate_spread_edge",
    "convert_spread_line_to_hurdle",
    "evaluate_spread_candidate",
    # Totals
    "TOTAL_SIGMA_BASE",
    "totals_cover_probability",
    "calculate_total_edge",
    "project_game_total_adjusted",
    "evaluate_total_candidate",
    # Team props
    "PlayerPropProhibitedError",
    "validate_team_prop",
    "is_sit_qb_candidate",
    "calibrated_prop_prob",
    "calculate_prop_edge",
    "evaluate_team_prop_candidate",
    "PROB_CAP",
    "SIT_QB_SPREAD",
    "DOG_REC_BLOWOUT",
    "BLOWOUT_GAME_YARDS",
    # Scoring and gating
    "calculate_play_score",
    "assign_play_tier",
    "evaluate_qualification_gates",
    "score_hit_rate",
    "score_edge",
    "score_line_movement",
    "score_price_value",
    "score_sample_size",
    "score_confirming_insights",
]
