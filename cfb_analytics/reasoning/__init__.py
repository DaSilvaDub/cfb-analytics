"""Multi-Factor Contextual Game Reasoning Package (Milestone 1).

Provides an end-to-end, multi-factor reasoning and market-mispricing pipeline
for NCAA College Football evaluating Moneylines, Spreads, Totals, and Team Props
using deep situational data:
- Recent Games & Form (Honest Tape vs Cupcake Stat Inflation)
- Venue, Weather & Travel Dynamics (Atmospheric Attenuation, Altitude, Road Fatigue)
- Injuries & Trench Health (Starting OL/DL Attrition)
- Roster Talent Composite & Blue-Chip Ratios
- Program Structure & Schedule Spot
- Negative Favorite Gate against blind market bias
"""

from __future__ import annotations

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

__all__ = [
    "SHADOW_MODE_DISCLAIMER",
    "MultiFactorReasoningEngine",
    "NegativeFavoriteGate",
    "NegativeGateResult",
    "OpponentTier",
    "PlayTier",
    "PositionUnitGrades",
    "ReasoningCard",
    "SituationalContext",
    "TalentProfile",
    "TapeCategory",
    "TapeGame",
    "TapeProfile",
    "TravelProfile",
    "TrenchHealth",
    "TrenchMatchupResult",
    "VenueProfile",
    "VolumeRedistribution",
    "WeatherProfile",
    "analyze_tape",
    "build_talent_profile",
    "calculate_altitude_fatigue_tax",
    "calculate_fg_range_boost",
    "calculate_haversine_distance_miles",
    "calculate_weather_multipliers",
    "classify_opponent",
    "classify_tape_game",
    "compute_blue_chip_ratio",
    "compute_true_talent_composite",
    "create_tape_game",
    "evaluate_travel_profile",
    "evaluate_trench_attrition",
    "evaluate_trench_matchup",
    "parse_kickoff_et_hour",
    "redistribute_pass_to_rush",
]
