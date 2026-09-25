"""Reasoning engine immutable models and contract definitions.

Provides frozen dataclasses, enums, and constants for multi-factor situational
evaluation across all 6 analytical dimensions:
1. Recent Games & Form (Honest Tape vs Junk Cupcake Exhibitions)
2. Position Unit Grades & QB Continuity
3. Venue, Weather & Travel Dynamics (Wind, Rain, Elevation, Fatigue)
4. Injuries & Trench Health (Starting OL/DL Attrition)
5. Program Structure & Coaching Context (Staff Continuity, Rest, Trap Spots)
6. Mathematical Model Edge & Consensus Vig-Free Fair Prices

All models strictly enforce:
- Pure Python standard library primitives (no pandas, numpy, or scipy).
- Absolute immutability via `@dataclass(frozen=True)` and tuple sequences.
- Shadow mode disclaimer watermark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Mandatory shadow mode disclaimer for unpromoted research output
SHADOW_MODE_DISCLAIMER: str = "UNPROMOTED - shadow output, not decision-grade"


class OpponentTier(str, Enum):
    """Hierarchical classification of opponent quality."""

    POWER_CONFERENCE = "POWER_CONFERENCE"  # SEC, Big Ten, Big 12, ACC, Notre Dame
    HONEST_FBS = "HONEST_FBS"  # Competitive G5 / top-75 FBS defense / non-cupcake FBS
    BOTTOM_G5 = "BOTTOM_G5"  # Low-tier G5 (bottom 15-20 FBS, low Elo, bottom MAC/C-USA)
    CUPCAKE_FCS = "CUPCAKE_FCS"  # All FCS programs (including top-ranked FCS like Montana/SDSU)
    UNKNOWN = "UNKNOWN"  # Unresolved or non-divisional


class TapeCategory(str, Enum):
    """Categorization of individual completed game tape."""

    HONEST_POWER = "HONEST_POWER"  # Game against Power conference opponent (Always Honest)
    HONEST_FBS = "HONEST_FBS"  # Game against respectable FBS opponent (Honest)
    HONEST_STRUGGLE = "HONEST_STRUGGLE"  # Close game or loss vs FCS/Bottom-G5 (Honest Floor Tape!)
    CUPCAKE_BLOWOUT = "CUPCAKE_BLOWOUT"  # Blowout stat-padding vs FCS/Bottom-G5 (FILTERED JUNK)
    UNCLASSIFIED = "UNCLASSIFIED"


class PlayTier(str, Enum):
    """Candidate wagering recommendation tier."""

    ELITE = "ELITE"  # Score 90.0 - 100.0, unanimous multi-factor confirmation
    STRONG = "STRONG"  # Score 80.0 - 89.9
    QUALIFIED = "QUALIFIED"  # Score 70.0 - 79.9 (Baseline threshold for actionable play)
    LEAN = "LEAN"  # Score 60.0 - 69.9
    PASS = "PASS"  # Score < 60.0 or failed qualification gate
    AVOID = "AVOID"  # High upset risk, QB uncertainty, or vetoed favorite


@dataclass(frozen=True)
class TapeGame:
    """Evaluated metrics of a single completed game from one team's perspective."""

    game_id: str
    season: int
    week: int
    kickoff_utc: str
    opponent_team_id: str
    opponent_name: str
    opponent_conference: str
    opponent_classification: str  # 'fbs' or 'fcs'
    is_home: bool
    points_scored: int
    points_allowed: int
    margin: int
    offensive_epa: float
    defensive_epa: float
    offensive_success_rate: float
    defensive_success_rate: float
    rushing_yards: float | None = None
    net_passing_yards: float | None = None
    total_yards: float | None = None
    opponent_tier: OpponentTier = OpponentTier.UNKNOWN
    tape_category: TapeCategory = TapeCategory.UNCLASSIFIED
    is_cupcake_blowout: bool = False


@dataclass(frozen=True)
class TapeProfile:
    """Aggregated game tape isolating honest FBS competition from cupcake blowouts."""

    honest_games: tuple[Any, ...] | list[Any] = field(default_factory=tuple)
    offensive_floor_epa: float = 0.0
    offensive_ceiling_epa: float = 0.0
    defensive_floor_epa: float = 0.0
    defensive_ceiling_epa: float = 0.0
    cupcake_games_filtered: int = 0
    honest_success_rate: float = 0.42
    honest_yards_per_play: float = 5.5
    team_id: str = ""
    team_name: str = ""
    total_games: int = 0
    honest_games_count: int = 0
    has_honest_tape: bool = True
    is_cupcake_inflated: bool = False
    mean_offensive_epa: float = 0.0
    offensive_floor_success_rate: float = 0.42
    offensive_ceiling_success_rate: float = 0.42
    mean_offensive_success_rate: float = 0.42
    mean_defensive_epa: float = 0.0
    defensive_floor_success_rate: float = 0.42
    defensive_ceiling_success_rate: float = 0.42
    mean_defensive_success_rate: float = 0.42
    offensive_floor_points: int = 0
    offensive_ceiling_points: int = 0
    mean_points_scored: float = 0.0
    defensive_floor_points: int = 0
    defensive_ceiling_points: int = 0
    mean_points_allowed: float = 0.0
    raw_ceiling_points: int = 0
    raw_ceiling_epa: float = 0.0
    inflation_gap_points: float = 0.0
    inflation_gap_epa: float = 0.0
    tape_confidence_penalty: float = 0.0
    cupcake_games: tuple[Any, ...] = field(default_factory=tuple)
    all_games: tuple[Any, ...] = field(default_factory=tuple)
    tape_summary: str = ""


@dataclass(frozen=True)
class PositionUnitGrades:
    """Positional unit group quality, trench disruption, and talent differentials."""

    team_id: str = ""
    qb_starter_name: str | None = None
    qb_starter_confirmed: bool = True
    qb_grade: float = 75.0
    qb_turnover_worthy_rate: float = 0.03
    qb_pressure_to_sack_rate: float = 0.15
    qb_epa_per_pass: float = 0.10
    rb_ol_grade: float = 75.0
    line_yards_avg: float = 3.0
    rush_ypc: float = 4.5
    defensive_front_grade: float = 75.0
    havoc_rate: float = 0.18
    stuff_rate: float = 0.20
    rush_defense_ypg_allowed: float = 140.0
    secondary_grade: float = 75.0
    pass_epa_allowed: float = 0.05
    depth_grade: float = 75.0
    roster_composite: float = 650.0
    blue_chip_ratio: float = 0.20
    recruiting_rank: int | None = None
    portal_net_score: float = 0.0
    trench_mismatch_score: float = 0.0


@dataclass(frozen=True)
class WeatherProfile:
    """Localized atmospheric impact curves and play volume redistribution."""

    temperature_c: float
    wind_kph: float
    gust_kph: float
    precip_mm: float
    is_dome: bool
    pass_vol_mult: float
    rush_vol_mult: float
    scoring_mult: float
    effective_wind_kph: float = 0.0
    excess_wind_kph: float = 0.0
    cold_deficit_c: float = 0.0
    heat_excess_c: float = 0.0
    precip_prob: float = 0.0
    humidity: float | None = None
    comp_prob_mult: float = 1.0
    ypc_mult: float = 1.0
    is_extreme_weather: bool = False
    weather_summary: str = ""


@dataclass(frozen=True)
class VenueProfile:
    """Venue geographical and physical attributes."""

    venue_id: str
    name: str
    city: str | None = None
    state: str | None = None
    elevation_m: float | None = None
    is_indoor: bool = False
    latitude: float | None = None
    longitude: float | None = None
    timezone: str = "America/New_York"


@dataclass(frozen=True)
class TravelProfile:
    """Travel distance, circadian disruption, and fatigue quantification."""

    distance_miles: float
    timezone_shift_hours: int
    is_westward: bool
    kickoff_et_hour: float
    is_late_kickoff: bool
    distance_fatigue_tax: float
    timezone_fatigue_tax: float
    late_kickoff_fatigue_tax: float
    altitude_fatigue_tax: float
    total_travel_fatigue_tax: float
    rule_c_triggered: bool


@dataclass(frozen=True)
class VolumeRedistribution:
    """Run/Pass volume shift resulting from adverse conditions."""

    orig_pass_attempts: float
    orig_rush_attempts: float
    adj_pass_attempts: float
    adj_rush_attempts: float
    delta_pass: float
    rush_boost: float
    adj_pace: float


@dataclass(frozen=True)
class TalentProfile:
    """Comprehensive program talent and recruiting profile."""

    team_id: str
    recruiting_composite: float
    recruiting_rank: int | None
    net_portal_composite: float
    true_talent_composite: float
    blue_chip_ratio: float
    is_blue_chip_program: bool  # BCR >= 0.50
    talent_tier: str  # ELITE, UPPER_P4, MID_P4, LOWER_P4_HIGH_G5, G5_BASELINE


@dataclass(frozen=True)
class TrenchHealth:
    """Unit-level trench availability and distortion factors."""

    team_id: str
    ol_attrition: float  # 0.0 to 1.0
    dl_attrition: float  # 0.0 to 1.0
    composite_trench_attrition: float  # 0.0 to 1.0
    ol_missing_starters: tuple[str, ...] = field(default_factory=tuple)
    dl_missing_starters: tuple[str, ...] = field(default_factory=tuple)
    line_yards_mult: float = 1.0
    rush_success_rate_mult: float = 1.0
    sack_rate_allowed_mult: float = 1.0
    def_stuff_rate_mult: float = 1.0
    def_havoc_mult: float = 1.0
    def_sack_rate_mult: float = 1.0
    is_trench_compromised: bool = False


@dataclass(frozen=True)
class TrenchMatchupResult:
    """Head-to-head trench confrontation."""

    offense_team_id: str
    defense_team_id: str
    ol_health: TrenchHealth
    dl_health: TrenchHealth
    projected_line_yards: float
    projected_sack_rate: float
    trench_mismatch_score: float
    is_decisive_mismatch: bool  # Qualifies Rule D
    summary_text: str = ""


@dataclass(frozen=True)
class SituationalContext:
    """Complete 6-dimensional situational dossier for a single matchup."""

    game_id: str
    home_team: str
    away_team: str
    kickoff_utc: str
    kickoff_et: str
    tape_home: TapeProfile
    tape_away: TapeProfile
    weather: WeatherProfile
    qb_home_confirmed: bool
    qb_away_confirmed: bool
    trench_attrition_home: float  # 0.0 to 1.0 (higher = worse)
    trench_attrition_away: float
    talent_composite_home: float
    talent_composite_away: float
    rest_days_home: float
    rest_days_away: float
    travel_fatigue_tax_away: float
    qb_home_name: str | None = None
    qb_away_name: str | None = None
    qb_home_attempt_share: float = 1.0
    qb_away_attempt_share: float = 1.0
    coaching_continuity_home: float = 1.0
    coaching_continuity_away: float = 1.0
    lookahead_flag_home: bool = False
    lookahead_flag_away: bool = False
    venue_elevation_m: float = 0.0
    is_neutral_site: bool = False
    home_team_id: str = ""
    away_team_id: str = ""
    position_grades_home: PositionUnitGrades | None = None
    position_grades_away: PositionUnitGrades | None = None
    is_dome: bool = False
    travel_distance_miles: float = 0.0
    time_zones_crossed: int = 0
    is_late_kickoff: bool = False
    home_field_advantage_points: float = 2.5
    ol_starters_out_home: int = 0
    ol_starters_out_away: int = 0
    dl_starters_out_home: int = 0
    dl_starters_out_away: int = 0
    key_injury_notes_home: tuple[str, ...] = field(default_factory=tuple)
    key_injury_notes_away: tuple[str, ...] = field(default_factory=tuple)
    rest_disparity: float = 0.0
    is_bye_week_home: bool = False
    is_bye_week_away: bool = False
    is_trap_or_lookahead_home: bool = False
    is_trap_or_lookahead_away: bool = False
    market_spread_home: float | None = None
    market_total: float | None = None
    market_ml_home_american: int | None = None
    market_ml_away_american: int | None = None


@dataclass(frozen=True)
class NegativeGateResult:
    """Audit payload emitted by the Anti-Blind-Favorite Gate."""

    is_vetoed: bool
    gate_status: str  # 'CLEARED', 'FLAGGED', 'VETOED'
    counter_thesis: str | None
    contra_indications: tuple[str, ...]
    disqualifying_reasons: tuple[str, ...]
    confidence_penalty: float


@dataclass(frozen=True)
class ReasoningCard:
    """Auditable 6-dimensional recommendation dossier emitted for each evaluated wager."""

    game_id: str
    market: str  # 'SPREAD', 'TOTAL', 'ML', 'TEAM_PROP'
    side: str  # 'HOME', 'AWAY', 'OVER', 'UNDER'
    recommended_play: str  # e.g., 'HOME -14.0' or 'OVER 52.5'
    confidence: float  # 0.0 to 10.0 scale
    tier: str  # 'ELITE', 'STRONG', 'QUALIFIED', 'LEAN', 'PASS', 'AVOID'
    tape_summary: str
    position_qb_summary: str
    weather_venue_summary: str
    injuries_trench_summary: str
    program_continuity_summary: str
    mathematical_edge_summary: str
    contra_indications: list[str] = field(default_factory=list)
    is_favorite_vetoed: bool = False
    counter_thesis: str | None = None
    matchup_label: str = ""
    kickoff_et: str = ""
    line: float | None = None
    price_american: int | None = None
    rule_triggers: tuple[str, ...] = field(default_factory=tuple)
    shadow_mode_disclaimer: str = SHADOW_MODE_DISCLAIMER
