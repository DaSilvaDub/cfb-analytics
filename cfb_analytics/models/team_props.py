"""Team Props Engine (Model 3).

Cascading logic that projects team offensive yards, rushing, receiving
yards, and team total points while strictly filtering out any player props.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfb_analytics.errors import SchemaError

SUPPORTED_TEAM_PROPS: tuple[str, ...] = (
    "team_offensive_yards",
    "team_rushing_yards",
    "team_receiving_yards",
    "team_total_points",
)

MARKET_ALIASES: dict[str, str] = {
    "offensive_yards": "team_offensive_yards",
    "receiving_yards": "team_receiving_yards",
    "rushing_yards": "team_rushing_yards",
    "passing_yards": "team_receiving_yards",
    "team_passing_yards": "team_receiving_yards",
    "team_pass_yards": "team_receiving_yards",
    "points": "team_total_points",
    "team_points": "team_total_points",
    "total_team_points": "team_total_points",
    "team_total": "team_total_points",
    "team_total_points": "team_total_points",
    "team_offensive_yards": "team_offensive_yards",
    "team_receiving_yards": "team_receiving_yards",
    "team_rushing_yards": "team_rushing_yards",
}


@dataclass(frozen=True)
class TeamPropsInputs:
    """Inputs to the team props engine."""

    pace: float
    expected_possession_count: float
    offensive_success_rate: float
    explosiveness: float
    expected_pass_attempts: float
    expected_rushing_attempts: float
    completion_probability: float
    yards_per_completion: float
    yards_before_contact: float
    yards_after_contact: float
    data_quality_score: float = 100.0
    points_multiplier: float = 1.0


@dataclass(frozen=True)
class TeamPropsProjection:
    """Cascaded projection of team offensive production."""

    expected_offensive_plays: float
    expected_yards_per_play: float
    projected_team_offensive_yards: float
    projected_team_receiving_yards: float
    projected_team_rushing_yards: float
    projected_team_total_points: float = 0.0


@dataclass(frozen=True)
class WeatherConditions:
    """Point-in-time weather conditions for a game."""

    temp_c: float | None = None
    wind_kph: float | None = None
    wind_gust_kph: float | None = None
    wind_dir_deg: float | None = None
    precip_mm: float | None = None
    precip_prob: float | None = None
    humidity: float | None = None
    is_indoor: bool = False
    is_forecast: bool = True
    as_of_utc: str | None = None


@dataclass(frozen=True)
class WeatherImpactFactors:
    """Attenuation multipliers and adjustments derived from weather conditions."""

    pass_attempt_multiplier: float = 1.0
    rush_attempt_multiplier: float = 1.0
    completion_prob_multiplier: float = 1.0
    yards_per_completion_multiplier: float = 1.0
    yards_before_contact_multiplier: float = 1.0
    yards_after_contact_multiplier: float = 1.0
    offensive_success_rate_multiplier: float = 1.0
    explosiveness_multiplier: float = 1.0
    total_points_multiplier: float = 1.0
    wind_attenuation: float = 1.0
    precip_attenuation: float = 1.0
    temp_attenuation: float = 1.0


@dataclass(frozen=True)
class GameTotalsProjection:
    """Projected game total and individual team totals with weather attenuation."""

    home_projected_points: float
    away_projected_points: float
    projected_game_total: float
    weather_impact: WeatherImpactFactors | None = None


def compute_weather_impact(weather: WeatherConditions | None) -> WeatherImpactFactors:
    """Compute weather attenuation factors for team props and game totals.

    Indoor games (domes) return unity multipliers (no weather attenuation).
    Outdoor conditions apply uncalibrated heuristic attenuation:
    - Wind speed / gusts: reduces passing volume, completion rate, and YPC; depresses totals.
    - Precipitation: reduces passing efficiency, shifts volume to rushing, dampens totals.
    - Temperature: below 5 Celsius dampens passing and totals; above 30 reduces volume.
    """
    if weather is None or weather.is_indoor:
        return WeatherImpactFactors()

    # 1. Wind speed and gusts
    wind = weather.wind_kph or 0.0
    gust = weather.wind_gust_kph or 0.0
    effective_wind = max(wind, gust * 0.75)
    excess_wind = max(0.0, effective_wind - 15.0)

    wind_pass_vol = max(0.65, 1.0 - 0.008 * excess_wind)
    wind_comp_prob = max(0.70, 1.0 - 0.006 * excess_wind)
    wind_ypc = max(0.70, 1.0 - 0.007 * excess_wind)
    wind_exp = max(0.75, 1.0 - 0.006 * excess_wind)
    wind_points = max(0.75, 1.0 - 0.006 * excess_wind)

    # 2. Precipitation
    precip = weather.precip_mm or 0.0
    if precip == 0.0 and (weather.precip_prob or 0.0) >= 60.0:
        precip = ((weather.precip_prob or 0.0) - 50.0) * 0.02
    excess_precip = min(15.0, max(0.0, precip - 0.1))

    precip_pass_vol = max(0.75, 1.0 - 0.025 * excess_precip)
    precip_comp_prob = max(0.80, 1.0 - 0.030 * excess_precip)
    precip_ypc = max(0.85, 1.0 - 0.020 * excess_precip)
    precip_rush_eff = max(0.90, 1.0 - 0.010 * excess_precip)
    precip_sr = max(0.85, 1.0 - 0.015 * excess_precip)
    precip_points = max(0.80, 1.0 - 0.025 * excess_precip)

    # 3. Temperature (Extreme Cold and Extreme Heat)
    temp = weather.temp_c
    if temp is not None:
        if temp < 5.0:
            cold_deficit = min(25.0, max(0.0, 5.0 - temp))
            temp_pass_vol = max(0.80, 1.0 - 0.008 * cold_deficit)
            temp_comp_prob = max(0.75, 1.0 - 0.010 * cold_deficit)
            temp_ypc = max(0.80, 1.0 - 0.008 * cold_deficit)
            temp_exp = max(0.80, 1.0 - 0.008 * cold_deficit)
            temp_points = max(0.75, 1.0 - 0.010 * cold_deficit)
        elif temp > 30.0:
            heat_excess = min(15.0, max(0.0, temp - 30.0))
            temp_pass_vol = max(0.95, 1.0 - 0.003 * heat_excess)
            temp_comp_prob = max(0.95, 1.0 - 0.003 * heat_excess)
            temp_ypc = 1.0
            temp_exp = 1.0
            temp_points = max(0.92, 1.0 - 0.006 * heat_excess)
        else:
            temp_pass_vol = 1.0
            temp_comp_prob = 1.0
            temp_ypc = 1.0
            temp_exp = 1.0
            temp_points = 1.0
    else:
        temp_pass_vol = 1.0
        temp_comp_prob = 1.0
        temp_ypc = 1.0
        temp_exp = 1.0
        temp_points = 1.0

    pass_att_mult = round(max(0.50, min(1.0, wind_pass_vol * precip_pass_vol * temp_pass_vol)), 4)
    rush_att_mult = round(min(1.35, max(1.0, 1.0 + (1.0 - pass_att_mult) * 0.8)), 4)
    comp_prob_mult = round(
        max(0.50, min(1.0, wind_comp_prob * precip_comp_prob * temp_comp_prob)), 4
    )
    ypc_mult = round(max(0.50, min(1.0, wind_ypc * precip_ypc * temp_ypc)), 4)
    ybc_mult = round(max(0.60, min(1.0, precip_rush_eff)), 4)
    yac_mult = round(max(0.60, min(1.0, precip_rush_eff)), 4)
    sr_mult = round(max(0.60, min(1.0, precip_sr)), 4)
    exp_mult = round(max(0.60, min(1.0, wind_exp * temp_exp)), 4)
    total_pts_mult = round(max(0.55, min(1.0, wind_points * precip_points * temp_points)), 4)

    return WeatherImpactFactors(
        pass_attempt_multiplier=pass_att_mult,
        rush_attempt_multiplier=rush_att_mult,
        completion_prob_multiplier=comp_prob_mult,
        yards_per_completion_multiplier=ypc_mult,
        yards_before_contact_multiplier=ybc_mult,
        yards_after_contact_multiplier=yac_mult,
        offensive_success_rate_multiplier=sr_mult,
        explosiveness_multiplier=exp_mult,
        total_points_multiplier=total_pts_mult,
        wind_attenuation=round(wind_points, 4),
        precip_attenuation=round(precip_points, 4),
        temp_attenuation=round(temp_points, 4),
    )


def adjust_team_props_inputs_for_weather(
    inputs: TeamPropsInputs,
    weather: WeatherConditions | None,
) -> TeamPropsInputs:
    """Adjust TeamPropsInputs based on weather conditions.

    Shifts passing volume to rushing volume, reduces passing efficiency,
    and adjusts line yards and explosive rates while preserving data quality score.
    """
    if weather is None or weather.is_indoor:
        return inputs

    if inputs.points_multiplier != 1.0:
        # Already weather-adjusted; do not double attenuate
        return inputs

    impact = compute_weather_impact(weather)
    if (
        impact.pass_attempt_multiplier == 1.0
        and impact.rush_attempt_multiplier == 1.0
        and impact.completion_prob_multiplier == 1.0
        and impact.yards_per_completion_multiplier == 1.0
        and impact.yards_before_contact_multiplier == 1.0
        and impact.offensive_success_rate_multiplier == 1.0
        and impact.explosiveness_multiplier == 1.0
        and impact.total_points_multiplier == 1.0
    ):
        return inputs

    adj_pass_att = round(
        max(10.0, min(65.0, inputs.expected_pass_attempts * impact.pass_attempt_multiplier)), 1
    )
    pass_delta = inputs.expected_pass_attempts - adj_pass_att
    rush_boost = max(0.0, pass_delta * 0.85)
    adj_rush_att = round(max(15.0, min(75.0, (inputs.expected_rushing_attempts + rush_boost))), 1)
    adj_pace = round(adj_pass_att + adj_rush_att, 1)

    adj_comp_prob = round(
        max(0.35, min(0.85, inputs.completion_probability * impact.completion_prob_multiplier)), 3
    )
    adj_ypc = round(
        max(5.0, min(25.0, inputs.yards_per_completion * impact.yards_per_completion_multiplier)), 2
    )
    adj_ybc = round(
        max(0.5, inputs.yards_before_contact * impact.yards_before_contact_multiplier), 2
    )
    adj_yac = round(max(0.5, inputs.yards_after_contact * impact.yards_after_contact_multiplier), 2)
    adj_sr = round(
        max(
            0.15,
            min(0.85, inputs.offensive_success_rate * impact.offensive_success_rate_multiplier),
        ),
        4,
    )
    adj_exp = round(max(0.50, inputs.explosiveness * impact.explosiveness_multiplier), 4)

    return TeamPropsInputs(
        pace=adj_pace,
        expected_possession_count=inputs.expected_possession_count,
        offensive_success_rate=adj_sr,
        explosiveness=adj_exp,
        expected_pass_attempts=adj_pass_att,
        expected_rushing_attempts=adj_rush_att,
        completion_probability=adj_comp_prob,
        yards_per_completion=adj_ypc,
        yards_before_contact=adj_ybc,
        yards_after_contact=adj_yac,
        data_quality_score=inputs.data_quality_score,
        points_multiplier=impact.total_points_multiplier,
    )


def project_game_total(
    home_inputs: TeamPropsInputs,
    away_inputs: TeamPropsInputs,
    weather: WeatherConditions | None = None,
) -> GameTotalsProjection:
    """Project game total and team points, adjusting for weather.

    Cascades team offensive production from Model 3 TeamPropsInputs (adjusted
    for weather efficiency), projects team points, and applies overall weather
    total dampening.
    """
    impact = compute_weather_impact(weather) if weather else WeatherImpactFactors()

    home_proj = project_team_production("TEAM_PROP", home_inputs, weather=weather)
    away_proj = project_team_production("TEAM_PROP", away_inputs, weather=weather)

    home_pts = home_proj.projected_team_total_points
    away_pts = away_proj.projected_team_total_points
    game_total = round(home_pts + away_pts, 2)

    return GameTotalsProjection(
        home_projected_points=home_pts,
        away_projected_points=away_pts,
        projected_game_total=game_total,
        weather_impact=impact,
    )


def is_player_prop(market_or_type: str) -> bool:
    """True if the market name or type indicates a player prop."""
    norm = market_or_type.strip().lower().replace("-", "_").replace(" ", "_")
    if norm.startswith("team_") or norm.startswith("team"):
        return False
    return (
        "player" in norm
        or "passer" in norm
        or "rusher" in norm
        or "receiver" in norm
        or "touchdown" in norm
        or "anytime" in norm
        or "first_td" in norm
        or norm.startswith("pass_")
        or norm.startswith("rush_")
        or norm.startswith("rec_")
        or norm.endswith("_player")
    )


def validate_market_type(market_type: str) -> None:
    """Require an explicit team-prop market family.

    Model 3 strictly prohibits player props due to rotation volatility,
    blowout substitutions, and unreliable injury reporting.
    """
    normalized = market_type.strip().upper()
    if normalized != "TEAM_PROP":
        if "PLAYER" in normalized:
            raise SchemaError("Player props are strictly prohibited in Model 3")
        raise SchemaError("Model 3 requires market_type='TEAM_PROP'")


def normalize_team_prop_market(market: str, *, market_type: str = "TEAM_PROP") -> str:
    """Normalize market identifier to canonical team prop name.

    Raises SchemaError if market is a player prop or unsupported.
    """
    validate_market_type(market_type)
    norm = market.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = MARKET_ALIASES.get(norm, norm)
    if canonical not in SUPPORTED_TEAM_PROPS:
        raise SchemaError(
            f"Unsupported market {market!r}; supported team props are: {SUPPORTED_TEAM_PROPS}"
        )
    return canonical


def is_supported_team_prop(market: str, *, market_type: str = "TEAM_PROP") -> bool:
    """True if market is one of the supported team props."""
    try:
        validate_market_type(market_type)
    except SchemaError:
        return False
    norm = market.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = MARKET_ALIASES.get(norm, norm)
    return canonical in SUPPORTED_TEAM_PROPS


def project_offensive_yards(
    inputs: TeamPropsInputs, rec_yards: float, rush_yards: float
) -> tuple[float, float, float]:
    """Project overall team offensive yards by cascading from passing and rushing."""
    expected_plays = inputs.expected_pass_attempts + inputs.expected_rushing_attempts
    projected_yards = rec_yards + rush_yards
    expected_ypp = projected_yards / expected_plays if expected_plays > 0 else 0.0
    return expected_plays, expected_ypp, projected_yards


def project_receiving_yards(inputs: TeamPropsInputs) -> float:
    """Project team receiving yards."""
    return (
        inputs.expected_pass_attempts * inputs.completion_probability * inputs.yards_per_completion
    )


def project_rushing_yards(inputs: TeamPropsInputs) -> float:
    """Project team rushing yards."""
    return inputs.expected_rushing_attempts * (
        inputs.yards_before_contact + inputs.yards_after_contact
    )


def project_team_points(
    inputs: TeamPropsInputs,
    total_yards: float | None = None,
    *,
    weather: WeatherConditions | None = None,
) -> float:
    """Project team total points from cascading pace and efficiency inputs."""
    if total_yards is None:
        rec_yards = project_receiving_yards(inputs)
        rush_yards = project_rushing_yards(inputs)
        _, _, total_yards = project_offensive_yards(inputs, rec_yards, rush_yards)

    if total_yards <= 0:
        return 0.0

    yards_baseline = total_yards / 14.5
    eff_factor = (
        (inputs.offensive_success_rate / 0.42) if inputs.offensive_success_rate > 0 else 1.0
    )
    exp_factor = (inputs.explosiveness / 1.25) if inputs.explosiveness > 0 else 1.0

    pts = yards_baseline * (0.6 * eff_factor + 0.4 * exp_factor)
    if inputs.points_multiplier != 1.0:
        pts *= inputs.points_multiplier
    elif weather is not None:
        impact = compute_weather_impact(weather)
        pts *= impact.total_points_multiplier

    return round(pts, 2)


def default_team_props_inputs(
    *,
    pace: float = 70.0,
    expected_possession_count: float = 12.0,
    offensive_success_rate: float = 0.42,
    explosiveness: float = 1.25,
    expected_pass_attempts: float = 32.0,
    expected_rushing_attempts: float = 36.0,
    completion_probability: float = 0.62,
    yards_per_completion: float = 11.5,
    yards_before_contact: float = 2.4,
    yards_after_contact: float = 2.1,
    data_quality_score: float = 0.0,
    points_multiplier: float = 1.0,
) -> TeamPropsInputs:
    """Standard FBS league-average baseline inputs when history is unpopulated."""
    return TeamPropsInputs(
        pace=pace,
        expected_possession_count=expected_possession_count,
        offensive_success_rate=offensive_success_rate,
        explosiveness=explosiveness,
        expected_pass_attempts=expected_pass_attempts,
        expected_rushing_attempts=expected_rushing_attempts,
        completion_probability=completion_probability,
        yards_per_completion=yards_per_completion,
        yards_before_contact=yards_before_contact,
        yards_after_contact=yards_after_contact,
        data_quality_score=data_quality_score,
        points_multiplier=points_multiplier,
    )


def project_team_production(
    market_type: str,
    inputs: TeamPropsInputs,
    *,
    weather: WeatherConditions | None = None,
) -> TeamPropsProjection:
    """Cascading logic to project team production."""
    validate_market_type(market_type)

    if weather is not None:
        inputs = adjust_team_props_inputs_for_weather(inputs, weather)

    rec_yards = project_receiving_yards(inputs)
    rush_yards = project_rushing_yards(inputs)
    expected_plays, expected_ypp, total_yards = project_offensive_yards(
        inputs, rec_yards, rush_yards
    )
    projected_points = project_team_points(inputs, total_yards, weather=weather)

    return TeamPropsProjection(
        expected_offensive_plays=expected_plays,
        expected_yards_per_play=expected_ypp,
        projected_team_offensive_yards=total_yards,
        projected_team_receiving_yards=rec_yards,
        projected_team_rushing_yards=rush_yards,
        projected_team_total_points=projected_points,
    )
