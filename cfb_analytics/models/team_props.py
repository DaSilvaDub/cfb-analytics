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


@dataclass(frozen=True)
class TeamPropsProjection:
    """Cascaded projection of team offensive production."""

    expected_offensive_plays: float
    expected_yards_per_play: float
    projected_team_offensive_yards: float
    projected_team_receiving_yards: float
    projected_team_rushing_yards: float
    projected_team_total_points: float = 0.0


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


def project_team_points(inputs: TeamPropsInputs, total_yards: float | None = None) -> float:
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

    return round(yards_baseline * (0.6 * eff_factor + 0.4 * exp_factor), 2)


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
    )


def project_team_production(market_type: str, inputs: TeamPropsInputs) -> TeamPropsProjection:
    """Cascading logic to project team production."""
    validate_market_type(market_type)

    rec_yards = project_receiving_yards(inputs)
    rush_yards = project_rushing_yards(inputs)
    expected_plays, expected_ypp, total_yards = project_offensive_yards(
        inputs, rec_yards, rush_yards
    )
    projected_points = project_team_points(inputs, total_yards)

    return TeamPropsProjection(
        expected_offensive_plays=expected_plays,
        expected_yards_per_play=expected_ypp,
        projected_team_offensive_yards=total_yards,
        projected_team_receiving_yards=rec_yards,
        projected_team_rushing_yards=rush_yards,
        projected_team_total_points=projected_points,
    )
