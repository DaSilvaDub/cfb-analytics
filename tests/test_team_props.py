"""Tests for the Team Props Engine (Model 3)."""

from __future__ import annotations

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.models.team_props import (
    TeamPropsInputs,
    project_team_production,
    validate_market_type,
)


def test_filters_player_props() -> None:
    """Proves player props are strictly filtered out."""
    with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
        validate_market_type("PLAYER_PROP")
    with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
        validate_market_type("player_prop")
    with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
        validate_market_type("PLAYER")

    # Team props should pass validation
    validate_market_type("TEAM_PROP")


def test_project_team_production() -> None:
    """Proves cascading logic correctly combines components."""
    inputs = TeamPropsInputs(
        pace=70.0,
        expected_possession_count=12.0,
        offensive_success_rate=0.45,
        explosiveness=1.5,
        expected_pass_attempts=35.0,
        expected_rushing_attempts=35.0,
        completion_probability=0.6,
        yards_per_completion=12.0,
        yards_before_contact=2.5,
        yards_after_contact=2.0,
    )

    # rec = 35 * 0.6 * 12 = 252 yards
    # rush = 35 * (2.5 + 2.0) = 157.5 yards
    # total yards = 252 + 157.5 = 409.5 yards
    # expected plays = 35 pass + 35 rush = 70
    # expected_ypp = 409.5 / 70 = 5.85

    result = project_team_production("TEAM_PROP", inputs)

    assert result.expected_offensive_plays == 70.0
    assert result.expected_yards_per_play == 5.85
    assert result.projected_team_offensive_yards == 409.5
    assert result.projected_team_receiving_yards == 252.0
    assert result.projected_team_rushing_yards == 157.5


def test_project_team_production_rejects_player_props() -> None:
    """Proves the main projection interface rejects player props."""
    inputs = TeamPropsInputs(
        pace=0,
        expected_possession_count=0,
        offensive_success_rate=0,
        explosiveness=0,
        expected_pass_attempts=0,
        expected_rushing_attempts=0,
        completion_probability=0,
        yards_per_completion=0,
        yards_before_contact=0,
        yards_after_contact=0,
    )
    with pytest.raises(SchemaError):
        project_team_production("PLAYER_PROP", inputs)
