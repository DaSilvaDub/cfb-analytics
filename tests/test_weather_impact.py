"""Comprehensive tests for Weather Impact Modeling and Open-Meteo integration.

Verifies:
1. Weather attenuation factors (wind, gusts, precipitation, extreme cold/heat).
2. Indoor / dome invariance (zero attenuation).
3. Adjustments to Model 3 TeamPropsInputs (volume shift and efficiency).
4. Cascading game totals projections and points dampening.
5. Invariants: strictly NO player props, pure Python standard library.
6. SQLite point-in-time weather retrieval and feature extraction.
"""

from __future__ import annotations

import sqlite3

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.weather import (
    build_weather_adjusted_inputs_for_slate,
    extract_weather_features,
    get_weather_for_game,
    load_weather_for_games,
    project_slate_game_totals,
)
from cfb_analytics.ingest import store
from cfb_analytics.models.team_props import (
    TeamPropsInputs,
    WeatherConditions,
    adjust_team_props_inputs_for_weather,
    compute_weather_impact,
    default_team_props_inputs,
    is_player_prop,
    project_game_total,
    project_team_points,
    project_team_production,
    validate_market_type,
)
from cfb_analytics.scoring import (
    load_team_props_inputs_for_slate,
    score_daily_slates,
    score_slate_team_props,
)


@pytest.fixture
def baseline_inputs() -> TeamPropsInputs:
    return TeamPropsInputs(
        pace=70.0,
        expected_possession_count=12.0,
        offensive_success_rate=0.45,
        explosiveness=1.35,
        expected_pass_attempts=35.0,
        expected_rushing_attempts=35.0,
        completion_probability=0.62,
        yards_per_completion=12.0,
        yards_before_contact=2.5,
        yards_after_contact=2.0,
        data_quality_score=95.0,
    )


class TestWeatherImpactFactors:
    def test_calm_weather_has_no_attenuation(self):
        calm = WeatherConditions(
            temp_c=20.0,
            wind_kph=10.0,
            wind_gust_kph=12.0,
            precip_mm=0.0,
            precip_prob=10.0,
            humidity=50.0,
            is_indoor=False,
        )
        impact = compute_weather_impact(calm)
        assert impact.pass_attempt_multiplier == 1.0
        assert impact.rush_attempt_multiplier == 1.0
        assert impact.completion_prob_multiplier == 1.0
        assert impact.yards_per_completion_multiplier == 1.0
        assert impact.total_points_multiplier == 1.0
        assert impact.wind_attenuation == 1.0
        assert impact.precip_attenuation == 1.0
        assert impact.temp_attenuation == 1.0

    def test_indoor_dome_strictly_ignores_weather_numbers(self):
        """Even if wind or rain values are populated, a dome has strictly 1.0 impact."""
        hurricane_dome = WeatherConditions(
            temp_c=-10.0,
            wind_kph=80.0,
            wind_gust_kph=120.0,
            precip_mm=30.0,
            is_indoor=True,
        )
        impact = compute_weather_impact(hurricane_dome)
        assert impact.pass_attempt_multiplier == 1.0
        assert impact.rush_attempt_multiplier == 1.0
        assert impact.completion_prob_multiplier == 1.0
        assert impact.yards_per_completion_multiplier == 1.0
        assert impact.total_points_multiplier == 1.0

    def test_none_weather_returns_unity(self):
        impact = compute_weather_impact(None)
        assert impact.pass_attempt_multiplier == 1.0
        assert impact.total_points_multiplier == 1.0

    def test_wind_attenuation(self):
        """High winds reduce pass attempts, completion rate, YPC, and totals."""
        windy = WeatherConditions(
            temp_c=18.0,
            wind_kph=35.0,
            wind_gust_kph=50.0,
            precip_mm=0.0,
            is_indoor=False,
        )
        impact = compute_weather_impact(windy)
        assert impact.pass_attempt_multiplier < 1.0
        assert impact.rush_attempt_multiplier > 1.0
        assert impact.completion_prob_multiplier < 1.0
        assert impact.yards_per_completion_multiplier < 1.0
        assert impact.total_points_multiplier < 1.0
        assert impact.wind_attenuation < 1.0

    def test_precipitation_attenuation(self):
        """Precipitation reduces passing efficiency, increases rushing share, dampens totals."""
        rainy = WeatherConditions(
            temp_c=16.0,
            wind_kph=8.0,
            precip_mm=5.0,
            is_indoor=False,
        )
        impact = compute_weather_impact(rainy)
        assert impact.pass_attempt_multiplier < 1.0
        assert impact.rush_attempt_multiplier > 1.0
        assert impact.completion_prob_multiplier < 1.0
        assert impact.yards_per_completion_multiplier < 1.0
        assert impact.yards_before_contact_multiplier < 1.0
        assert impact.total_points_multiplier < 1.0
        assert impact.precip_attenuation < 1.0

    def test_extreme_cold_attenuation(self):
        """Sub-freezing temperatures dampen passing efficiency and totals."""
        freezing = WeatherConditions(
            temp_c=-8.0,
            wind_kph=10.0,
            precip_mm=0.0,
            is_indoor=False,
        )
        impact = compute_weather_impact(freezing)
        assert impact.completion_prob_multiplier < 1.0
        assert impact.yards_per_completion_multiplier < 1.0
        assert impact.total_points_multiplier < 1.0
        assert impact.temp_attenuation < 1.0

    def test_extreme_heat_attenuation(self):
        """Extreme heat slightly dampens pace and scoring."""
        scorching = WeatherConditions(
            temp_c=38.0,
            wind_kph=8.0,
            precip_mm=0.0,
            is_indoor=False,
        )
        impact = compute_weather_impact(scorching)
        assert impact.total_points_multiplier < 1.0
        assert impact.temp_attenuation < 1.0


class TestAdjustTeamPropsInputs:
    def test_adjusts_volume_and_efficiency(self, baseline_inputs):
        adverse_weather = WeatherConditions(
            temp_c=2.0,
            wind_kph=32.0,
            wind_gust_kph=48.0,
            precip_mm=4.0,
            is_indoor=False,
        )
        adjusted = adjust_team_props_inputs_for_weather(baseline_inputs, adverse_weather)

        # Passing attempts decrease
        assert adjusted.expected_pass_attempts < baseline_inputs.expected_pass_attempts
        # Rushing attempts increase (absorbing pass volume)
        assert adjusted.expected_rushing_attempts > baseline_inputs.expected_rushing_attempts
        # Efficiency decreases
        assert adjusted.completion_probability < baseline_inputs.completion_probability
        assert adjusted.yards_per_completion < baseline_inputs.yards_per_completion
        assert adjusted.yards_before_contact <= baseline_inputs.yards_before_contact
        # Data quality score is strictly preserved
        assert adjusted.data_quality_score == baseline_inputs.data_quality_score

    def test_indoor_weather_preserves_inputs(self, baseline_inputs):
        dome_weather = WeatherConditions(
            temp_c=-15.0,
            wind_kph=60.0,
            is_indoor=True,
        )
        adjusted = adjust_team_props_inputs_for_weather(baseline_inputs, dome_weather)
        assert adjusted == baseline_inputs

    def test_none_weather_preserves_inputs(self, baseline_inputs):
        adjusted = adjust_team_props_inputs_for_weather(baseline_inputs, None)
        assert adjusted == baseline_inputs

    def test_adjust_team_props_inputs_sets_points_multiplier(self, baseline_inputs):
        bad_weather = WeatherConditions(temp_c=0.0, wind_kph=40.0, precip_mm=5.0, is_indoor=False)
        adjusted = adjust_team_props_inputs_for_weather(baseline_inputs, bad_weather)
        assert adjusted.points_multiplier < 1.0
        # project_team_points uses the points_multiplier on the adjusted inputs
        pts_unadjusted = project_team_points(baseline_inputs)
        pts_adjusted = project_team_points(adjusted)
        assert pts_adjusted < pts_unadjusted

    def test_adjust_team_props_inputs_is_idempotent(self, baseline_inputs):
        bad_weather = WeatherConditions(temp_c=2.0, wind_kph=35.0, precip_mm=3.0, is_indoor=False)
        adjusted_once = adjust_team_props_inputs_for_weather(baseline_inputs, bad_weather)
        adjusted_twice = adjust_team_props_inputs_for_weather(adjusted_once, bad_weather)
        # Must not double-attenuate attempts or efficiency
        assert adjusted_twice.expected_pass_attempts == adjusted_once.expected_pass_attempts
        assert adjusted_twice.expected_rushing_attempts == adjusted_once.expected_rushing_attempts
        assert adjusted_twice.completion_probability == adjusted_once.completion_probability
        assert adjusted_twice.points_multiplier == adjusted_once.points_multiplier


class TestGameTotalsProjection:
    def test_project_game_total_combines_home_and_away(self, baseline_inputs):
        home_inputs = baseline_inputs
        away_inputs = default_team_props_inputs()

        proj_fair = project_game_total(home_inputs, away_inputs, None)
        assert proj_fair.home_projected_points > 0
        assert proj_fair.away_projected_points > 0
        assert proj_fair.projected_game_total == pytest.approx(
            proj_fair.home_projected_points + proj_fair.away_projected_points, abs=0.01
        )

    def test_adverse_weather_depresses_game_totals(self, baseline_inputs):
        home_inputs = baseline_inputs
        away_inputs = baseline_inputs

        fair_weather = WeatherConditions(temp_c=20.0, wind_kph=5.0, precip_mm=0.0, is_indoor=False)
        bad_weather = WeatherConditions(
            temp_c=0.0, wind_kph=40.0, wind_gust_kph=55.0, precip_mm=6.0, is_indoor=False
        )

        proj_fair = project_game_total(home_inputs, away_inputs, fair_weather)
        proj_bad = project_game_total(home_inputs, away_inputs, bad_weather)

        assert proj_bad.projected_game_total < proj_fair.projected_game_total
        assert proj_bad.home_projected_points < proj_fair.home_projected_points
        assert proj_bad.away_projected_points < proj_fair.away_projected_points

    def test_project_team_production_with_weather(self, baseline_inputs):
        weather = WeatherConditions(wind_kph=35.0, precip_mm=4.0, is_indoor=False)
        fair_prod = project_team_production("TEAM_PROP", baseline_inputs)
        bad_prod = project_team_production("TEAM_PROP", baseline_inputs, weather=weather)

        assert bad_prod.projected_team_receiving_yards < fair_prod.projected_team_receiving_yards
        assert bad_prod.projected_team_total_points < fair_prod.projected_team_total_points

    def test_project_team_points_with_weather(self, baseline_inputs):
        weather = WeatherConditions(wind_kph=30.0, precip_mm=3.0, is_indoor=False)
        fair_pts = project_team_points(baseline_inputs)
        bad_pts = project_team_points(baseline_inputs, weather=weather)
        assert bad_pts < fair_pts


class TestInvariants:
    def test_strictly_rejects_player_props(self, baseline_inputs):
        """Model 3 strictly prohibits player props; only team props and game totals allowed."""
        assert is_player_prop("passing_touchdowns") is True
        assert is_player_prop("rushing_yards_player") is True
        assert is_player_prop("team_rushing_yards") is False
        assert is_player_prop("team_total_points") is False

        with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
            validate_market_type("PLAYER_PROP")

        with pytest.raises(SchemaError):
            project_team_production("PLAYER_PROP", baseline_inputs)

    def test_edge_cases_and_zero_inputs(self):
        zero_inputs = TeamPropsInputs(
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
        extreme_weather = WeatherConditions(
            temp_c=-40.0,
            wind_kph=150.0,
            wind_gust_kph=200.0,
            precip_mm=80.0,
            is_indoor=False,
        )
        # Must not raise division by zero or negative outputs
        adj = adjust_team_props_inputs_for_weather(zero_inputs, extreme_weather)
        assert adj.expected_pass_attempts >= 0
        assert adj.expected_rushing_attempts >= 0
        assert adj.completion_probability >= 0

        proj = project_game_total(zero_inputs, zero_inputs, extreme_weather)
        assert proj.projected_game_total >= 0.0


@pytest.fixture
def weather_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    store.upsert_venue(
        conn,
        {
            "venue_id": "v1",
            "name": "Field 1",
            "city": "City 1",
            "state": "ST",
            "latitude": 35.0,
            "longitude": -97.0,
            "elevation_m": 300.0,
            "surface": "grass",
            "dome": 0,
            "capacity": 70000,
            "timezone": "America/Chicago",
        },
    )
    store.upsert_venue(
        conn,
        {
            "venue_id": "v2",
            "name": "Dome 2",
            "city": "City 2",
            "state": "ST",
            "latitude": 40.0,
            "longitude": -85.0,
            "elevation_m": 200.0,
            "surface": "turf",
            "dome": 1,
            "capacity": 60000,
            "timezone": "America/Indiana/Indianapolis",
        },
    )
    for tid, name in (("h1", "Home 1"), ("a1", "Away 1"), ("h2", "Home 2"), ("a2", "Away 2")):
        store.upsert_team(conn, {"team_id": tid, "school": name, "alias": name, "market": name})

    conn.execute(
        """INSERT INTO games (game_id, season, kickoff_utc, football_date, home_team_id,
                              away_team_id, venue_id, source, ingested_utc)
           VALUES ('g-1', 2026, '2026-09-05T20:00:00+00:00', '2026-09-05',
                   'h1', 'a1', 'v1', 'cfbd', 'x')"""
    )
    conn.execute(
        """INSERT INTO games (game_id, season, kickoff_utc, football_date, home_team_id,
                              away_team_id, venue_id, source, ingested_utc)
           VALUES ('g-2', 2026, '2026-09-05T23:00:00+00:00', '2026-09-05',
                   'h2', 'a2', 'v2', 'cfbd', 'x')"""
    )
    conn.execute(
        """INSERT INTO weather (game_id, as_of_utc, hours_to_kick, temp_c, wind_kph,
                                wind_gust_kph, precip_mm, precip_prob, humidity,
                                is_forecast, is_indoor, source)
           VALUES ('g-1', '2026-09-04T12:00:00+00:00',
                   32.0, 18.0, 12.0, 15.0, 0.0, 10.0, 50.0, 1, 0, 'open-meteo')"""
    )
    conn.execute(
        """INSERT INTO weather (game_id, as_of_utc, hours_to_kick, temp_c, wind_kph,
                                wind_gust_kph, precip_mm, precip_prob, humidity,
                                is_forecast, is_indoor, source)
           VALUES ('g-1', '2026-09-05T12:00:00+00:00',
                   8.0, 12.0, 32.0, 45.0, 4.0, 85.0, 80.0, 1, 0, 'open-meteo')"""
    )
    conn.execute(
        """INSERT INTO weather (game_id, as_of_utc, hours_to_kick, temp_c, wind_kph,
                                wind_gust_kph, precip_mm, precip_prob, humidity,
                                is_forecast, is_indoor, source)
           VALUES ('g-2', '2026-09-05T12:00:00+00:00',
                   11.0, NULL, NULL, NULL, NULL, NULL, NULL, 0, 1, 'open-meteo')"""
    )
    return conn


class TestFeaturesWeather:
    def test_get_weather_for_game_point_in_time(self, weather_db):
        # As of early date: gets first capture
        w_early = get_weather_for_game(weather_db, "g-1", as_of_utc="2026-09-04T18:00:00+00:00")
        assert w_early is not None
        assert w_early.wind_kph == pytest.approx(12.0)
        assert w_early.as_of_utc == "2026-09-04T12:00:00+00:00"

        # As of game day: gets updated capture
        w_late = get_weather_for_game(weather_db, "g-1", as_of_utc="2026-09-05T15:00:00+00:00")
        assert w_late is not None
        assert w_late.wind_kph == pytest.approx(32.0)
        assert w_late.precip_mm == pytest.approx(4.0)

    def test_load_weather_for_games(self, weather_db):
        weathers = load_weather_for_games(
            weather_db, ["g-1", "g-2"], as_of_utc="2026-09-05T15:00:00+00:00"
        )
        assert len(weathers) == 2
        assert weathers["g-1"].is_indoor is False
        assert weathers["g-2"].is_indoor is True

    def test_extract_weather_features(self):
        weather = WeatherConditions(
            temp_c=3.0,
            wind_kph=35.0,
            wind_gust_kph=45.0,
            precip_mm=3.0,
            humidity=75.0,
            is_indoor=False,
        )
        features = extract_weather_features(weather)
        assert features["temp_c"] == 3.0
        assert features["wind_kph"] == 35.0
        assert features["effective_wind_kph"] == 35.0
        assert features["is_indoor"] == 0.0
        assert features["is_extreme_wind"] == 1.0
        assert features["is_precipitation"] == 1.0
        assert features["wind_attenuation"] < 1.0
        assert features["precip_attenuation"] < 1.0

    def test_extract_weather_features_dome_zeroes_extreme_factors(self):
        hurricane_dome = WeatherConditions(
            temp_c=-10.0,
            wind_kph=80.0,
            wind_gust_kph=120.0,
            precip_mm=30.0,
            humidity=90.0,
            is_indoor=True,
        )
        features = extract_weather_features(hurricane_dome)
        assert features["is_indoor"] == 1.0
        assert features["effective_wind_kph"] == 0.0
        assert features["is_extreme_wind"] == 0.0
        assert features["is_precipitation"] == 0.0
        assert features["is_extreme_cold"] == 0.0
        assert features["is_extreme_heat"] == 0.0
        assert features["wind_attenuation"] == 1.0
        assert features["total_points_multiplier"] == 1.0

    def test_extract_weather_features_none(self):
        features = extract_weather_features(None)
        assert features["temp_c"] is None
        assert features["wind_attenuation"] == 1.0

    def test_project_slate_game_totals(self, weather_db):
        totals = project_slate_game_totals(weather_db, "2026-09-05")
        assert "g-1" in totals
        assert "g-2" in totals
        assert totals["g-1"].projected_game_total > 0
        assert totals["g-2"].projected_game_total > 0

    def test_build_weather_adjusted_inputs_for_slate(self, weather_db):
        inputs = build_weather_adjusted_inputs_for_slate(weather_db, "2026-09-05")
        assert "h1" in inputs
        assert "a1" in inputs
        assert "h2" in inputs
        assert "a2" in inputs


class TestScoringIntegration:
    def test_load_team_props_inputs_for_slate_with_weather(self, weather_db):
        inputs_no_weather = load_team_props_inputs_for_slate(
            weather_db, "2026-09-05", with_weather=False
        )
        inputs_with_weather = load_team_props_inputs_for_slate(
            weather_db, "2026-09-05", with_weather=True
        )

        # Team h1 has adverse weather in g-1 (wind 32 kph, rain 4mm)
        assert (
            inputs_with_weather["h1"].expected_pass_attempts
            < inputs_no_weather["h1"].expected_pass_attempts
        )
        assert (
            inputs_with_weather["h1"].completion_probability
            < inputs_no_weather["h1"].completion_probability
        )
        assert inputs_with_weather["h1"].points_multiplier < 1.0

        # Team h2 is in a dome in g-2 -> no weather attenuation
        assert (
            inputs_with_weather["h2"].expected_pass_attempts
            == inputs_no_weather["h2"].expected_pass_attempts
        )
        assert inputs_with_weather["h2"].points_multiplier == 1.0

    def test_score_slate_team_props_attenuates_team_total_points_with_weather(self, weather_db):
        weather_db.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, best_price, best_book, hold,
                prob_multiplicative, prob_shin)
               VALUES ('g-1', 'h1', 'team_total_points', 28.5, 'OVER',
                       '2026-09-05T12:00:00+00:00', 3, -110, -105,
                       'FanDuel', 0.045, 0.52, 0.52)"""
        )
        scores_no_weather = score_slate_team_props(
            weather_db, "2026-09-05", as_of_utc="2026-09-05T12:00:00+00:00", with_weather=False
        )
        scores_with_weather = score_slate_team_props(
            weather_db, "2026-09-05", as_of_utc="2026-09-05T12:00:00+00:00", with_weather=True
        )
        assert len(scores_no_weather) == 1
        assert len(scores_with_weather) == 1
        assert scores_with_weather[0].projected_value < scores_no_weather[0].projected_value

    def test_score_daily_slates_with_weather(self, weather_db):
        weather_db.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, best_price, best_book, hold,
                prob_multiplicative, prob_shin)
               VALUES ('g-1', 'h1', 'team_total_points', 28.5, 'OVER',
                       '2026-09-05T12:00:00+00:00', 3, -110, -105,
                       'FanDuel', 0.045, 0.52, 0.52)"""
        )
        scores = score_daily_slates(
            weather_db, ["2026-09-05"], as_of_utc="2026-09-05T12:00:00+00:00", with_weather=True
        )
        assert len(scores) == 1


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-09-05T20:00:00+00:00",
        "2026-09-05T21:00:00+00:00",
    ],
)
def test_weather_rejects_kickoff_and_postkickoff_snapshots(weather_db, stamp):
    weather_db.execute(
        "UPDATE weather SET as_of_utc=? WHERE game_id='g-1' AND wind_kph=32", (stamp,)
    )
    for cutoff in (None, "2026-09-06T00:00:00+00:00"):
        result = get_weather_for_game(weather_db, "g-1", as_of_utc=cutoff)
        assert result is not None
        assert result.wind_kph == 12.0


def test_weather_compares_offsets_chronologically(weather_db):
    weather_db.execute(
        "UPDATE weather SET as_of_utc='2026-09-05T09:00:00-04:00' "
        "WHERE game_id='g-1' AND wind_kph=32"
    )
    early = get_weather_for_game(weather_db, "g-1", as_of_utc="2026-09-05T12:00:00Z")
    late = get_weather_for_game(weather_db, "g-1", as_of_utc="2026-09-05T10:00:00-04:00")
    assert early is not None and early.wind_kph == 12.0
    assert late is not None and late.wind_kph == 32.0
    assert late.as_of_utc == "2026-09-05T13:00:00+00:00"


@pytest.mark.parametrize("stamp", ["bad", ""])
def test_weather_invalid_snapshot_timestamp_fails_closed(weather_db, stamp):
    from cfb_analytics.errors import LeakageError

    weather_db.execute(
        "UPDATE weather SET as_of_utc=? WHERE game_id='g-1' AND wind_kph=32", (stamp,)
    )
    with pytest.raises(LeakageError):
        load_weather_for_games(weather_db, ["g-1"])


def test_weather_invalid_cutoff_fails_closed(weather_db):
    from cfb_analytics.errors import LeakageError

    with pytest.raises(LeakageError):
        get_weather_for_game(weather_db, "g-1", as_of_utc="not a date")


def test_reanalysis_does_not_replace_forecast(weather_db):
    weather_db.execute("UPDATE weather SET is_forecast=0 WHERE wind_kph=32")
    result = get_weather_for_game(weather_db, "g-1", as_of_utc="2026-09-05T15:00:00Z")
    assert result is not None and result.wind_kph == 12.0


def test_default_weather_cutoff_uses_now(weather_db, monkeypatch):
    monkeypatch.setattr(
        "cfb_analytics.features.weather.utc_now_iso", lambda: "2026-09-04T18:00:00+00:00"
    )
    result = get_weather_for_game(weather_db, "g-1")
    assert result is not None and result.wind_kph == 12.0


def test_totals_respect_weather_opt_out(weather_db):
    before = project_slate_game_totals(weather_db, "2026-09-05", with_weather=False)
    after = project_slate_game_totals(weather_db, "2026-09-05", with_weather=True)
    assert after["g-1"].projected_game_total < before["g-1"].projected_game_total
    assert after["g-2"].projected_game_total == before["g-2"].projected_game_total
