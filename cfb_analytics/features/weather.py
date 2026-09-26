"""Weather feature extraction, database retrieval, and slate-level weather adjustments.

Integrates Open-Meteo forecasts/reanalysis with Model 3 team props and game totals.
Strictly adheres to team props and game totals only (NO player props).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from cfb_analytics.errors import LeakageError
from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.models.team_props import (
    GameTotalsProjection,
    TeamPropsInputs,
    WeatherConditions,
    compute_weather_impact,
    project_game_total,
)
from cfb_analytics.utils import utc_now_iso


def _weather_timestamp(value: Any, *, field: str) -> datetime:
    """Parse availability timestamps without losing subsecond precision."""
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise LeakageError(f"Invalid {field}: {value!r}") from exc
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def get_weather_for_game(
    conn: sqlite3.Connection,
    game_id: str,
    *,
    as_of_utc: str | None = None,
) -> WeatherConditions | None:
    """Return the latest forecast known by the cutoff and strictly before kickoff."""
    return load_weather_for_games(conn, [game_id], as_of_utc=as_of_utc).get(game_id)


def load_weather_for_games(
    conn: sqlite3.Connection,
    game_ids: list[str],
    *,
    as_of_utc: str | None = None,
) -> dict[str, WeatherConditions]:
    """Read pregame forecasts using chronological, per-game availability guards.

    Reanalysis is retrospective and is never an input to pregame predictions.
    Indoor records have no outdoor conditions and need not be forecasts.
    Missing/invalid timestamps fail closed rather than bypassing the cutoff.
    """
    cutoff = _weather_timestamp(
        as_of_utc if as_of_utc is not None else utc_now_iso(), field="weather cutoff"
    )
    if not game_ids:
        return {}
    placeholders = ",".join("?" for _ in game_ids)
    rows = conn.execute(
        f"""SELECT w.*, g.kickoff_utc
            FROM weather w JOIN games g ON g.game_id = w.game_id
            WHERE w.game_id IN ({placeholders})""",
        game_ids,
    ).fetchall()
    latest: dict[str, tuple[datetime, sqlite3.Row]] = {}
    for row in rows:
        stamp = _weather_timestamp(row["as_of_utc"], field="weather.as_of_utc")
        kickoff = _weather_timestamp(row["kickoff_utc"], field="game.kickoff_utc")
        reader = AsOfReader(game_id=str(row["game_id"]), kickoff_utc=kickoff.isoformat())
        if stamp > cutoff or stamp >= kickoff:
            continue
        reader.check(stamp, what="weather")
        if not row["is_indoor"] and not row["is_forecast"]:
            continue
        game_id = str(row["game_id"])
        if game_id not in latest or stamp > latest[game_id][0]:
            latest[game_id] = (stamp, row)

    result: dict[str, WeatherConditions] = {}
    for game_id, (stamp, row) in latest.items():
        result[game_id] = WeatherConditions(
            temp_c=row["temp_c"],
            wind_kph=row["wind_kph"],
            wind_gust_kph=row["wind_gust_kph"],
            wind_dir_deg=row["wind_dir_deg"],
            precip_mm=row["precip_mm"],
            precip_prob=row["precip_prob"],
            humidity=row["humidity"],
            is_indoor=bool(row["is_indoor"]),
            is_forecast=bool(row["is_forecast"]),
            as_of_utc=stamp.isoformat(),
        )
    return result


def extract_weather_features(weather: WeatherConditions | None) -> dict[str, float | None]:
    """Extract tabular feature representation of game weather for downstream models."""
    if weather is None:
        return {
            "temp_c": None,
            "wind_kph": None,
            "wind_gust_kph": None,
            "effective_wind_kph": None,
            "precip_mm": None,
            "precip_prob": None,
            "humidity": None,
            "is_indoor": 0.0,
            "is_extreme_wind": 0.0,
            "is_precipitation": 0.0,
            "is_extreme_cold": 0.0,
            "is_extreme_heat": 0.0,
            "wind_attenuation": 1.0,
            "precip_attenuation": 1.0,
            "temp_attenuation": 1.0,
            "total_points_multiplier": 1.0,
        }

    if weather.is_indoor:
        return {
            "temp_c": weather.temp_c,
            "wind_kph": weather.wind_kph,
            "wind_gust_kph": weather.wind_gust_kph,
            "effective_wind_kph": 0.0,
            "precip_mm": weather.precip_mm,
            "precip_prob": weather.precip_prob,
            "humidity": weather.humidity,
            "is_indoor": 1.0,
            "is_extreme_wind": 0.0,
            "is_precipitation": 0.0,
            "is_extreme_cold": 0.0,
            "is_extreme_heat": 0.0,
            "wind_attenuation": 1.0,
            "precip_attenuation": 1.0,
            "temp_attenuation": 1.0,
            "total_points_multiplier": 1.0,
        }

    impact = compute_weather_impact(weather)
    wind = weather.wind_kph or 0.0
    gust = weather.wind_gust_kph or 0.0
    eff_wind = max(wind, gust * 0.75)
    precip = weather.precip_mm or 0.0
    temp = weather.temp_c

    return {
        "temp_c": weather.temp_c,
        "wind_kph": weather.wind_kph,
        "wind_gust_kph": weather.wind_gust_kph,
        "effective_wind_kph": round(eff_wind, 2),
        "precip_mm": weather.precip_mm,
        "precip_prob": weather.precip_prob,
        "humidity": weather.humidity,
        "is_indoor": 0.0,
        "is_extreme_wind": 1.0 if eff_wind >= 30.0 else 0.0,
        "is_precipitation": 1.0 if precip >= 0.5 or (weather.precip_prob or 0.0) >= 60.0 else 0.0,
        "is_extreme_cold": 1.0 if (temp is not None and temp <= 0.0) else 0.0,
        "is_extreme_heat": 1.0 if (temp is not None and temp >= 32.0) else 0.0,
        "wind_attenuation": impact.wind_attenuation,
        "precip_attenuation": impact.precip_attenuation,
        "temp_attenuation": impact.temp_attenuation,
        "total_points_multiplier": impact.total_points_multiplier,
    }


def build_weather_adjusted_inputs_for_slate(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    season: int | None = None,
    as_of_utc: str | None = None,
) -> dict[str, TeamPropsInputs]:
    """Load and weather-adjust TeamPropsInputs for all teams playing on slate."""
    from cfb_analytics.scoring import load_team_props_inputs_for_slate

    return load_team_props_inputs_for_slate(
        conn,
        slate_date,
        season=season,
        as_of_utc=as_of_utc,
        with_weather=True,
    )


def project_slate_game_totals(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    season: int | None = None,
    as_of_utc: str | None = None,
    with_weather: bool = True,
) -> dict[str, GameTotalsProjection]:
    """Project game totals for all games on a slate using team props and weather."""
    from cfb_analytics.scoring import build_team_props_inputs_from_db

    as_of = _weather_timestamp(
        as_of_utc if as_of_utc is not None else utc_now_iso(), field="totals cutoff"
    )
    games = conn.execute(
        """SELECT game_id, home_team_id, away_team_id, season, kickoff_utc
           FROM games WHERE football_date = ?""",
        (slate_date,),
    ).fetchall()

    game_ids = [str(g["game_id"]) for g in games]
    weather_by_game = (
        load_weather_for_games(conn, game_ids, as_of_utc=as_of.isoformat()) if with_weather else {}
    )

    projections: dict[str, GameTotalsProjection] = {}
    for g in games:
        game_id = str(g["game_id"])
        home_id = str(g["home_team_id"]) if g["home_team_id"] else None
        away_id = str(g["away_team_id"]) if g["away_team_id"] else None
        game_season = season if season is not None else (int(g["season"]) if g["season"] else None)

        if not home_id or not away_id:
            continue

        kickoff = _weather_timestamp(g["kickoff_utc"], field="game.kickoff_utc")
        game_cutoff = min(as_of, kickoff - timedelta(seconds=1)).isoformat()
        home_inputs = build_team_props_inputs_from_db(
            conn,
            home_id,
            opponent_team_id=away_id,
            season=game_season,
            as_of_utc=game_cutoff,
        )
        away_inputs = build_team_props_inputs_from_db(
            conn,
            away_id,
            opponent_team_id=home_id,
            season=game_season,
            as_of_utc=game_cutoff,
        )

        weather = weather_by_game.get(game_id)
        proj = project_game_total(home_inputs, away_inputs, weather)
        projections[game_id] = proj

    return projections
