"""Fetch and store weather for scheduled games.

Indoor games are recorded as ``is_indoor=1`` with null conditions rather than
skipped silently. "Roof" and "we could not fetch" are different facts, and the
totals model needs to tell them apart: a dome is a *known* absence of wind,
while a missing fetch is an unknown that should reduce confidence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cfb_analytics.errors import CfbAnalyticsError
from cfb_analytics.sources.weather import (
    SOURCE,
    WeatherClient,
    choose_endpoint,
    observation_at,
)
from cfb_analytics.utils import utc_now_iso


@dataclass
class WeatherIngestSummary:
    games_considered: int = 0
    written: int = 0
    indoor: int = 0
    no_venue: int = 0
    no_coordinates: int = 0
    outside_window: int = 0
    failed: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def as_text(self) -> str:
        lines = [
            "weather ingest",
            f"  games considered : {self.games_considered}",
            f"  written          : {self.written}",
        ]
        for label, value in (
            ("indoor (no weather)", self.indoor),
            ("no venue linked", self.no_venue),
            ("venue has no lat/lon", self.no_coordinates),
            ("outside forecast/archive window", self.outside_window),
            ("fetch failed", self.failed),
        ):
            if value:
                lines.append(f"  {label:<32}: {value}")
        return "\n".join(lines)


def _games_needing_weather(
    conn: sqlite3.Connection, slate_dates: list[str]
) -> list[dict[str, Any]]:
    if not slate_dates:
        return []
    placeholders = ",".join("?" for _ in slate_dates)
    rows = conn.execute(
        f"""SELECT g.game_id, g.kickoff_utc, g.venue_id, g.venue_name,
                   v.latitude, v.longitude, v.dome
            FROM games g
            LEFT JOIN venues v ON v.venue_id = g.venue_id
            WHERE g.football_date IN ({placeholders})
            ORDER BY g.kickoff_utc""",
        slate_dates,
    ).fetchall()
    return [dict(row) for row in rows]


def ingest_weather(
    conn: sqlite3.Connection,
    slate_dates: list[str],
    *,
    client: WeatherClient | None = None,
    now: datetime | None = None,
    as_of_utc: str | None = None,
) -> WeatherIngestSummary:
    summary = WeatherIngestSummary()
    games = _games_needing_weather(conn, slate_dates)
    summary.games_considered = len(games)
    if not games:
        return summary

    moment = now or datetime.now(UTC)
    stamp = as_of_utc or utc_now_iso()
    weather_client = client or WeatherClient()

    for game in games:
        if game.get("dome"):
            _write(conn, game["game_id"], stamp, game["kickoff_utc"], moment,
                   observation=None, is_indoor=True, is_forecast=False)
            summary.indoor += 1
            continue
        if not game.get("venue_id"):
            summary.no_venue += 1
            continue
        latitude, longitude = game.get("latitude"), game.get("longitude")
        if latitude is None or longitude is None:
            summary.no_coordinates += 1
            continue

        endpoint = choose_endpoint(game["kickoff_utc"], now=moment)
        if endpoint is None:
            # Beyond the forecast horizon, or inside the reanalysis lag.
            summary.outside_window += 1
            continue

        try:
            hourly = weather_client.fetch_hourly(
                float(latitude), float(longitude),
                str(game["kickoff_utc"])[:10],
                archive=endpoint == "archive",
            )
        except CfbAnalyticsError as exc:
            summary.failed += 1
            key = type(exc).__name__
            summary.reasons[key] = summary.reasons.get(key, 0) + 1
            continue

        observation = observation_at(hourly, game["kickoff_utc"])
        if observation is None:
            summary.failed += 1
            continue

        _write(conn, game["game_id"], stamp, game["kickoff_utc"], moment,
               observation=observation, is_indoor=False,
               is_forecast=endpoint == "forecast")
        summary.written += 1

    return summary


def _write(
    conn: sqlite3.Connection,
    game_id: str,
    as_of_utc: str,
    kickoff_utc: str,
    now: datetime,
    *,
    observation: Any,
    is_indoor: bool,
    is_forecast: bool,
) -> None:
    kickoff = datetime.fromisoformat(kickoff_utc)
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC)
    hours_to_kick = (kickoff - now).total_seconds() / 3600.0

    conn.execute(
        """INSERT OR REPLACE INTO weather
           (game_id, as_of_utc, hours_to_kick, temp_c, wind_kph, wind_gust_kph,
            wind_dir_deg, precip_mm, precip_prob, humidity, is_forecast,
            is_indoor, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            game_id, as_of_utc, hours_to_kick,
            getattr(observation, "temp_c", None),
            getattr(observation, "wind_kph", None),
            getattr(observation, "wind_gust_kph", None),
            getattr(observation, "wind_dir_deg", None),
            getattr(observation, "precip_mm", None),
            getattr(observation, "precip_prob", None),
            getattr(observation, "humidity", None),
            1 if is_forecast else 0,
            1 if is_indoor else 0,
            SOURCE,
        ),
    )
