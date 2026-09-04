"""Open-Meteo weather, keyed to stadium coordinates.

No API key, no session, no rate-limit headaches -- which is why this works in
the scheduled job where the Outlier feed cannot.

Two endpoints, chosen by whether kickoff is in the future:

* **forecast** for upcoming games. Horizon is ~16 days; beyond that Open-Meteo
  returns nothing and the game is reported unforecastable rather than guessed.
* **ERA5 archive** for past games, used to backfill history for the totals
  model. The archive lags roughly 5 days behind real time, so recent games sit
  in a gap where the forecast has expired and the reanalysis has not landed;
  that gap is reported, not interpolated.

Units are pinned explicitly (celsius, km/h, mm) because Open-Meteo's defaults
are locale-independent but silent -- a units change upstream would otherwise
turn 15 km/h of wind into 15 mph without anything failing.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from cfb_analytics import config
from cfb_analytics.errors import SchemaError

SOURCE = "open-meteo"

HOURLY_FIELDS = (
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
)
# Open-Meteo's forecast horizon. Past it, there is no forecast to have.
FORECAST_HORIZON_DAYS = 16
# ERA5 reanalysis publishes on roughly a five-day delay.
ARCHIVE_LAG_DAYS = 5


class PayloadClient(Protocol):
    def get_payload(self, url: str) -> Any: ...


@dataclass(frozen=True)
class WeatherObservation:
    """Conditions at the hour nearest kickoff."""

    temp_c: float | None
    wind_kph: float | None
    wind_gust_kph: float | None
    wind_dir_deg: float | None
    precip_mm: float | None
    precip_prob: float | None
    humidity: float | None
    is_forecast: bool
    observed_hour_utc: str


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


class WeatherClient:
    def __init__(self, http: PayloadClient | None = None) -> None:
        conf = config.sources()["weather"]
        self.forecast_url = str(conf["base_url"])
        self.archive_url = str(conf["archive_url"])
        if http is not None:
            self.http = http
            return
        from cfb_analytics.sources.http import HttpClient

        self.http = HttpClient(
            name="weather",
            timeout_seconds=int(conf.get("timeout_seconds", 30)),
            max_retries=int(conf.get("max_retries", 3)),
            cache_ttl_seconds=int(conf.get("cache_ttl_seconds", 3600)),
            headers={"Accept": "application/json"},
        )

    def _url(self, base: str, latitude: float, longitude: float, day: str) -> str:
        from urllib.parse import urlencode

        params = {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "hourly": ",".join(HOURLY_FIELDS),
            "start_date": day,
            "end_date": day,
            "timezone": "UTC",
            # Pinned, not defaulted -- see the module docstring.
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }
        return f"{base}?{urlencode(params)}"

    def fetch_hourly(
        self, latitude: float, longitude: float, day: str, *, archive: bool
    ) -> dict[str, list[Any]]:
        base = self.archive_url if archive else self.forecast_url
        payload = self.http.get_payload(self._url(base, latitude, longitude, day))
        if not isinstance(payload, dict):
            raise SchemaError("Open-Meteo did not return a JSON object")
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise SchemaError("Open-Meteo response has no 'hourly.time' series")
        return hourly


def choose_endpoint(kickoff_utc: str, *, now: datetime | None = None) -> str | None:
    """Which Open-Meteo endpoint can answer for this kickoff.

    Returns 'forecast', 'archive', or None when neither can -- too far ahead
    for a forecast, or too recent for the reanalysis. None is a real answer and
    the caller records it as unavailable rather than substituting a guess.
    """
    moment = now or datetime.now(UTC)
    kickoff = datetime.fromisoformat(kickoff_utc)
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC)
    delta_days = (kickoff - moment).total_seconds() / 86400.0

    if delta_days >= 0:
        return "forecast" if delta_days <= FORECAST_HORIZON_DAYS else None
    return "archive" if -delta_days >= ARCHIVE_LAG_DAYS else None


def observation_at(hourly: dict[str, list[Any]], kickoff_utc: str) -> WeatherObservation | None:
    """The hourly sample nearest kickoff.

    Open-Meteo returns whole days on an hourly grid; a 19:30 kickoff has no
    exact row, so the closest hour is taken. If the nearest sample is more than
    90 minutes away the series does not actually cover kickoff and None is
    returned rather than reporting a distant hour as if it were game time.
    """
    times = hourly.get("time") or []
    if not times:
        return None

    kickoff = datetime.fromisoformat(kickoff_utc)
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC)

    parsed = []
    for stamp in times:
        try:
            moment = datetime.fromisoformat(str(stamp))
        except ValueError:
            return None
        parsed.append(moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment)

    index = bisect.bisect_left(parsed, kickoff)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(parsed)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs((parsed[i] - kickoff).total_seconds()))
    if abs((parsed[best] - kickoff).total_seconds()) > timedelta(minutes=90).total_seconds():
        return None

    def series(name: str) -> float | None:
        values = hourly.get(name)
        if not isinstance(values, list) or best >= len(values):
            return None
        return _float(values[best])

    return WeatherObservation(
        temp_c=series("temperature_2m"),
        wind_kph=series("wind_speed_10m"),
        wind_gust_kph=series("wind_gusts_10m"),
        wind_dir_deg=series("wind_direction_10m"),
        precip_mm=series("precipitation"),
        precip_prob=series("precipitation_probability"),
        humidity=series("relative_humidity_2m"),
        is_forecast=True,  # corrected by the caller, which knows the endpoint
        observed_hour_utc=parsed[best].isoformat(),
    )
