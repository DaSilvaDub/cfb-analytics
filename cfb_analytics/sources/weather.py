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
FORECAST_HOURLY_FIELDS = HOURLY_FIELDS
ARCHIVE_HOURLY_FIELDS = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
)
# Open-Meteo's forecast horizon. Past it, there is no forecast to have.
FORECAST_HORIZON_DAYS = 16
# ERA5 reanalysis publishes on roughly a five-day delay.
ARCHIVE_LAG_DAYS = 5


# Geolocation fallback mapping for FBS stadiums (normalized venue name -> (lat, lon, is_dome))
FALLBACK_STADIUM_COORDINATES: dict[str, tuple[float, float, bool]] = {
    "rose bowl": (34.1613, -118.1676, False),
    "michigan stadium": (42.2658, -83.7487, False),
    "beaver stadium": (40.8122, -77.8561, False),
    "ohio stadium": (40.0016, -83.0197, False),
    "kyle field": (30.6102, -96.3407, False),
    "neyland stadium": (35.9550, -83.9250, False),
    "tiger stadium": (30.4120, -91.1838, False),
    "bryant denny stadium": (33.2083, -87.5504, False),
    "bryant denny": (33.2083, -87.5504, False),
    "darrell k royal texas memorial stadium": (30.2837, -97.7323, False),
    "sanford stadium": (33.9498, -83.3734, False),
    "ben hill griffin stadium": (29.6500, -82.3486, False),
    "jordan hare stadium": (32.6022, -85.4897, False),
    "notre dame stadium": (41.6984, -86.2339, False),
    "los angeles memorial coliseum": (34.0141, -118.2878, False),
    "la coliseum": (34.0141, -118.2878, False),
    "husky stadium": (47.6503, -122.3016, False),
    "autzen stadium": (44.0583, -123.0686, False),
    "camp randall stadium": (43.0700, -89.4125, False),
    "kinnick stadium": (41.6586, -91.5511, False),
    "doak campbell stadium": (30.4381, -84.3044, False),
    "lane stadium": (37.2197, -80.4178, False),
    "kenan memorial stadium": (35.9070, -79.0478, False),
    "carter finley stadium": (35.7958, -78.7106, False),
    "hard rock stadium": (25.9580, -80.2389, False),
    "albertsons stadium": (43.6028, -116.1958, False),
    "rice eccles stadium": (40.7597, -111.8489, False),
    "lavell edwards stadium": (40.2575, -111.6544, False),
    "folsom field": (40.0094, -105.2669, False),
    "falcon stadium": (38.9969, -104.8436, False),
    "michie stadium": (41.3914, -73.9639, False),
    "navy marine corps memorial stadium": (38.9847, -76.5075, False),
    "mercedes benz stadium": (33.7554, -84.4008, True),
    "att stadium": (32.7473, -97.0945, True),
    "at t stadium": (32.7473, -97.0945, True),
    "lucas oil stadium": (39.7601, -86.1639, True),
    "allegiant stadium": (36.0908, -115.1833, True),
    "caesars superdome": (29.9511, -90.0812, True),
    "superdome": (29.9511, -90.0812, True),
    "nrg stadium": (29.6847, -95.4107, True),
    "state farm stadium": (33.5276, -112.2626, True),
    "carrier dome": (43.0362, -76.1365, True),
    "jma wireless dome": (43.0362, -76.1365, True),
    "alamodome": (29.4169, -98.4789, True),
    "the dome at americas center": (38.6328, -90.1886, True),
    "kibbie dome": (46.7261, -117.0172, True),
    "dakotadome": (42.7911, -96.9317, True),
    "fargodome": (46.9000, -96.8000, True),
    "alerus center": (47.9042, -97.0722, True),
    "walkup skydome": (35.1806, -111.6542, True),
    "holt arena": (42.8714, -112.4319, True),
}


def normalize_venue_name(name: str) -> str:
    """Normalize venue name for dictionary lookup."""
    import re

    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    return " ".join(cleaned.split())


def _stadium_lookup_keys(venue_name: str | None) -> list[str]:
    """Generate normalized search keys for stadium lookups."""
    if not venue_name:
        return []
    import re

    norm = normalize_venue_name(venue_name)
    keys = [norm]

    # Strip leading "the "
    if norm.startswith("the "):
        keys.append(norm[4:])

    # Strip trailing venue type words
    stripped = re.sub(r"\s+(stadium|field|dome|center|coliseum|arena)$", "", norm).strip()
    if stripped and stripped != norm:
        keys.append(stripped)
        if stripped.startswith("the "):
            keys.append(stripped[4:])

    # Add trailing "stadium" if not already ending in a venue word
    if not re.search(r"\b(stadium|field|dome|center|coliseum|arena)$", norm):
        keys.append(f"{norm} stadium")

    return list(dict.fromkeys(keys))


def resolve_venue_coordinates(
    conn: Any | None = None,
    *,
    venue_id: str | None = None,
    venue_name: str | None = None,
    fallback_lookup: Any | None = None,
) -> tuple[float | None, float | None, bool | None]:
    """Resolve venue latitude, longitude, and dome status.

    Checks:
    1. Direct DB lookup on `venues` table by venue_id if conn is given.
    2. Name lookup on `venues` via store.resolve_venue_id_by_name if venue_id
       is missing or has NULL coordinates.
    3. Caller-provided fallback_lookup mapping.
    4. In-memory FALLBACK_STADIUM_COORDINATES table.

    Returns:
        (latitude, longitude, is_dome)
    """
    lat: float | None = None
    lon: float | None = None
    dome: bool | None = None

    if conn is not None and venue_id:
        row = conn.execute(
            "SELECT latitude, longitude, dome, name FROM venues WHERE venue_id = ?",
            (venue_id,),
        ).fetchone()
        if row is not None:
            lat = _float(row["latitude"])
            lon = _float(row["longitude"])
            dome = bool(row["dome"]) if row["dome"] is not None else None
            if not venue_name and row["name"]:
                venue_name = str(row["name"])
            if lat is not None and lon is not None:
                return lat, lon, dome

    if conn is not None and venue_name and (lat is None or lon is None):
        try:
            from cfb_analytics.ingest import store

            resolved_id = store.resolve_venue_id_by_name(conn, venue_name)
            if resolved_id and resolved_id != venue_id:
                row = conn.execute(
                    "SELECT latitude, longitude, dome FROM venues WHERE venue_id = ?",
                    (resolved_id,),
                ).fetchone()
                if row is not None:
                    lat = _float(row["latitude"])
                    lon = _float(row["longitude"])
                    dome = bool(row["dome"]) if row["dome"] is not None else None
                    if lat is not None and lon is not None:
                        return lat, lon, dome
        except Exception:
            pass

    # Check caller-provided fallback_lookup
    if fallback_lookup:
        candidates = []
        if venue_name:
            candidates.append(venue_name)
        candidates.extend(_stadium_lookup_keys(venue_name))
        if venue_id:
            candidates.append(venue_id)
        for key in candidates:
            val = fallback_lookup.get(key)
            if val is None and isinstance(key, str):
                val = fallback_lookup.get(normalize_venue_name(key))
            if val is None and isinstance(key, str) and hasattr(fallback_lookup, "items"):
                norm_target = normalize_venue_name(key)
                for fb_k, fb_v in fallback_lookup.items():
                    if isinstance(fb_k, str) and normalize_venue_name(fb_k) == norm_target:
                        val = fb_v
                        break
            if val is not None and isinstance(val, (tuple, list)):
                f_lat = _float(val[0])
                f_lon = _float(val[1])
                f_dome = bool(val[2]) if len(val) >= 3 else (dome or False)
                return f_lat, f_lon, f_dome

    # Check built-in FALLBACK_STADIUM_COORDINATES
    candidates = _stadium_lookup_keys(venue_name)
    if venue_id:
        candidates.append(venue_id)
    for key in candidates:
        norm_key = normalize_venue_name(key)
        if norm_key in FALLBACK_STADIUM_COORDINATES:
            f_lat, f_lon, f_dome = FALLBACK_STADIUM_COORDINATES[norm_key]
            return f_lat, f_lon, f_dome

    return lat, lon, dome


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

    def _url(
        self, base: str, latitude: float, longitude: float, day: str, *, archive: bool = False
    ) -> str:
        from urllib.parse import urlencode

        fields = ARCHIVE_HOURLY_FIELDS if archive else FORECAST_HOURLY_FIELDS
        params = {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "hourly": ",".join(fields),
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
        payload = self.http.get_payload(self._url(base, latitude, longitude, day, archive=archive))
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
