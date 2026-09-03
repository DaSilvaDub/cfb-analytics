"""CFBD client for historical teams, venues, and games.

Contract verified against the official CFBD API docs on 2026-09-02:

* Base URL: https://api.collegefootballdata.com
* Auth: Authorization: Bearer <key>
* Teams: GET /teams/fbs?year=...
* Venues: GET /venues
* Games: GET /games?year=...&seasonType=...&classification=...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode

from cfb_analytics import config
from cfb_analytics.errors import SchemaError
from cfb_analytics.sources.http import HttpClient
from cfb_analytics.utils import football_date, to_utc_iso


class PayloadClient(Protocol):
    def get_payload(self, url: str) -> Any: ...


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        value = None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"CFBD field {field!r} was not an integer: {value!r}") from exc


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SchemaError(f"CFBD field {field!r} was not a boolean: {value!r}")


def _as_str(value: Any, field: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise SchemaError(f"CFBD field {field!r} was missing or blank")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


@dataclass(frozen=True)
class CFBDBackfillSummary:
    seasons: int
    venues: int
    teams: int
    team_seasons: int
    aliases: int
    games: int

    def as_text(self) -> str:
        return (
            f"CFBD backfill wrote {self.games} games, {self.team_seasons} team seasons, "
            f"{self.teams} teams, {self.aliases} aliases, and {self.venues} venues "
            f"across {self.seasons} season(s)."
        )


class CFBDClient:
    def __init__(self, http: PayloadClient | None = None) -> None:
        conf = config.sources()["cfbd"]
        self.base_url = str(conf["base_url"]).rstrip("/")
        if http is not None:
            self.http = http
            return
        key = config.cfbd_api_key()
        self.http = HttpClient(
            name="cfbd",
            timeout_seconds=int(conf.get("timeout_seconds", 30)),
            max_retries=int(conf.get("max_retries", 3)),
            cache_ttl_seconds=int(conf.get("cache_ttl_seconds", 86400)),
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            auth_error_hint="the CFBD API key is missing, expired, or lacks access",
        )

    def _url(self, path: str, **params: Any) -> str:
        filtered = {key: value for key, value in params.items() if value is not None}
        if not filtered:
            return f"{self.base_url}{path}"
        return f"{self.base_url}{path}?{urlencode(filtered)}"

    def _get_rows(self, path: str, **params: Any) -> list[dict[str, Any]]:
        payload = self.http.get_payload(self._url(path, **params))
        if not isinstance(payload, list):
            raise SchemaError(f"CFBD {path} did not return a JSON array")
        rows = [row for row in payload if isinstance(row, dict)]
        if len(rows) != len(payload):
            raise SchemaError(f"CFBD {path} contained a non-object row")
        return rows

    def fetch_venues(self) -> list[dict[str, Any]]:
        return self._get_rows("/venues")

    def fetch_fbs_teams(self, year: int) -> list[dict[str, Any]]:
        return self._get_rows("/teams/fbs", year=year)

    def fetch_games(
        self, year: int, *, season_type: str = "both", classification: str = "fbs"
    ) -> list[dict[str, Any]]:
        return self._get_rows(
            "/games",
            year=year,
            seasonType=season_type,
            classification=classification,
        )


def parse_venue(row: dict[str, Any]) -> dict[str, Any] | None:
    venue_id = row.get("id")
    name = _string_or_none(row.get("name"))
    if venue_id in (None, "") or name is None:
        return None
    grass = row.get("grass")
    surface = None if grass is None else ("grass" if _as_bool(grass, "grass") else "artificial")
    capacity = row.get("capacity")
    dome = row.get("dome")
    return {
        "venue_id": str(_as_int(venue_id, "id")),
        "name": name,
        "city": _string_or_none(row.get("city")),
        "state": _string_or_none(row.get("state")),
        "latitude": _float_or_none(row.get("latitude")),
        "longitude": _float_or_none(row.get("longitude")),
        # CFBD's current OpenAPI declares elevation as a string but does not
        # document its unit. Do not put an unverified value in a metres column.
        "elevation_m": None,
        "surface": surface,
        "dome": None if dome is None else 1 if _as_bool(dome, "dome") else 0,
        "capacity": _as_int(capacity, "capacity") if capacity is not None else None,
        "timezone": _string_or_none(row.get("timezone")),
    }


def parse_team(row: dict[str, Any]) -> dict[str, Any]:
    school = _as_str(row.get("school"), "school")
    cfbd_id = _as_int(row.get("id"), "id")
    raw_location = row.get("location")
    location = raw_location if isinstance(raw_location, dict) else {}
    venue_id = location.get("id")
    abbreviation = _string_or_none(row.get("abbreviation"))
    return {
        "team_id": f"cfbd:{cfbd_id}",
        "cfbd_id": cfbd_id,
        "school": school,
        "alias": abbreviation or school,
        "market": school,
        "conference": _string_or_none(row.get("conference")),
        "classification": _string_or_none(row.get("classification")),
        "venue_id": str(venue_id) if venue_id not in (None, "") else None,
    }


def parse_team_season(row: dict[str, Any], *, year: int) -> dict[str, Any]:
    team = parse_team(row)
    raw_location = row.get("location")
    location = raw_location if isinstance(raw_location, dict) else {}
    venue_id = location.get("id")
    return {
        "team_id": team["team_id"],
        "season": year,
        "source": "cfbd",
        "conference": _string_or_none(row.get("conference")),
        "division": _string_or_none(row.get("division")),
        "classification": _string_or_none(row.get("classification")),
        "venue_id": str(venue_id) if venue_id not in (None, "") else None,
    }


def parse_team_aliases(row: dict[str, Any]) -> list[dict[str, Any]]:
    cfbd_id = _as_int(row.get("id"), "id")
    raw = row.get("alternateNames")
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for value in raw:
        alias = _string_or_none(value)
        if not alias or alias in seen:
            continue
        seen.add(alias)
        rows.append(
            {
                "team_id": f"cfbd:{cfbd_id}",
                "source": "cfbd",
                "alias": alias,
                "alias_type": "alternate_name",
            }
        )
    return rows


def parse_game(row: dict[str, Any]) -> dict[str, Any]:
    kickoff = to_utc_iso(row.get("startDate"))
    if kickoff is None:
        raise SchemaError("CFBD game had an unparseable startDate")
    completed = _as_bool(row.get("completed"), "completed")
    home_points = (
        _as_int(row["homePoints"], "homePoints")
        if row.get("homePoints") is not None
        else None
    )
    away_points = (
        _as_int(row["awayPoints"], "awayPoints")
        if row.get("awayPoints") is not None
        else None
    )
    return {
        "game_id": f"cfbd:{_as_int(row.get('id'), 'id')}",
        "season": _as_int(row.get("season"), "season"),
        "week": _as_int(row.get("week"), "week"),
        "season_type": _as_str(row.get("seasonType"), "seasonType"),
        "kickoff_utc": kickoff,
        "football_date": football_date(kickoff),
        "neutral_site": 1 if _as_bool(row.get("neutralSite"), "neutralSite") else 0,
        "conference_game": 1 if _as_bool(row.get("conferenceGame"), "conferenceGame") else 0,
        "home_team_id": f"cfbd:{_as_int(row.get('homeId'), 'homeId')}",
        "away_team_id": f"cfbd:{_as_int(row.get('awayId'), 'awayId')}",
        "venue_name": _string_or_none(row.get("venue")),
        "status": "completed" if completed else "scheduled",
        "home_points": home_points,
        "away_points": away_points,
        "completed": 1 if completed else 0,
        "source": "cfbd",
    }


def parse_game_team(row: dict[str, Any], *, side: str) -> dict[str, Any]:
    """Build the team dimension carried by a game row.

    `/games?classification=fbs` includes games involving an FBS team, so the
    opponent can be FCS and absent from `/teams/fbs`. Materialising both sides
    prevents foreign-key failures and preserves the actual historical slate.
    """
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")
    return parse_team(
        {
            "id": row.get(f"{side}Id"),
            "school": row.get(f"{side}Team"),
            "abbreviation": None,
            "conference": row.get(f"{side}Conference"),
            "classification": row.get(f"{side}Classification"),
            "location": None,
        }
    )
