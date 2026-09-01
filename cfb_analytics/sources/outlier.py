"""Outlier client for NCAA football (league token ``NCAAFB``).

Two structural facts about this feed, both verified against the live API on
2026-08-31 and both easy to get wrong:

1. **``outcomes[].books`` is not parallel to ``outcomes[].odds``.** In a sampled
   moneyline market ``books[0]`` was ``FLIFF`` while ``odds[0]["book"]`` was
   ``FANATICS``. Every odds entry names its own book, so the price/book pairing
   is read from inside the entry and the sibling ``books`` list is treated as a
   coverage hint only. Zipping the two by index mis-attributes prices silently.

2. **A proposition spans several market rows per event.** The sample averaged
   three ``MONEYLINE`` rows and ~5.6 ``SPREAD`` rows per event, each carrying a
   different subset of books; the ~19-20 book coverage is the union across rows,
   not the contents of any single one. Consensus therefore unions all rows for a
   proposition and de-duplicates on (book, side, line).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from cfb_analytics import config
from cfb_analytics.errors import SchemaError, UnknownLeagueError
from cfb_analytics.sources import session
from cfb_analytics.sources.http import HttpClient
from cfb_analytics.utils import (
    american_to_decimal,
    football_date,
    stable_id,
    to_utc_iso,
    weekday_matches_feed,
)

# Verified 2026-08-31: NCAAF / CFB / FBS / CFP / NCAAFOOTBALL all return HTTP 502.
LEAGUE_TOKEN = "NCAAFB"
# A token known to resolve, used to tell "unknown league" apart from "API is down"
# when the API answers both with 502.
CONTROL_LEAGUE_TOKEN = "NFL"

PROPOSITION_TO_MARKET = {
    "MONEYLINE": "ML",
    "SPREAD": "SPREAD",
    "TOTAL": "TOTAL",
}


@dataclass(frozen=True)
class OddsRow:
    """One book's price on one side of one market, ready for the store."""

    game_id: str
    market_id: str | None
    book: str
    market: str
    side: str | None
    line: float | None
    price_american: int | None
    price_decimal: float | None
    is_primary: bool
    captured_utc: str

    @property
    def snapshot_id(self) -> str:
        return stable_id(
            self.game_id, self.book, self.market, self.side,
            self.line, self.price_american, self.captured_utc,
        )


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip().replace("+", ""))
    except (TypeError, ValueError):
        return None


class OutlierClient:
    def __init__(self, http: HttpClient | None = None, league: str = LEAGUE_TOKEN) -> None:
        self.league = league
        if http is not None:
            self.http = http
            self.base_url = ""
            return
        conf = config.sources()["outlier"]
        self.base_url = str(conf["base_url"]).rstrip("/")
        self.http = HttpClient(
            name="outlier",
            timeout_seconds=int(conf.get("timeout_seconds", 30)),
            max_retries=int(conf.get("max_retries", 3)),
            cache_ttl_seconds=int(conf.get("cache_ttl_seconds", 300)),
            headers=session.build_headers(session.load_storage_state()),
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def fetch_schedule(self, league: str | None = None) -> list[dict[str, Any]]:
        token = league or self.league
        try:
            payload = self.http.get_json(self._url(f"/sportsdata/leagues/{token}/schedule"))
        except UnknownLeagueError:
            # 502 is ambiguous here, so check a token known to work before
            # telling the operator their league is unsupported.
            try:
                self.http.get_json(
                    self._url(f"/sportsdata/leagues/{CONTROL_LEAGUE_TOKEN}/schedule")
                )
            except Exception as exc:  # control also failed -> the API is unwell
                raise UnknownLeagueError(
                    f"League {token!r} failed and the {CONTROL_LEAGUE_TOKEN} control also "
                    f"failed - treat this as an API outage, not an unsupported league: {exc}"
                ) from exc
            raise UnknownLeagueError(
                f"League token {token!r} is not recognised by Outlier "
                f"(the {CONTROL_LEAGUE_TOKEN} control resolved, so the API is up). "
                f"The verified NCAA football token is {LEAGUE_TOKEN!r}."
            ) from None
        events = payload.get("events")
        if not isinstance(events, list):
            raise SchemaError(f"Outlier schedule for {token} had no 'events' list")
        return [event for event in events if isinstance(event, dict)]

    def fetch_event_markets(
        self, event_id: str, market_type: str = "GAMELINE"
    ) -> list[dict[str, Any]]:
        payload = self.http.get_json(
            self._url(f"/sportsdata/events/{event_id}/markets?marketType={quote(market_type)}")
        )
        markets = payload.get("markets")
        return [m for m in markets if isinstance(m, dict)] if isinstance(markets, list) else []

    def fetch_team_injuries(self, team_id: str, league: str | None = None) -> list[dict[str, Any]]:
        token = league or self.league
        payload = self.http.get_json(
            self._url(f"/sportsdata/leagues/{token}/teams/{team_id}/injuries")
        )
        players = payload.get("players")
        return [p for p in players if isinstance(p, dict)] if isinstance(players, list) else []


def parse_odds_rows(
    game_id: str,
    markets: list[dict[str, Any]],
    captured_utc: str,
) -> list[OddsRow]:
    """Flatten gameline market rows into one OddsRow per (book, side, line).

    De-duplicates across the several market rows a proposition spans, keeping the
    first observation of each (book, market, side, line) triple.
    """
    seen: set[tuple[str, str, str | None, float | None]] = set()
    rows: list[OddsRow] = []

    for market in markets:
        market_code = PROPOSITION_TO_MARKET.get(str(market.get("proposition") or "").upper())
        if market_code is None:
            continue  # DOUBLE_RESULT, WINNING_MARGIN, MONEYLINE_THREE_WAY: out of scope
        market_id = market.get("marketId")
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            continue

        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            side = outcome.get("position") or outcome.get("label")
            side = str(side).upper() if side is not None else None
            line = _as_float(outcome.get("line"))
            is_primary = bool(outcome.get("primary"))

            for entry in outcome.get("odds") or []:
                if not isinstance(entry, dict):
                    continue
                # The book is read from the entry, never by index into `books`.
                book = str(entry.get("book") or "").strip().upper()
                if not book:
                    continue
                key = (book, market_code, side, line)
                if key in seen:
                    continue
                seen.add(key)
                decimal_price = _as_float(entry.get("decimal"))
                american = _as_int(entry.get("american"))
                if decimal_price is None and american is not None:
                    decimal_price = american_to_decimal(american)
                rows.append(
                    OddsRow(
                        game_id=game_id,
                        market_id=str(market_id) if market_id else None,
                        book=book,
                        market=market_code,
                        side=side,
                        line=line,
                        price_american=american,
                        price_decimal=decimal_price,
                        is_primary=is_primary,
                        captured_utc=captured_utc,
                    )
                )
    return rows


def parse_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalise one schedule event. Raises rather than defaulting a kickoff time."""
    event_id = event.get("eventId")
    home = event.get("home")
    away = event.get("away")
    if not event_id or not isinstance(home, dict) or not isinstance(away, dict):
        raise SchemaError(f"Outlier event missing eventId/home/away: keys={sorted(event)[:12]}")

    kickoff = to_utc_iso(event.get("scheduledTime"))
    if kickoff is None:
        raise SchemaError(f"Outlier event {event_id} has an unparseable scheduledTime")

    return {
        "game_id": str(event_id),
        "kickoff_utc": kickoff,
        # The slate this game belongs to (US Eastern date), NOT kickoff_utc[:10].
        "football_date": football_date(kickoff),
        "day_of_week": event.get("dayOfWeek"),
        "weekday_agrees": weekday_matches_feed(kickoff, event.get("dayOfWeek")),
        "season": event.get("season"),
        "status": event.get("status"),
        "network": event.get("network"),
        "venue_name": event.get("venue") if isinstance(event.get("venue"), str) else None,
        "home": {
            "team_id": str(home.get("teamId")),
            "school": home.get("name"),
            "alias": home.get("alias"),
            "market": home.get("market"),
        },
        "away": {
            "team_id": str(away.get("teamId")),
            "school": away.get("name"),
            "alias": away.get("alias"),
            "market": away.get("market"),
        },
    }


def parse_injury_rows(
    game_id: str,
    team_id: str,
    players: list[dict[str, Any]],
    as_of_utc: str,
) -> list[dict[str, Any]]:
    """Normalise the injury feed.

    Player names are deliberately not stored: ``playerId`` is the join key and
    the models need position and designation, not identity.
    """
    rows = []
    for player in players:
        injury = player.get("injury")
        injury = injury if isinstance(injury, dict) else {}
        designation = str(injury.get("status") or "").strip() or None
        player_id = player.get("playerId")
        if not player_id:
            continue
        position = str(player.get("position") or "").strip().upper() or None
        rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "player_id": str(player_id),
                "as_of_utc": as_of_utc,
                "position": position,
                "position_group": _position_group(position),
                "designation": designation,
                "injury_type": injury.get("injury"),
                "return_date": injury.get("returnDate"),
                "last_updated_utc": to_utc_iso(injury.get("lastUpdated")),
                "has_news": 1 if injury.get("hasNews") else 0,
                "source": "outlier",
            }
        )
    return rows


_POSITION_GROUPS = {
    "QB": "QB",
    "RB": "SKILL", "FB": "SKILL", "WR": "SKILL", "TE": "SKILL",
    "OL": "OL", "OT": "OL", "OG": "OL", "C": "OL", "G": "OL", "T": "OL",
    "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
    "K": "ST", "P": "ST", "LS": "ST",
}


def _position_group(position: str | None) -> str | None:
    if not position:
        return None
    return _POSITION_GROUPS.get(position, "OTHER")
