"""Ingest CFBD betting lines into odds_snapshots and line_movement.

Why this exists alongside the Outlier ingest: CFBD is the source a *scheduled*
job can actually use. Its credential is a static API key with no expiry, no
browser session, and no OTP, whereas the Outlier access token lives 24 hours
and is refreshed by an interactive Playwright login. A daily cloud run on the
Outlier feed would authenticate for one day and 403 forever after.

CFBD also hands over the opening number next to the current one
(``spreadOpen``, ``overUnderOpen``), so movement is measurable from the first
run rather than after a week of accumulated captures.

What CFBD does **not** give: the price (juice) on a spread or total. Those rows
are stored with a null ``price_american`` and are therefore visible to the
movement layer but invisible to the devig layer, which requires two priced
sides. Moneylines do carry prices and devig normally.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.sources.outlier import OddsRow
from cfb_analytics.utils import american_to_decimal, football_date, stable_id, to_utc_iso

SOURCE = "cfbd"


class LinesClient(Protocol):
    def fetch_lines(
        self, year: int, *, week: int | None = ..., season_type: str = ...
    ) -> list[dict[str, Any]]: ...


@dataclass
class LinesIngestSummary:
    year: int
    week: int | None = None
    games_seen: int = 0
    games_matched: int = 0
    games_unmatched: int = 0
    odds_rows: int = 0
    movement_rows: int = 0
    providers: set[str] = field(default_factory=set)
    slates: set[str] = field(default_factory=set)

    def as_text(self) -> str:
        lines = [
            f"cfbd lines {self.year}" + (f" week {self.week}" if self.week else ""),
            f"  games in feed    : {self.games_seen}",
            f"  priced locally   : {self.games_matched}",
            f"  odds rows        : {self.odds_rows}",
            f"  movement rows    : {self.movement_rows}",
        ]
        if self.games_unmatched:
            # Expected, not a fault: CFBD publishes the whole season while the
            # store holds only what has been ingested. Worth showing so a real
            # name-matching regression is still visible, but phrased so it does
            # not read as a 90% failure rate.
            lines.append(
                f"  not in store     : {self.games_unmatched} "
                "(season-wide feed; only stored games can be priced)"
            )
        if self.providers:
            lines.append(f"  providers        : {', '.join(sorted(self.providers))}")
        if self.slates:
            lines.append(f"  slates touched   : {', '.join(sorted(self.slates))}")
        return "\n".join(lines)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _team_key(name: Any) -> str:
    return str(name or "").strip().upper()


def build_game_index(conn: sqlite3.Connection) -> dict[tuple[str, str, str], str]:
    """Map (football_date, home, away) to a local game_id.

    CFBD and Outlier name teams differently ("Ohio State" vs "OSU"), so the
    join goes through school name, alias and market, all upper-cased. A game
    that cannot be matched is counted and skipped, never guessed at.
    """
    index: dict[tuple[str, str, str], str] = {}
    rows = conn.execute(
        """SELECT g.game_id, g.football_date,
                  ht.school AS h_school, ht.alias AS h_alias, ht.market AS h_market,
                  at.school AS a_school, at.alias AS a_alias, at.market AS a_market
           FROM games g
           JOIN teams ht ON ht.team_id = g.home_team_id
           JOIN teams at ON at.team_id = g.away_team_id"""
    ).fetchall()
    for row in rows:
        home_names = {_team_key(row[k]) for k in ("h_school", "h_alias", "h_market") if row[k]}
        away_names = {_team_key(row[k]) for k in ("a_school", "a_alias", "a_market") if row[k]}
        for home in home_names:
            for away in away_names:
                index[(str(row["football_date"]), home, away)] = str(row["game_id"])
    return index


def parse_line_rows(
    game_id: str,
    provider_row: Mapping[str, Any],
    captured_utc: str,
) -> list[OddsRow]:
    """One provider's numbers for one game, as OddsRows.

    Spread and total rows carry a null price because CFBD does not publish the
    juice. They are stored so movement can be measured; the devig layer skips
    them because it requires two priced sides.
    """
    provider = str(provider_row.get("provider") or "").strip().upper()
    if not provider:
        return []

    rows: list[OddsRow] = []

    def add(market: str, side: str, line: float | None, price: int | None) -> None:
        rows.append(
            OddsRow(
                game_id=game_id, market_id=None, book=provider, market=market,
                side=side, line=line, price_american=price,
                price_decimal=american_to_decimal(price) if price is not None else None,
                is_primary=True, captured_utc=captured_utc, source=SOURCE,
            )
        )

    home_ml = _int(provider_row.get("homeMoneyline"))
    away_ml = _int(provider_row.get("awayMoneyline"))
    if home_ml is not None and away_ml is not None:
        add("ML", "HOME", 0.0, home_ml)
        add("ML", "AWAY", 0.0, away_ml)

    spread = _float(provider_row.get("spread"))
    if spread is not None:
        # CFBD states the spread from the HOME perspective.
        add("SPREAD", "HOME", spread, None)
        add("SPREAD", "AWAY", -spread, None)

    total = _float(provider_row.get("overUnder"))
    if total is not None:
        add("TOTAL", "OVER", total, None)
        add("TOTAL", "UNDER", total, None)

    return rows


def parse_movement_rows(
    game_id: str,
    provider_row: Mapping[str, Any],
    as_of_utc: str,
) -> list[dict[str, Any]]:
    """Open-to-current movement, straight from CFBD's own opening fields."""
    provider = str(provider_row.get("provider") or "").strip().upper()
    if not provider:
        return []

    out: list[dict[str, Any]] = []

    def movement(market: str, side: str, open_line: float | None, current_line: float | None):
        if open_line is None or current_line is None:
            return
        magnitude = current_line - open_line
        out.append({
            "game_id": game_id, "market": market, "side": side, "as_of_utc": as_of_utc,
            "open_line": open_line, "open_price": None,
            "current_line": current_line, "current_price": None,
            "move_magnitude": magnitude,
            "move_direction": "toward" if magnitude < 0 else ("away" if magnitude > 0 else "flat"),
            # No ticket/money percentages exist in CFBD either, so the RLM
            # basis stays honest about what it is.
            "rlm_flag": 0, "rlm_basis": "line_only",
        })

    spread_open = _float(provider_row.get("spreadOpen"))
    spread_now = _float(provider_row.get("spread"))
    movement("SPREAD", "HOME", spread_open, spread_now)
    if spread_open is not None and spread_now is not None:
        movement("SPREAD", "AWAY", -spread_open, -spread_now)

    total_open, total_now = (
        _float(provider_row.get("overUnderOpen")), _float(provider_row.get("overUnder"))
    )
    movement("TOTAL", "OVER", total_open, total_now)
    movement("TOTAL", "UNDER", total_open, total_now)

    return out


def ingest_lines(
    conn: sqlite3.Connection,
    client: LinesClient,
    year: int,
    *,
    week: int | None = None,
    season_type: str = "regular",
    captured_utc: str | None = None,
) -> LinesIngestSummary:
    from cfb_analytics.utils import utc_now_iso

    stamp = captured_utc or utc_now_iso()
    summary = LinesIngestSummary(year=year, week=week)
    index = build_game_index(conn)

    for game in client.fetch_lines(year, week=week, season_type=season_type):
        summary.games_seen += 1
        kickoff = to_utc_iso(game.get("startDate"))
        if kickoff is None:
            summary.games_unmatched += 1
            continue
        slate = football_date(kickoff)
        key = (slate, _team_key(game.get("homeTeam")), _team_key(game.get("awayTeam")))
        game_id = index.get(key)
        if game_id is None:
            summary.games_unmatched += 1
            continue

        summary.games_matched += 1
        summary.slates.add(slate)

        providers = game.get("lines")
        if not isinstance(providers, list):
            raise SchemaError(f"CFBD /lines game {game.get('id')} has no 'lines' array")

        for provider_row in providers:
            if not isinstance(provider_row, dict):
                continue
            name = str(provider_row.get("provider") or "").strip().upper()
            if name:
                summary.providers.add(name)
            summary.odds_rows += store.insert_odds(
                conn, parse_line_rows(game_id, provider_row, stamp)
            )
            summary.movement_rows += _write_movement(
                conn, parse_movement_rows(game_id, provider_row, stamp)
            )

    return summary


def _write_movement(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT OR REPLACE INTO line_movement
           (game_id, market, side, as_of_utc, open_line, open_price,
            current_line, current_price, move_magnitude, move_direction,
            rlm_flag, rlm_basis)
           VALUES (:game_id, :market, :side, :as_of_utc, :open_line, :open_price,
                   :current_line, :current_price, :move_magnitude, :move_direction,
                   :rlm_flag, :rlm_basis)""",
        rows,
    )
    return len(rows)


__all__ = [
    "LinesIngestSummary",
    "build_game_index",
    "ingest_lines",
    "parse_line_rows",
    "parse_movement_rows",
    "stable_id",
]
