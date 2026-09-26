"""Historical CFBD /lines backfill with leakage-safe synthetic capture stamps.

CFBD's historical ``/lines`` response carries opening and closing *prices* but
no capture clock. This module invents two stamps that are always strictly
before kickoff so ``AsOfReader`` and ``build_market_for_slate`` can treat the
rows like any other pregame odds:

* **open**  -- ``kickoff_utc - 7 days`` (week-ahead proxy); fields
  ``spreadOpen`` / ``overUnderOpen`` only. Moneyline open is not in the CFBD
  schema, so it is never invented.
* **close** -- ``kickoff_utc - 60 seconds``; fields ``spread`` / ``overUnder`` /
  ``homeMoneyline`` / ``awayMoneyline``.

``source='cfbd_historical'`` keeps these rows distinct from the live
``source='cfbd'`` ingest path in ``cfbd_lines.py``, which is intentionally
untouched.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from cfb_analytics.errors import LeakageError, SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_lines import (
    LinesClient,
    LinesIngestSummary,
    _float,
    _int,
    _team_key,
    _write_movement,
    build_game_index,
    parse_movement_rows,
)
from cfb_analytics.sources.outlier import OddsRow
from cfb_analytics.utils import american_to_decimal, football_date, to_utc_iso

SOURCE = "cfbd_historical"

# Open is a week-ahead proxy; close is one minute before kickoff. Both must
# remain strictly before kickoff (tested).
OPEN_LOOKBACK = timedelta(days=7)
CLOSE_LOOKBACK = timedelta(seconds=60)


def _parse_kickoff(kickoff_utc: str) -> datetime:
    stamp = to_utc_iso(kickoff_utc)
    if stamp is None:
        raise LeakageError(
            f"Cannot stamp historical lines: kickoff_utc={kickoff_utc!r} is not parseable. "
            "An unparseable kickoff cannot be proven pre-kickoff."
        )
    return datetime.fromisoformat(stamp)


def synthetic_open_utc(kickoff_utc: str) -> str:
    """Week-ahead open stamp, clamped so it is always strictly before kickoff."""
    kickoff = _parse_kickoff(kickoff_utc)
    stamp = kickoff - OPEN_LOOKBACK
    if stamp >= kickoff:
        stamp = kickoff - timedelta(seconds=1)
    if stamp >= kickoff:
        raise LeakageError(
            f"Cannot synthesise an open stamp before kickoff {kickoff.isoformat()}"
        )
    return stamp.astimezone(UTC).isoformat(timespec="seconds")


def synthetic_close_utc(kickoff_utc: str) -> str:
    """Close stamp at kickoff minus 60 seconds. Never at or after kickoff."""
    kickoff = _parse_kickoff(kickoff_utc)
    stamp = kickoff - CLOSE_LOOKBACK
    if stamp >= kickoff:
        raise LeakageError(
            f"Close stamp {stamp.isoformat()} is not strictly before kickoff "
            f"{kickoff.isoformat()}"
        )
    return stamp.astimezone(UTC).isoformat(timespec="seconds")


def _odds_row(
    game_id: str,
    book: str,
    market: str,
    side: str,
    line: float | None,
    price: int | None,
    captured_utc: str,
) -> OddsRow:
    return OddsRow(
        game_id=game_id,
        market_id=None,
        book=book,
        market=market,
        side=side,
        line=line,
        price_american=price,
        price_decimal=american_to_decimal(price) if price is not None else None,
        is_primary=True,
        captured_utc=captured_utc,
        source=SOURCE,
    )


def parse_open_rows(
    game_id: str,
    provider_row: Mapping[str, Any],
    captured_utc: str,
) -> list[OddsRow]:
    """Open-role rows: spreadOpen / overUnderOpen only. No moneyline open."""
    provider = str(provider_row.get("provider") or "").strip().upper()
    if not provider:
        return []

    rows: list[OddsRow] = []
    spread_open = _float(provider_row.get("spreadOpen"))
    if spread_open is not None:
        rows.append(_odds_row(game_id, provider, "SPREAD", "HOME", spread_open, None, captured_utc))
        rows.append(
            _odds_row(game_id, provider, "SPREAD", "AWAY", -spread_open, None, captured_utc)
        )

    total_open = _float(provider_row.get("overUnderOpen"))
    if total_open is not None:
        rows.append(_odds_row(game_id, provider, "TOTAL", "OVER", total_open, None, captured_utc))
        rows.append(_odds_row(game_id, provider, "TOTAL", "UNDER", total_open, None, captured_utc))

    return rows


def parse_close_rows(
    game_id: str,
    provider_row: Mapping[str, Any],
    captured_utc: str,
) -> list[OddsRow]:
    """Close-role rows: spread / total / both moneylines when present."""
    provider = str(provider_row.get("provider") or "").strip().upper()
    if not provider:
        return []

    rows: list[OddsRow] = []

    home_ml = _int(provider_row.get("homeMoneyline"))
    away_ml = _int(provider_row.get("awayMoneyline"))
    if home_ml is not None and away_ml is not None:
        rows.append(_odds_row(game_id, provider, "ML", "HOME", 0.0, home_ml, captured_utc))
        rows.append(_odds_row(game_id, provider, "ML", "AWAY", 0.0, away_ml, captured_utc))

    spread = _float(provider_row.get("spread"))
    if spread is not None:
        rows.append(_odds_row(game_id, provider, "SPREAD", "HOME", spread, None, captured_utc))
        rows.append(_odds_row(game_id, provider, "SPREAD", "AWAY", -spread, None, captured_utc))

    total = _float(provider_row.get("overUnder"))
    if total is not None:
        rows.append(_odds_row(game_id, provider, "TOTAL", "OVER", total, None, captured_utc))
        rows.append(_odds_row(game_id, provider, "TOTAL", "UNDER", total, None, captured_utc))

    return rows


@dataclass
class HistoricalLinesIngestSummary(LinesIngestSummary):
    """Extends the live summary with open/close bookkeeping."""

    open_odds_rows: int = 0
    close_odds_rows: int = 0
    games_with_ml_close: int = 0
    season_type: str = "regular"

    def as_text(self) -> str:
        base = super().as_text().replace("cfbd lines", "cfbd historical lines", 1)
        extra = [
            f"  open odds rows   : {self.open_odds_rows}",
            f"  close odds rows  : {self.close_odds_rows}",
            f"  games w/ ML close: {self.games_with_ml_close}",
            f"  source           : {SOURCE}",
        ]
        return base + "\n" + "\n".join(extra)


def ingest_historical_lines(
    conn: sqlite3.Connection,
    client: LinesClient,
    year: int,
    *,
    week: int | None = None,
    season_type: str = "regular",
) -> HistoricalLinesIngestSummary:
    """Fetch one week (or season) of CFBD lines and dual-stamp into odds_snapshots.

    Idempotent via ``insert_odds`` snapshot hashes. Does not touch live
    ``source='cfbd'`` rows.
    """
    summary = HistoricalLinesIngestSummary(year=year, week=week, season_type=season_type)
    index = build_game_index(conn)
    games_with_ml: set[str] = set()

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

        open_stamp = synthetic_open_utc(kickoff)
        close_stamp = synthetic_close_utc(kickoff)
        # Belt-and-suspenders: never write a stamp that AsOfReader would refuse.
        if not (open_stamp < kickoff and close_stamp < kickoff):
            raise LeakageError(
                f"Historical stamp invariant failed for game {game_id}: "
                f"open={open_stamp} close={close_stamp} kickoff={kickoff}"
            )

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

            open_rows = parse_open_rows(game_id, provider_row, open_stamp)
            close_rows = parse_close_rows(game_id, provider_row, close_stamp)
            written_open = store.insert_odds(conn, open_rows)
            written_close = store.insert_odds(conn, close_rows)
            summary.open_odds_rows += written_open
            summary.close_odds_rows += written_close
            summary.odds_rows += written_open + written_close

            if any(r.market == "ML" for r in close_rows):
                games_with_ml.add(game_id)

            # Movement uses CFBD's own open fields; as_of is the close stamp so
            # the row describes open→close as of the closing capture.
            summary.movement_rows += _write_movement(
                conn, parse_movement_rows(game_id, provider_row, close_stamp)
            )

    summary.games_with_ml_close = len(games_with_ml)
    return summary


def backfill_historical_lines(
    conn: sqlite3.Connection,
    client: LinesClient,
    year: int,
    *,
    weeks: range | None = None,
    season_type: str = "regular",
) -> HistoricalLinesIngestSummary:
    """Week loop for one season. Default weeks 1–15 (regular season)."""
    week_range = weeks if weeks is not None else range(1, 16)
    totals = HistoricalLinesIngestSummary(year=year, season_type=season_type)

    for week in week_range:
        part = ingest_historical_lines(
            conn, client, year, week=week, season_type=season_type
        )
        totals.games_seen += part.games_seen
        totals.games_matched += part.games_matched
        totals.games_unmatched += part.games_unmatched
        totals.odds_rows += part.odds_rows
        totals.open_odds_rows += part.open_odds_rows
        totals.close_odds_rows += part.close_odds_rows
        totals.movement_rows += part.movement_rows
        totals.providers |= part.providers
        totals.slates |= part.slates

    # Count after the full week loop so idempotent re-runs still report coverage.
    totals.games_with_ml_close = len(_ml_close_game_ids(conn, year))
    totals.week = None
    return totals


def _ml_close_game_ids(conn: sqlite3.Connection, year: int) -> list[str]:
    rows = conn.execute(
        """SELECT DISTINCT o.game_id
           FROM odds_snapshots o
           JOIN games g ON g.game_id = o.game_id
           WHERE g.season = ? AND o.source = ? AND o.market = 'ML'
             AND o.price_american IS NOT NULL""",
        (year, SOURCE),
    ).fetchall()
    return [str(r[0]) for r in rows]


def football_dates_for_season(conn: sqlite3.Connection, year: int) -> list[str]:
    """Distinct Eastern slate dates for a season, ordered."""
    rows = conn.execute(
        """SELECT DISTINCT football_date FROM games
           WHERE season = ? AND football_date IS NOT NULL
           ORDER BY football_date""",
        (year,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def rebuild_market_for_season(
    conn: sqlite3.Connection,
    year: int,
    *,
    min_books_for_consensus: int | None = None,
) -> dict[str, Any]:
    """Rebuild market_consensus for every Eastern football_date in ``year``.

    ``min_books_for_consensus`` overrides the live floor for CFBD-historical
    baseline scoring only (documented as ``historical_cfbd_min_books``). Pass
    ``None`` to use ``settings.market.min_books_for_consensus``.
    """
    from cfb_analytics.features.build_market import build_market_for_slate

    dates = football_dates_for_season(conn, year)
    totals = {
        "year": year,
        "slates": 0,
        "games": 0,
        "consensus_rows": 0,
        "movement_rows": 0,
        "unpriced_groups": 0,
        "min_books_for_consensus": min_books_for_consensus,
    }
    for slate in dates:
        summary = build_market_for_slate(
            conn,
            slate,
            min_books_for_consensus=min_books_for_consensus,
        )
        totals["slates"] += 1
        totals["games"] += summary.games
        totals["consensus_rows"] += summary.consensus_rows
        totals["movement_rows"] += summary.movement_rows
        totals["unpriced_groups"] += summary.unpriced_groups
    return totals


def probe_lines_coverage(
    client: LinesClient,
    year: int,
    *,
    weeks: range | None = None,
    season_type: str = "regular",
) -> dict[str, Any]:
    """Read-only tally of CFBD /lines coverage for a season (no DB writes)."""
    week_range = weeks if weeks is not None else range(1, 16)
    providers: set[str] = set()
    games = 0
    games_with_ml_both = 0
    games_with_spread_open = 0
    games_with_total_open = 0
    provider_game_counts: dict[str, int] = {}

    for week in week_range:
        for game in client.fetch_lines(year, week=week, season_type=season_type):
            games += 1
            lines = game.get("lines") if isinstance(game.get("lines"), list) else []
            has_ml = False
            has_spread_open = False
            has_total_open = False
            for provider_row in lines:
                if not isinstance(provider_row, dict):
                    continue
                name = str(provider_row.get("provider") or "").strip().upper()
                if not name:
                    continue
                providers.add(name)
                provider_game_counts[name] = provider_game_counts.get(name, 0) + 1
                if (
                    _int(provider_row.get("homeMoneyline")) is not None
                    and _int(provider_row.get("awayMoneyline")) is not None
                ):
                    has_ml = True
                if _float(provider_row.get("spreadOpen")) is not None:
                    has_spread_open = True
                if _float(provider_row.get("overUnderOpen")) is not None:
                    has_total_open = True
            if has_ml:
                games_with_ml_both += 1
            if has_spread_open:
                games_with_spread_open += 1
            if has_total_open:
                games_with_total_open += 1

    return {
        "year": year,
        "weeks": [week_range.start, week_range.stop - 1],
        "games": games,
        "games_with_ml_both_sides": games_with_ml_both,
        "games_with_spread_open": games_with_spread_open,
        "games_with_total_open": games_with_total_open,
        "providers": sorted(providers),
        "provider_game_counts": dict(sorted(provider_game_counts.items())),
        "ml_coverage_rate": (games_with_ml_both / games) if games else 0.0,
    }


__all__ = [
    "SOURCE",
    "HistoricalLinesIngestSummary",
    "backfill_historical_lines",
    "football_dates_for_season",
    "ingest_historical_lines",
    "parse_close_rows",
    "parse_open_rows",
    "probe_lines_coverage",
    "rebuild_market_for_season",
    "synthetic_close_utc",
    "synthetic_open_utc",
]
