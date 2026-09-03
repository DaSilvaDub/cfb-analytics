"""Leakage-labelled CFBD fundamentals backfill."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Any

from cfb_analytics.errors import SchemaError, SourceError
from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import (
    CFBDClient,
    parse_advanced_rows,
    parse_elo_rating,
    parse_returning_production,
    parse_sp_rating,
    parse_srs_rating,
    parse_talent,
    parse_team,
)


@dataclass(frozen=True)
class FundamentalsSummary:
    seasons: int
    endpoints: int
    ratings: int
    advanced: int
    returning: int
    talent: int
    filtered_sp: int
    filtered_srs: int
    filtered_elo: int
    filtered_advanced: int
    filtered_returning: int
    filtered_talent: int

    def as_text(self) -> str:
        total_filtered = (
            self.filtered_sp
            + self.filtered_srs
            + self.filtered_elo
            + self.filtered_advanced
            + self.filtered_returning
            + self.filtered_talent
        )
        return (
            f"CFBD fundamentals wrote {self.ratings} rating snapshots, "
            f"{self.advanced} advanced-stat snapshots, {self.returning} returning-"
            f"production rows, and {self.talent} talent rows across {self.seasons} "
            f"season(s) and {self.endpoints} healthy endpoints. "
            f"Filtered {total_filtered} non-FBS rows."
        )


def _fetch(
    run: store.RunRecorder,
    endpoint: str,
    call: Callable[[], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        rows = call()
    except (SchemaError, SourceError) as exc:
        run.record_health("cfbd", endpoint, ok=False, detail=str(exc))
        raise
    run.record_health("cfbd", endpoint, ok=True, rows=len(rows))
    return rows


def _name_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _team_resolver(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    ambiguous: set[str] = set()
    for raw in rows:
        team = parse_team(raw)
        names = [raw.get("school"), raw.get("abbreviation"), *(raw.get("alternateNames") or [])]
        for name in names:
            key = _name_key(name)
            if not key:
                continue
            prior = resolved.get(key)
            if prior is not None and prior != team["team_id"]:
                ambiguous.add(key)
            else:
                resolved[key] = team["team_id"]
    for key in ambiguous:
        resolved.pop(key, None)
    return resolved


def _resolve_rows(
    rows: Iterable[dict[str, Any]],
    resolver: dict[str, str],
    *,
    endpoint: str,
    allow_unresolved: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    resolved: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        name = row.pop("team_name")
        team_id = resolver.get(_name_key(name))
        if team_id is None:
            if allow_unresolved:
                skipped += 1
                continue
            raise SchemaError(f"CFBD {endpoint} team {name!r} did not resolve to an FBS ID")
        resolved.append({**row, "team_id": team_id})
    return resolved, skipped


def _cutoffs(conn: sqlite3.Connection, season: int) -> tuple[dict[int, str], str]:
    rows = conn.execute(
        """SELECT week, kickoff_utc FROM games
           WHERE source = 'cfbd' AND season = ? AND season_type = 'regular'
             AND week IS NOT NULL""",
        (season,),
    ).fetchall()
    if not rows:
        raise SchemaError(f"No CFBD regular-season games stored for {season}; backfill games first")

    by_week: dict[int, list[datetime]] = {}
    for row in rows:
        stamp = datetime.fromisoformat(str(row["kickoff_utc"]))
        by_week.setdefault(int(row["week"]), []).append(stamp)
    weekly = {
        week: (max(stamps) + timedelta(days=1)).isoformat() for week, stamps in by_week.items()
    }
    all_games = conn.execute(
        "SELECT kickoff_utc FROM games WHERE source = 'cfbd' AND season = ?",
        (season,),
    ).fetchall()
    season_final = (
        max(datetime.fromisoformat(str(row["kickoff_utc"])) for row in all_games)
        + timedelta(days=1)
    ).isoformat()
    return weekly, season_final


def _check_season(rows: Iterable[dict[str, Any]], year: int, endpoint: str) -> None:
    wrong = sorted({int(row["season"]) for row in rows if int(row["season"]) != year})
    if wrong:
        raise SchemaError(f"CFBD {endpoint} requested {year} but returned seasons {wrong}")


def backfill_fundamentals(
    conn: sqlite3.Connection,
    client: CFBDClient,
    *,
    start_year: int,
    end_year: int,
) -> FundamentalsSummary:
    if start_year > end_year:
        raise SchemaError("start_year must be less than or equal to end_year")

    rating_count = advanced_count = returning_count = talent_count = 0
    filtered_sp = filtered_srs = filtered_elo = filtered_advanced = 0
    filtered_returning = filtered_talent = endpoints = 0
    command = f"backfill-fundamentals --start-year {start_year} --end-year {end_year}"

    with store.RunRecorder(conn, command) as run:
        for year in range(start_year, end_year + 1):
            weekly_cutoffs, season_final = _cutoffs(conn, year)
            team_rows = _fetch(run, f"teams/fbs:{year}", partial(client.fetch_fbs_teams, year))
            endpoints += 1
            resolver = _team_resolver(team_rows)

            sp_endpoint = f"ratings/sp:{year}"
            raw_sp = _fetch(run, sp_endpoint, partial(client.fetch_sp, year))
            endpoints += 1
            sp = [
                parse_sp_rating(row, as_of_utc=season_final)
                for row in raw_sp
                if _name_key(row.get("team")) != "nationalaverages"
            ]
            _check_season(sp, year, sp_endpoint)
            sp, skipped_sp = _resolve_rows(
                sp, resolver, endpoint=sp_endpoint, allow_unresolved=True
            )
            filtered_sp += skipped_sp
            for row in sp:
                row["provenance_mode"] = "reconstructed"
            rating_count += store.insert_team_ratings(conn, sp)

            srs_endpoint = f"ratings/srs:{year}"
            raw_srs = _fetch(run, srs_endpoint, partial(client.fetch_srs, year))
            endpoints += 1
            srs = [parse_srs_rating(row, as_of_utc=season_final) for row in raw_srs]
            _check_season(srs, year, srs_endpoint)
            srs, skipped = _resolve_rows(
                srs, resolver, endpoint=srs_endpoint, allow_unresolved=True
            )
            filtered_srs += skipped
            for row in srs:
                row["provenance_mode"] = "reconstructed"
            rating_count += store.insert_team_ratings(conn, srs)

            returning_endpoint = f"player/returning:{year}"
            raw_returning = _fetch(
                run, returning_endpoint, partial(client.fetch_returning_production, year)
            )
            endpoints += 1
            returning = [parse_returning_production(row) for row in raw_returning]
            _check_season(returning, year, returning_endpoint)
            returning, skipped_returning = _resolve_rows(
                returning, resolver, endpoint=returning_endpoint, allow_unresolved=True
            )
            filtered_returning += skipped_returning
            returning_count += store.insert_returning_production(conn, returning)

            talent_endpoint = f"talent:{year}"
            raw_talent = _fetch(run, talent_endpoint, partial(client.fetch_talent, year))
            endpoints += 1
            talent = [parse_talent(row) for row in raw_talent]
            _check_season(talent, year, talent_endpoint)
            talent, skipped_talent = _resolve_rows(
                talent, resolver, endpoint=talent_endpoint, allow_unresolved=True
            )
            filtered_talent += skipped_talent
            talent_count += store.insert_team_talent(conn, talent)

            for week, as_of_utc in sorted(weekly_cutoffs.items()):
                elo_endpoint = f"ratings/elo:{year}:week:{week}"
                raw_elo = _fetch(run, elo_endpoint, partial(client.fetch_elo, year, week))
                endpoints += 1
                elo = [parse_elo_rating(row, week=week, as_of_utc=as_of_utc) for row in raw_elo]
                _check_season(elo, year, elo_endpoint)
                elo, skipped_elo = _resolve_rows(
                    elo, resolver, endpoint=elo_endpoint, allow_unresolved=True
                )
                filtered_elo += skipped_elo
                for row in elo:
                    row["provenance_mode"] = "reconstructed"
                rating_count += store.insert_team_ratings(conn, elo)

                advanced_endpoint = f"stats/season/advanced:{year}:week:{week}"
                raw_advanced = _fetch(
                    run, advanced_endpoint, partial(client.fetch_advanced, year, week)
                )
                endpoints += 1
                advanced = [
                    parsed
                    for raw in raw_advanced
                    for parsed in parse_advanced_rows(raw, week=week, as_of_utc=as_of_utc)
                ]
                _check_season(advanced, year, advanced_endpoint)
                advanced, skipped_advanced = _resolve_rows(
                    advanced, resolver, endpoint=advanced_endpoint, allow_unresolved=True
                )
                filtered_advanced += skipped_advanced
                for row in advanced:
                    row["provenance_mode"] = "reconstructed"
                advanced_count += store.insert_team_advanced(conn, advanced)
            conn.commit()

        total = rating_count + advanced_count + returning_count + talent_count
        run.add_rows(total)
        conn.commit()

    return FundamentalsSummary(
        seasons=end_year - start_year + 1,
        endpoints=endpoints,
        ratings=rating_count,
        advanced=advanced_count,
        returning=returning_count,
        talent=talent_count,
        filtered_sp=filtered_sp,
        filtered_srs=filtered_srs,
        filtered_elo=filtered_elo,
        filtered_advanced=filtered_advanced,
        filtered_returning=filtered_returning,
        filtered_talent=filtered_talent,
    )
