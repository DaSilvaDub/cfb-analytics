"""Historical CFBD backfill for teams, venues, and games."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from functools import partial
from typing import Any

from cfb_analytics.errors import SchemaError, SourceError
from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import (
    CFBDBackfillSummary,
    CFBDClient,
    parse_game,
    parse_game_team,
    parse_team,
    parse_team_aliases,
    parse_team_season,
    parse_venue,
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


def backfill_years(
    conn: sqlite3.Connection,
    client: CFBDClient,
    *,
    start_year: int,
    end_year: int,
) -> CFBDBackfillSummary:
    if start_year > end_year:
        raise SchemaError("start_year must be less than or equal to end_year")
    years = list(range(start_year, end_year + 1))
    venues_written = teams_written = team_seasons_written = aliases_written = games_written = 0

    command = f"backfill-cfbd --start-year {start_year} --end-year {end_year}"
    with store.RunRecorder(conn, command) as run:
        venue_rows = _fetch(run, "venues", client.fetch_venues)
        known_venue_ids: set[str] = set()
        try:
            parsed_venues = [row for raw in venue_rows if (row := parse_venue(raw))]
        except SchemaError as exc:
            run.record_health("cfbd", "venues", ok=False, detail=str(exc))
            raise
        for parsed in parsed_venues:
            store.upsert_venue(conn, parsed)
            known_venue_ids.add(parsed["venue_id"])
            venues_written += 1

        for year in years:
            raw_teams = _fetch(
                run, f"teams/fbs:{year}", partial(client.fetch_fbs_teams, year)
            )
            raw_games = _fetch(
                run, f"games:{year}", partial(client.fetch_games, year)
            )

            try:
                parsed_teams = [parse_team(raw) for raw in raw_teams]
            except SchemaError as exc:
                run.record_health("cfbd", f"teams/fbs:{year}", ok=False, detail=str(exc))
                raise
            try:
                team_season_rows = {
                    row["team_id"]: row for row in (
                        parse_team_season(raw, year=year) for raw in raw_teams
                    )
                }
            except SchemaError as exc:
                run.record_health("cfbd", f"teams/fbs:{year}", ok=False, detail=str(exc))
                raise
            try:
                parsed_games = [parse_game(raw) for raw in raw_games]
            except SchemaError as exc:
                run.record_health("cfbd", f"games:{year}", ok=False, detail=str(exc))
                raise
            known_team_ids = {team["team_id"] for team in parsed_teams}

            # Games involving FBS teams can have non-FBS opponents that are not
            # returned by /teams/fbs. Add those source-carried identities before
            # the game rows so the database foreign keys remain meaningful.
            try:
                for raw in raw_games:
                    for side in ("home", "away"):
                        team = parse_game_team(raw, side=side)
                        if team["team_id"] not in known_team_ids:
                            parsed_teams.append(team)
                            known_team_ids.add(team["team_id"])
                            team_season_rows[team["team_id"]] = {
                                "team_id": team["team_id"],
                                "season": year,
                                "source": "cfbd",
                                "conference": team.get("conference"),
                                "division": None,
                                "classification": team.get("classification"),
                                "venue_id": team.get("venue_id"),
                            }
            except SchemaError as exc:
                run.record_health("cfbd", f"games:{year}", ok=False, detail=str(exc))
                raise

            for team in parsed_teams:
                if team.get("venue_id") not in known_venue_ids:
                    team["venue_id"] = None
                season_row = team_season_rows[team["team_id"]]
                if season_row.get("venue_id") not in known_venue_ids:
                    season_row["venue_id"] = None
                store.upsert_team(conn, team)
                store.upsert_team_season(conn, season_row)
                teams_written += 1
                team_seasons_written += 1

            # Catalog rows carry the richer alternate-name set.
            for raw_team in raw_teams:
                aliases_written += store.insert_team_aliases(
                    conn, parse_team_aliases(raw_team)
                )

            for parsed_game in parsed_games:
                store.upsert_cfbd_game(conn, parsed_game)
                games_written += 1
            conn.commit()

        run.add_rows(
            venues_written + teams_written + team_seasons_written + aliases_written
            + games_written
        )
        conn.commit()

    return CFBDBackfillSummary(
        seasons=len(years),
        venues=venues_written,
        teams=teams_written,
        team_seasons=team_seasons_written,
        aliases=aliases_written,
        games=games_written,
    )
