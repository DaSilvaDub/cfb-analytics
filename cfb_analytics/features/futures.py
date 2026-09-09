"""Integration layer connecting Model 4 (Season Futures) with SQLite store.

Loads preseason talent, returning production, conference affiliations, strength
of schedule, and scheduled games from the database, then feeds them to the pure
mathematical futures engine in ``cfb_analytics.models.futures``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.asof import AsOfReader
from cfb_analytics.models.futures import (
    NILTier,
    PortalComposite,
    QBContinuity,
    QBTier,
    ReturningProduction,
    RosterTalentInputs,
    ScheduledOpponent,
    SeasonFuturesProjection,
    calculate_player_portal_score,
    project_season_futures,
)


def _preferred_rating_row(
    rows: list[dict[str, Any]], reader: AsOfReader | None
) -> dict[str, Any] | None:
    """Latest admissible row from the first available, compatible rating source."""
    for source in ("sp", "srs", "elo_cfbd"):
        source_rows = [row for row in rows if row.get("source") == source]
        if not source_rows:
            continue
        if reader is not None:
            latest = reader.latest(
                source_rows,
                what=f"team_ratings.{source}",
                as_of_field="as_of_utc",
            )
            return dict(latest) if latest is not None else None
        return max(
            source_rows,
            key=lambda row: (str(row["as_of_utc"]), str(row.get("snapshot_id", ""))),
        )
    return None


def load_team_roster_inputs(
    conn: sqlite3.Connection,
    team_id: str,
    season: int,
    *,
    as_of_utc: str | None = None,
    portal_composite: PortalComposite | float | None = None,
    nil_tier: NILTier | str | None = None,
    nil_budget_millions: float | None = None,
    qb_tier: QBTier | str | None = None,
    qb_continuity: QBContinuity | str | None = None,
    base_power_rating: float | None = None,
    true_talent_composite: float | None = None,
    strength_of_schedule: float | None = None,
    input_manifest: dict[str, Any] | None = None,
    recruiting_composite: float | None = None,
) -> RosterTalentInputs:
    """Build RosterTalentInputs from database records and optional caller overrides.

    Pulls recruiting talent composite from ``team_recruiting`` or ``team_talent``,
    portal composite from ``team_portal_composites`` or ``transfer_portal_players``,
    returning production from ``returning_production``, historical conference from
    ``team_seasons``, and strength of schedule from ``team_ratings`` or a caller override.
    """
    reader = (
        AsOfReader(game_id="futures-roster-lookup", kickoff_utc=as_of_utc, season=season)
        if as_of_utc is not None
        else None
    )
    if reader is not None:
        _ = reader.kickoff

    recruiting_comp: float | None = None
    if recruiting_composite is not None:
        recruiting_comp = float(recruiting_composite)
        if input_manifest is not None:
            input_manifest["recruiting_composite"] = {
                "value": recruiting_comp,
                "resolution": "caller_supplied",
            }
    else:
        talent_row = conn.execute(
            """SELECT snapshot_id, season, availability_class, ingested_utc, talent_composite
               FROM team_talent
               WHERE team_id = ? AND season = ? AND availability_class = 'preseason'
               ORDER BY ingested_utc DESC, snapshot_id DESC LIMIT 1""",
            (team_id, season),
        ).fetchone()
        if talent_row is not None:
            if reader is not None:
                reader.check_availability_class(
                    str(talent_row["availability_class"]), feature="team_talent"
                )
                reader.check_preseason_season(talent_row["season"], feature="team_talent")
            if input_manifest is not None:
                input_manifest["team_talent"] = dict(talent_row)
            recruiting_comp = float(talent_row["talent_composite"])

        rec_rows = conn.execute(
            """SELECT snapshot_id, season, availability_class, as_of_utc, ingested_utc,
                      rank, points, recruiting_composite
               FROM team_recruiting
               WHERE team_id = ? AND season = ? AND availability_class = 'preseason'""",
            (team_id, season),
        ).fetchall()
        if rec_rows:
            admissible_rec = (
                reader.admissible(
                    [dict(r) for r in rec_rows],
                    what="team_recruiting",
                    as_of_field="as_of_utc",
                )
                if reader is not None
                else [dict(r) for r in rec_rows]
            )
            if admissible_rec:
                rec_row = max(
                    admissible_rec,
                    key=lambda r: (
                        str(r.get("as_of_utc", "")),
                        str(r.get("ingested_utc", "")),
                        str(r.get("snapshot_id", "")),
                    ),
                )
                if reader is not None:
                    reader.check_availability_class(
                        str(rec_row["availability_class"]), feature="team_recruiting"
                    )
                    reader.check_preseason_season(
                        rec_row["season"], feature="team_recruiting"
                    )
                if input_manifest is not None:
                    input_manifest["team_recruiting"] = dict(rec_row)
                if recruiting_comp is None:
                    val = rec_row.get("recruiting_composite")
                    if val is not None:
                        recruiting_comp = float(val)

        if recruiting_comp is None:
            raise SchemaError(
                f"No preseason recruiting or talent row for team {team_id!r} in season {season}"
            )

    ret_row = conn.execute(
        """SELECT snapshot_id, season, availability_class, ingested_utc, percent_ppa
           FROM returning_production
           WHERE team_id = ? AND season = ? AND availability_class = 'preseason'
             AND percent_ppa IS NOT NULL
           ORDER BY ingested_utc DESC, snapshot_id DESC LIMIT 1""",
        (team_id, season),
    ).fetchone()
    if ret_row is None:
        raise SchemaError(
            f"No preseason returning-production row for team {team_id!r} in season {season}"
        )
    if reader is not None:
        reader.check_availability_class(
            str(ret_row["availability_class"]), feature="returning_production"
        )
        reader.check_preseason_season(ret_row["season"], feature="returning_production")
    # CFBD's percentPPA is an offensive total. It does not expose a defensive
    # returning-production percentage, so defense remains explicitly unknown.
    ret_prod = ReturningProduction(percent_ppa_offense=float(ret_row["percent_ppa"]))

    team_row = conn.execute(
        """SELECT conference FROM team_seasons
           WHERE team_id = ? AND season = ? AND source = 'cfbd'""",
        (team_id, season),
    ).fetchone()
    if team_row is None or not team_row["conference"]:
        raise SchemaError(f"No conference affiliation for team {team_id!r} in season {season}")
    conference = str(team_row["conference"])

    if input_manifest is not None:
        input_manifest["returning_production"] = dict(ret_row)
        input_manifest["team_season"] = {
            "team_id": team_id,
            "season": season,
            "source": "cfbd",
            "conference": conference,
        }

    if strength_of_schedule is not None:
        sos = strength_of_schedule
    else:
        sos_rows = conn.execute(
            """SELECT snapshot_id, period, sos, source, snapshot_scope,
                      provenance_mode, as_of_utc, ingested_utc
               FROM team_ratings
               WHERE team_id = ? AND season = ? AND sos IS NOT NULL""",
            (team_id, season),
        ).fetchall()
        candidates = [dict(row) for row in sos_rows]
        if reader is not None:
            candidates = [
                dict(row)
                for row in reader.admissible(
                    candidates, what="team_ratings.sos", as_of_field="as_of_utc"
                )
            ]
        sos_row = _preferred_rating_row(candidates, reader=None)
        if sos_row is None:
            raise SchemaError(
                f"No admissible strength-of-schedule row for team {team_id!r} in season {season}"
            )
        sos = float(sos_row["sos"])
        if input_manifest is not None:
            input_manifest["strength_of_schedule"] = {
                **sos_row,
                "value": sos,
                "resolution": "database_rating",
            }

    if input_manifest is not None and "strength_of_schedule" not in input_manifest:
        input_manifest["strength_of_schedule"] = {
            "value": sos,
            "resolution": "caller_supplied",
        }

    if nil_tier is None:
        if nil_budget_millions is None:
            raise SchemaError(
                "NIL tier or budget is required; conference affiliation is not an NIL estimate"
            )
        resolved_nil_tier: NILTier | str = NILTier.from_budget(nil_budget_millions)
    else:
        resolved_nil_tier = nil_tier

    resolved_portal: PortalComposite | float | None = portal_composite
    if resolved_portal is None:
        portal_rows = conn.execute(
            """SELECT snapshot_id, season, availability_class, as_of_utc, ingested_utc,
                      additions_score, departures_score, net_composite,
                      additions_count, departures_count
               FROM team_portal_composites
               WHERE team_id = ? AND season = ? AND availability_class = 'preseason'""",
            (team_id, season),
        ).fetchall()
        if portal_rows:
            admissible_portal = (
                reader.admissible(
                    [dict(r) for r in portal_rows],
                    what="team_portal_composites",
                    as_of_field="as_of_utc",
                )
                if reader is not None
                else [dict(r) for r in portal_rows]
            )
            if admissible_portal:
                portal_row = max(
                    admissible_portal,
                    key=lambda r: (
                        str(r.get("as_of_utc", "")),
                        str(r.get("ingested_utc", "")),
                        str(r.get("snapshot_id", "")),
                    ),
                )
                if reader is not None:
                    reader.check_availability_class(
                        str(portal_row["availability_class"]),
                        feature="team_portal_composites",
                    )
                    reader.check_preseason_season(
                        portal_row["season"], feature="team_portal_composites"
                    )
                resolved_portal = PortalComposite(
                    additions_score=float(portal_row["additions_score"]),
                    departures_score=float(portal_row["departures_score"]),
                    net_composite=float(portal_row["net_composite"]),
                    additions_count=int(portal_row["additions_count"]),
                    departures_count=int(portal_row["departures_count"]),
                )
                if input_manifest is not None:
                    input_manifest["team_portal"] = dict(portal_row)

        if resolved_portal is None:
            player_rows = conn.execute(
                """SELECT transfer_id, season, first_name, last_name, origin_team_id,
                          destination_team_id, origin_name, destination_name,
                          rating, stars, transfer_date, as_of_utc, ingested_utc
                   FROM transfer_portal_players
                   WHERE season = ? AND (origin_team_id = ? OR destination_team_id = ?)""",
                (season, team_id, team_id),
            ).fetchall()
            if player_rows:
                admissible_players = (
                    reader.admissible(
                        [dict(r) for r in player_rows],
                        what="transfer_portal_players",
                        as_of_field="as_of_utc",
                    )
                    if reader is not None
                    else [dict(r) for r in player_rows]
                )
                if admissible_players:
                    deduped: dict[tuple[Any, ...], Mapping[str, Any]] = {}
                    for p in admissible_players:
                        movement_key = (
                            p.get("season"),
                            p.get("first_name"),
                            p.get("last_name"),
                            p.get("origin_name") or p.get("origin_team_id"),
                            p.get("destination_name") or p.get("destination_team_id"),
                            p.get("transfer_date"),
                        )
                        existing = deduped.get(movement_key)
                        if existing is None or (
                            str(p.get("as_of_utc", "")),
                            str(p.get("ingested_utc", "")),
                        ) > (
                            str(existing.get("as_of_utc", "")),
                            str(existing.get("ingested_utc", "")),
                        ):
                            deduped[movement_key] = p

                    unique_admissible = list(deduped.values())
                    additions = [
                        p for p in unique_admissible if p.get("destination_team_id") == team_id
                    ]
                    departures = [
                        p for p in unique_admissible if p.get("origin_team_id") == team_id
                    ]
                    add_score = round(
                        sum(calculate_player_portal_score(p) for p in additions), 2
                    )
                    dep_score = round(
                        sum(calculate_player_portal_score(p) for p in departures), 2
                    )
                    net = round(add_score - dep_score, 2)
                    resolved_portal = PortalComposite(
                        additions_score=add_score,
                        departures_score=dep_score,
                        net_composite=net,
                        additions_count=len(additions),
                        departures_count=len(departures),
                    )
                    if input_manifest is not None:
                        input_manifest["team_portal"] = {
                            "team_id": team_id,
                            "season": season,
                            "additions_score": add_score,
                            "departures_score": dep_score,
                            "net_composite": net,
                            "additions_count": len(additions),
                            "departures_count": len(departures),
                            "resolution": "aggregated_from_players",
                        }

    if resolved_portal is None:
        raise SchemaError(
            "Portal composite is required; missing transfer data cannot be treated as neutral"
        )
    if qb_tier is None or qb_continuity is None:
        raise SchemaError(
            "QB tier and continuity are required; missing quarterback data cannot be inferred"
        )

    return RosterTalentInputs(
        team_id=team_id,
        recruiting_composite=recruiting_comp,
        portal_composite=resolved_portal,
        nil_tier=resolved_nil_tier,
        nil_budget_millions=nil_budget_millions,
        returning_production=ret_prod,
        qb_tier=qb_tier,
        qb_continuity=qb_continuity,
        strength_of_schedule=sos,
        base_power_rating=base_power_rating,
        conference=conference,
        true_talent_composite=true_talent_composite,
    )


def load_team_schedule(
    conn: sqlite3.Connection,
    team_id: str,
    season: int,
    *,
    as_of_utc: str | None = None,
    reader_game_id: str = "futures-schedule-lookup",
    opponent_ratings: Mapping[str, float] | None = None,
    default_opp_rating: float | None = None,
    input_manifest: dict[str, Any] | None = None,
) -> list[ScheduledOpponent]:
    """Load regular season schedule for team_id from the ``games`` table.

    Each game resolves opponent_power_rating via ``opponent_ratings`` mapping,
    falling back to database ratings or ``default_opp_rating`` when absent.
    """
    rows = conn.execute(
        """SELECT game_id, week, kickoff_utc, home_team_id, away_team_id,
                  neutral_site, conference_game, season_type, ingested_utc,
                  completed, home_points, away_points
           FROM games
           WHERE source = 'cfbd' AND season = ?
             AND (home_team_id = ? OR away_team_id = ?)
           ORDER BY kickoff_utc ASC, game_id ASC""",
        (season, team_id, team_id),
    ).fetchall()

    if as_of_utc is not None:
        reader = AsOfReader(game_id=reader_game_id, kickoff_utc=as_of_utc, season=season)
        games_list = reader.admissible(
            [dict(r) for r in rows], what="games", as_of_field="ingested_utc"
        )
    else:
        games_list = [dict(r) for r in rows]

    ratings = dict(opponent_ratings or {})
    schedule: list[ScheduledOpponent] = []
    schedule_manifest: list[dict[str, Any]] = []

    for g in games_list:
        # Ignore postseason, bowl, and conference championship games for regular season win totals
        st = g.get("season_type")
        if st and str(st).lower() in ("postseason", "bowl", "championship", "ccg"):
            continue

        home = str(g["home_team_id"])
        away = str(g["away_team_id"])
        is_home = home == team_id
        opp_id = away if is_home else home

        if opp_id in ratings:
            opp_rating = ratings[opp_id]
            rating_manifest: dict[str, Any] = {
                "resolution": "caller_supplied_unverified",
                "value": opp_rating,
            }
        else:
            rating_rows = conn.execute(
                """SELECT snapshot_id, period, rating, source, snapshot_scope,
                          provenance_mode, as_of_utc, ingested_utc
                   FROM team_ratings
                   WHERE team_id = ? AND season = ? AND rating IS NOT NULL""",
                (opp_id, season),
            ).fetchall()
            rating_candidates = [dict(row) for row in rating_rows]
            if as_of_utc is not None:
                rating_candidates = [
                    dict(row)
                    for row in reader.admissible(
                        rating_candidates,
                        what=f"team_ratings.{opp_id}",
                        as_of_field="as_of_utc",
                    )
                ]
            opp_row = _preferred_rating_row(rating_candidates, reader=None)

            if opp_row and opp_row["rating"] is not None:
                raw_rating = float(opp_row["rating"])
                # If the rating is on the Elo scale (> 100.0, or
                # source == 'elo_cfbd'), convert it to spread points.
                if opp_row["source"] == "elo_cfbd" or raw_rating > 100.0:
                    opp_rating = round((raw_rating - 1500.0) / 25.0, 2)
                else:
                    opp_rating = raw_rating
                rating_manifest = {
                    **opp_row,
                    "raw_value": raw_rating,
                    "value": opp_rating,
                    "resolution": "database_rating",
                }
            else:
                team_meta = conn.execute(
                    """SELECT classification FROM team_seasons
                       WHERE team_id = ? AND season = ? AND source = 'cfbd'""",
                    (opp_id, season),
                ).fetchone()
                if (
                    team_meta
                    and team_meta["classification"]
                    and str(team_meta["classification"]).lower() == "fcs"
                ):
                    opp_rating = -25.0
                    rating_manifest = {
                        "resolution": "fcs_classification_fallback",
                        "source": "cfbd",
                        "season": season,
                        "classification": str(team_meta["classification"]),
                        "value": opp_rating,
                    }
                elif default_opp_rating is not None:
                    opp_rating = default_opp_rating
                    rating_manifest = {
                        "resolution": "caller_supplied_default_unverified",
                        "value": opp_rating,
                    }
                else:
                    raise SchemaError(
                        f"No admissible opponent rating for {opp_id!r} in season {season}"
                    )

        known_result = None
        if bool(g.get("completed", 0)):
            home_points = g.get("home_points")
            away_points = g.get("away_points")
            if home_points is None or away_points is None:
                raise SchemaError(f"Completed game {g['game_id']!r} is missing a final score")
            team_points = int(home_points) if is_home else int(away_points)
            opponent_points = int(away_points) if is_home else int(home_points)
            if team_points == opponent_points:
                raise SchemaError(
                    f"Completed college-football game {g['game_id']!r} has a tied score"
                )
            known_result = 1.0 if team_points > opponent_points else 0.0

        schedule.append(
            ScheduledOpponent(
                opponent_id=opp_id,
                opponent_power_rating=opp_rating,
                is_home=is_home,
                is_neutral=bool(g.get("neutral_site", 0)),
                game_week=int(g["week"]) if g.get("week") is not None else None,
                is_conference=bool(g.get("conference_game", 0)),
                known_result=known_result,
            )
        )
        schedule_manifest.append(
            {
                "game_id": str(g["game_id"]),
                "week": g.get("week"),
                "kickoff_utc": g.get("kickoff_utc"),
                "ingested_utc": g.get("ingested_utc"),
                "opponent_id": opp_id,
                "is_home": is_home,
                "completed": bool(g.get("completed", 0)),
                "home_points": g.get("home_points"),
                "away_points": g.get("away_points"),
                "known_result": known_result,
                "opponent_rating": rating_manifest,
            }
        )

    if input_manifest is not None:
        input_manifest["schedule"] = schedule_manifest

    return schedule


def project_team_futures_from_db(
    conn: sqlite3.Connection,
    team_id: str,
    season: int,
    *,
    as_of_utc: str | None = None,
    portal_composite: PortalComposite | float | None = None,
    nil_tier: NILTier | str | None = None,
    nil_budget_millions: float | None = None,
    qb_tier: QBTier | str | None = None,
    qb_continuity: QBContinuity | str | None = None,
    opponent_ratings: Mapping[str, float] | None = None,
    posted_lines: list[float] | None = None,
    true_talent_composite: float | None = None,
    strength_of_schedule: float | None = None,
    input_manifest: dict[str, Any] | None = None,
    recruiting_composite: float | None = None,
) -> SeasonFuturesProjection:
    """End-to-end integration: query database and generate SeasonFuturesProjection."""
    if as_of_utc is None:
        raise SchemaError("DB-backed futures projections require an explicit as_of_utc cutoff")

    schedule = load_team_schedule(
        conn,
        team_id,
        season,
        as_of_utc=as_of_utc,
        opponent_ratings=opponent_ratings,
        input_manifest=input_manifest,
    )
    if not schedule:
        raise SchemaError(
            f"No regular season scheduled games found for team {team_id!r} in season {season}"
        )

    # When no separately projected SOS is supplied, use the mean of the same
    # point-in-time opponent ratings that drive the schedule projection. This
    # avoids pulling same-season final SP+/SRS SOS into an in-season forecast.
    resolved_sos = (
        strength_of_schedule
        if strength_of_schedule is not None
        else sum(game.opponent_power_rating for game in schedule) / len(schedule)
    )
    if input_manifest is not None:
        input_manifest["strength_of_schedule"] = {
            "value": resolved_sos,
            "resolution": (
                "caller_supplied"
                if strength_of_schedule is not None
                else "derived_schedule_opponent_rating_mean"
            ),
        }
    inputs = load_team_roster_inputs(
        conn,
        team_id,
        season,
        as_of_utc=as_of_utc,
        portal_composite=portal_composite,
        nil_tier=nil_tier,
        nil_budget_millions=nil_budget_millions,
        qb_tier=qb_tier,
        qb_continuity=qb_continuity,
        true_talent_composite=true_talent_composite,
        strength_of_schedule=resolved_sos,
        input_manifest=input_manifest,
        recruiting_composite=recruiting_composite,
    )

    return project_season_futures(
        inputs,
        schedule,
        posted_lines=posted_lines,
    )
