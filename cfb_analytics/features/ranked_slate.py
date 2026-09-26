"""Select the AP Top 25 games plus each conference's current top two.

The Saturday OVER board is not the full FBS slate. Human polls are the Top 25
source of truth; computer ratings (SP+, SRS) are season-final and must not
stand in. When no poll snapshot is stored, conference order falls back to the
latest weekly CFBD Elo -- never to SP+.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

AP_POLL_KEYS = frozenset({"ap top 25", "ap poll", "ap"})
TOP_25 = 25
CONFERENCE_TOP_N = 2


@dataclass(frozen=True)
class RankedSlateGame:
    """One game admitted to the ranked/conference-leader slate."""

    game_id: str
    football_date: str
    kickoff_utc: str
    season: int
    week: int | None
    home_team_id: str
    away_team_id: str
    home_school: str
    away_school: str
    home_alias: str
    away_alias: str
    inclusion_reasons: tuple[str, ...]


def _is_ap_poll(name: str) -> bool:
    return " ".join(name.strip().casefold().split()) in AP_POLL_KEYS


def latest_ap_poll_week(
    conn: sqlite3.Connection, season: int, *, through_week: int | None = None
) -> int | None:
    query = """SELECT MAX(week) AS week FROM team_polls
               WHERE season = ? AND LOWER(poll) IN ('ap top 25', 'ap poll', 'ap')"""
    params: list[object] = [season]
    if through_week is not None:
        query += " AND week <= ?"
        params.append(through_week)
    row = conn.execute(query, params).fetchone()
    if row is None or row["week"] is None:
        return None
    return int(row["week"])


def load_ap_ranks(
    conn: sqlite3.Connection, season: int, *, week: int | None = None
) -> dict[str, int]:
    poll_week = latest_ap_poll_week(conn, season, through_week=week)
    if poll_week is None:
        return {}
    rows = conn.execute(
        """SELECT team_id, rank, poll FROM team_polls
           WHERE season = ? AND week = ?""",
        (season, poll_week),
    ).fetchall()
    ranks: dict[str, int] = {}
    for row in rows:
        if not _is_ap_poll(str(row["poll"])):
            continue
        rank = int(row["rank"])
        if rank < 1 or rank > TOP_25:
            continue
        ranks[str(row["team_id"])] = rank
    return ranks


def load_latest_elo(
    conn: sqlite3.Connection, season: int, *, as_of_utc: str | None = None
) -> dict[str, float]:
    query = """SELECT MAX(as_of_utc) AS as_of FROM team_ratings
               WHERE source = 'elo_cfbd' AND season = ? AND snapshot_scope = 'weekly'
                 AND rating IS NOT NULL"""
    params: list[object] = [season]
    if as_of_utc is not None:
        query += " AND as_of_utc <= ?"
        params.append(as_of_utc)
    row = conn.execute(query, params).fetchone()
    if row is None or row["as_of"] is None:
        return {}
    ratings = conn.execute(
        """SELECT team_id, rating FROM team_ratings
           WHERE source = 'elo_cfbd' AND season = ? AND snapshot_scope = 'weekly'
             AND as_of_utc = ? AND rating IS NOT NULL""",
        (season, row["as_of"]),
    ).fetchall()
    return {str(item["team_id"]): float(item["rating"]) for item in ratings}


def conference_top_teams(
    conn: sqlite3.Connection,
    season: int,
    ap_ranks: dict[str, int],
    elo: dict[str, float],
    *,
    top_n: int = CONFERENCE_TOP_N,
) -> set[str]:
    rows = conn.execute(
        """SELECT team_id, conference FROM team_seasons
           WHERE season = ? AND conference IS NOT NULL AND TRIM(conference) != ''
             AND LOWER(COALESCE(classification, '')) = 'fbs'""",
        (season,),
    ).fetchall()
    by_conf: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for row in rows:
        team_id = str(row["team_id"])
        if team_id not in ap_ranks and team_id not in elo:
            continue
        poll_rank = ap_ranks.get(team_id, 10_000)
        elo_rating = elo.get(team_id, float("-inf"))
        by_conf[str(row["conference"])].append((poll_rank, -elo_rating, team_id))
    selected: set[str] = set()
    for members in by_conf.values():
        members.sort()
        for _rank, _elo, team_id in members[:top_n]:
            selected.add(team_id)
    return selected


def select_ranked_slate(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    season: int | None = None,
    week: int | None = None,
) -> list[RankedSlateGame]:
    """Games on ``slate_date`` involving an AP Top 25 team or a conference top-two."""
    games = conn.execute(
        """SELECT g.game_id, g.football_date, g.kickoff_utc, g.season, g.week,
                  g.home_team_id, g.away_team_id,
                  COALESCE(ht.school, ht.alias, g.home_team_id) AS home_school,
                  COALESCE(at.school, at.alias, g.away_team_id) AS away_school,
                  COALESCE(ht.alias, ht.school, g.home_team_id) AS home_alias,
                  COALESCE(at.alias, at.school, g.away_team_id) AS away_alias
           FROM games g
           JOIN teams ht ON ht.team_id = g.home_team_id
           JOIN teams at ON at.team_id = g.away_team_id
           WHERE g.football_date = ?
           ORDER BY g.kickoff_utc, g.game_id""",
        (slate_date,),
    ).fetchall()
    if not games:
        return []
    resolved_season = season if season is not None else int(games[0]["season"] or 0)
    slate_week = week
    if slate_week is None:
        game_weeks = [int(row["week"]) for row in games if row["week"] is not None]
        slate_week = max(game_weeks) if game_weeks else None
    as_of_utc = min(str(row["kickoff_utc"]) for row in games)
    ap_ranks = load_ap_ranks(conn, resolved_season, week=slate_week)
    elo = load_latest_elo(conn, resolved_season, as_of_utc=as_of_utc)
    conf_leaders = conference_top_teams(conn, resolved_season, ap_ranks, elo)
    selected: list[RankedSlateGame] = []
    for row in games:
        home_id = str(row["home_team_id"])
        away_id = str(row["away_team_id"])
        reasons: list[str] = []
        home_rank = ap_ranks.get(home_id)
        away_rank = ap_ranks.get(away_id)
        if (home_rank is not None and home_rank <= TOP_25) or (
            away_rank is not None and away_rank <= TOP_25
        ):
            reasons.append("ap_top_25")
        if home_id in conf_leaders or away_id in conf_leaders:
            reasons.append("conference_top_2")
        if not reasons:
            continue
        selected.append(
            RankedSlateGame(
                game_id=str(row["game_id"]),
                football_date=str(row["football_date"]),
                kickoff_utc=str(row["kickoff_utc"]),
                season=int(row["season"] or resolved_season),
                week=int(row["week"]) if row["week"] is not None else None,
                home_team_id=home_id,
                away_team_id=away_id,
                home_school=str(row["home_school"]),
                away_school=str(row["away_school"]),
                home_alias=str(row["home_alias"]),
                away_alias=str(row["away_alias"]),
                inclusion_reasons=tuple(reasons),
            )
        )
    return selected
