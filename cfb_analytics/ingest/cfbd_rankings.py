"""Ingest CFBD weekly human polls into ``team_polls``."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_fundamentals import _name_key
from cfb_analytics.sources.cfbd import CFBDClient, parse_rankings
from cfb_analytics.utils import utc_now_iso


@dataclass(frozen=True)
class RankingsSummary:
    weeks: int
    rows: int
    unresolved: int

    def as_text(self) -> str:
        return (
            f"CFBD rankings wrote {self.rows} poll ranks across {self.weeks} week(s); "
            f"{self.unresolved} names did not resolve."
        )


def _team_name_index(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT team_id, school, alias, market FROM teams").fetchall()
    resolved: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in rows:
        team_id = str(row["team_id"])
        for name in (row["school"], row["alias"], row["market"]):
            key = _name_key(name)
            if not key:
                continue
            prior = resolved.get(key)
            if prior is not None and prior != team_id:
                ambiguous.add(key)
            else:
                resolved[key] = team_id
    for key in ambiguous:
        resolved.pop(key, None)
    aliases = conn.execute("SELECT team_id, alias FROM team_aliases").fetchall()
    for row in aliases:
        key = _name_key(row["alias"])
        if not key or key in ambiguous:
            continue
        prior = resolved.get(key)
        team_id = str(row["team_id"])
        if prior is not None and prior != team_id:
            resolved.pop(key, None)
            ambiguous.add(key)
        else:
            resolved[key] = team_id
    return resolved


def ingest_rankings(
    conn: sqlite3.Connection,
    client: CFBDClient,
    year: int,
    *,
    week: int | None = None,
    season_type: str = "regular",
) -> RankingsSummary:
    payload = client.fetch_rankings(year, week=week, season_type=season_type)
    as_of = utc_now_iso()
    resolver = _team_name_index(conn)
    written = 0
    unresolved = 0
    weeks: set[int] = set()
    with store.RunRecorder(conn, "backfill-rankings") as run:
        run.record_health("cfbd", "/rankings", ok=True, rows=len(payload))
        batch: list[dict[str, Any]] = []
        for raw in payload:
            for parsed in parse_rankings(raw, as_of_utc=as_of):
                team_id = resolver.get(_name_key(parsed.pop("team_name")))
                if team_id is None:
                    unresolved += 1
                    continue
                weeks.add(int(parsed["week"]))
                batch.append({**parsed, "team_id": team_id})
        written = store.insert_team_polls(conn, batch)
        run.add_rows(written)
    if week is not None and payload and not weeks:
        raise SchemaError(f"CFBD rankings for {year} week {week} resolved zero teams")
    conn.commit()
    return RankingsSummary(weeks=len(weeks), rows=written, unresolved=unresolved)
