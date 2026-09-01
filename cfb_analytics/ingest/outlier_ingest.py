"""Ingest the Outlier NCAAFB slice: schedule, gameline odds, injuries.

A partial failure degrades the run rather than aborting it — one event's markets
failing must not lose the other twenty-nine — but every failure is recorded in
``source_health`` and counted in the summary, so a degraded run is visible
instead of silently thin.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from cfb_analytics.errors import SchemaError, SourceError
from cfb_analytics.ingest import store
from cfb_analytics.sources.outlier import (
    OutlierClient,
    parse_event,
    parse_injury_rows,
    parse_odds_rows,
)
from cfb_analytics.utils import utc_now_iso


@dataclass
class IngestSummary:
    slate_date: str
    events_seen: int = 0
    games_written: int = 0
    teams_written: int = 0
    odds_rows: int = 0
    injury_rows: int = 0
    books: set[str] = field(default_factory=set)
    market_failures: list[str] = field(default_factory=list)
    injury_failures: list[str] = field(default_factory=list)
    schema_failures: list[str] = field(default_factory=list)
    # Cross-check: the feed's own dayOfWeek code must agree with the Eastern
    # weekday we derive. A mismatch means the slate definition has drifted.
    weekday_mismatches: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"slate {self.slate_date}",
            f"  events on slate : {self.events_seen}",
            f"  games written   : {self.games_written}",
            f"  teams written   : {self.teams_written}",
            f"  odds rows       : {self.odds_rows}  across {len(self.books)} books",
            f"  injury rows     : {self.injury_rows}",
        ]
        if self.books:
            lines.append(f"  books           : {', '.join(sorted(self.books))}")
        for label, failures in (
            ("market fetch failures", self.market_failures),
            ("injury fetch failures", self.injury_failures),
            ("schema failures", self.schema_failures),
            ("weekday cross-check mismatches", self.weekday_mismatches),
        ):
            if failures:
                lines.append(f"  {label}: {len(failures)}")
                for item in failures[:5]:
                    lines.append(f"      - {item}")
                if len(failures) > 5:
                    lines.append(f"      ... and {len(failures) - 5} more")
        return "\n".join(lines)


def ingest_slate(
    conn: sqlite3.Connection,
    client: OutlierClient,
    slate_date: str,
    *,
    with_odds: bool = True,
    with_injuries: bool = True,
    limit: int | None = None,
) -> IngestSummary:
    summary = IngestSummary(slate_date=slate_date)
    captured_utc = utc_now_iso()

    with store.RunRecorder(conn, f"ingest --date {slate_date}") as run:
        events = client.fetch_schedule()
        run.record_health("outlier", "schedule", ok=True, rows=len(events))

        parsed = []
        for event in events:
            try:
                record = parse_event(event)
            except SchemaError as exc:
                summary.schema_failures.append(str(exc))
                continue
            # Slate membership is the US Eastern date, not the UTC date. See
            # utils.football_date: UTC grouping misassigns ~24% of a Saturday slate.
            if record["football_date"] == slate_date:
                parsed.append(record)
                if record["weekday_agrees"] is False:
                    summary.weekday_mismatches.append(
                        f"{record['game_id']}: feed dayOfWeek={record['day_of_week']} "
                        f"disagrees with Eastern weekday for {record['kickoff_utc']}"
                    )

        summary.events_seen = len(parsed)
        if limit is not None:
            parsed = parsed[:limit]

        for record in parsed:
            for side in ("home", "away"):
                store.upsert_team(conn, record[side])
                summary.teams_written += 1
            store.upsert_game(
                conn,
                {
                    "game_id": record["game_id"],
                    "season": record["season"],
                    "kickoff_utc": record["kickoff_utc"],
                    "football_date": record["football_date"],
                    "day_of_week": record["day_of_week"],
                    "home_team_id": record["home"]["team_id"],
                    "away_team_id": record["away"]["team_id"],
                    "venue_name": record["venue_name"],
                    "network": record["network"],
                    "status": record["status"],
                },
            )
            summary.games_written += 1
        conn.commit()

        if with_odds:
            _ingest_odds(conn, client, parsed, captured_utc, summary, run)
        if with_injuries:
            _ingest_injuries(conn, client, parsed, captured_utc, summary, run)

        run.add_rows(summary.odds_rows + summary.injury_rows + summary.games_written)
        conn.commit()

    return summary


def _ingest_odds(conn, client, parsed, captured_utc, summary, run) -> None:
    for record in parsed:
        game_id = record["game_id"]
        try:
            markets = client.fetch_event_markets(game_id, "GAMELINE")
        except SourceError as exc:
            summary.market_failures.append(f"{game_id}: {exc}")
            run.record_health("outlier", f"markets:{game_id}", ok=False, detail=str(exc))
            continue
        rows = parse_odds_rows(game_id, markets, captured_utc)
        written = store.insert_odds(conn, rows)
        summary.odds_rows += written
        summary.books.update(row.book for row in rows)
        run.record_health("outlier", f"markets:{game_id}", ok=True, rows=written)
    conn.commit()


def _ingest_injuries(conn, client, parsed, captured_utc, summary, run) -> None:
    for record in parsed:
        game_id = record["game_id"]
        for side in ("home", "away"):
            team_id = record[side]["team_id"]
            try:
                players = client.fetch_team_injuries(team_id)
            except SourceError as exc:
                summary.injury_failures.append(f"{team_id}: {exc}")
                run.record_health("outlier", f"injuries:{team_id}", ok=False, detail=str(exc))
                continue
            rows = parse_injury_rows(game_id, team_id, players, captured_utc)
            summary.injury_rows += store.insert_availability(conn, rows)
            run.record_health("outlier", f"injuries:{team_id}", ok=True, rows=len(rows))
    conn.commit()
