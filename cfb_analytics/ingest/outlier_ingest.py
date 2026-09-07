"""Ingest the Outlier NCAAFB slice: schedule, gameline odds, injuries.

A partial failure degrades the run rather than aborting it — one event's markets
failing must not lose the other twenty-nine — but every failure is recorded in
``source_health`` and counted in the summary, so a degraded run is visible
instead of silently thin.

**This ingest writes no games and no teams.** It resolves each event onto the
canonical CFBD row and attaches odds and availability there. Writing its own
``games`` row under Outlier's ``eventId`` is what put every current-slate game
in the store twice, splitting one game's book set across two keys. An event
that does not resolve is reported and skipped, never materialised — see
``ingest.identity``.

Note the asymmetry that runs through this module: the Outlier **API** is called
with Outlier's own ids (``eventId``, ``teamId``), while everything **stored**
carries canonical ids. Confusing the two is the easy mistake here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from cfb_analytics.errors import SchemaError, SourceError
from cfb_analytics.features.team_props import build_team_prop_consensus
from cfb_analytics.ingest import store
from cfb_analytics.ingest.identity import CanonicalResolver, EventResolution
from cfb_analytics.sources.outlier import (
    OutlierClient,
    parse_event,
    parse_injury_rows,
    parse_odds_rows,
    parse_team_prop_rows,
)
from cfb_analytics.utils import utc_now_iso


@dataclass
class IngestSummary:
    slate_date: str
    events_seen: int = 0
    games_matched: int = 0
    aliases_written: int = 0
    odds_rows: int = 0
    injury_rows: int = 0
    team_prop_rows: int = 0
    team_prop_consensus_rows: int = 0
    books: set[str] = field(default_factory=set)
    market_failures: list[str] = field(default_factory=list)
    injury_failures: list[str] = field(default_factory=list)
    schema_failures: list[str] = field(default_factory=list)
    # Events with no unique canonical game. Named, never silently dropped.
    unresolved_events: list[str] = field(default_factory=list)
    # Cross-check: the feed's own dayOfWeek code must agree with the Eastern
    # weekday we derive. A mismatch means the slate definition has drifted.
    weekday_mismatches: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"slate {self.slate_date}",
            f"  events on slate : {self.events_seen}",
            f"  games matched   : {self.games_matched}",
            f"  team aliases    : {self.aliases_written}",
            f"  odds rows       : {self.odds_rows}  across {len(self.books)} books",
            f"  injury rows     : {self.injury_rows}",
            f"  team prop rows  : {self.team_prop_rows}",
            f"  prop consensus  : {self.team_prop_consensus_rows}",
        ]
        if self.books:
            lines.append(f"  books           : {', '.join(sorted(self.books))}")
        for label, failures in (
            ("market fetch failures", self.market_failures),
            ("injury fetch failures", self.injury_failures),
            ("schema failures", self.schema_failures),
            ("unresolved events", self.unresolved_events),
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
    with_props: bool = False,
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

        resolver = CanonicalResolver(conn)
        resolved: list[tuple[dict, EventResolution]] = []
        for record in parsed:
            resolution = resolver.event(record)
            if not resolution.ok:
                summary.unresolved_events.append(resolution.reason)
                run.record_health(
                    "outlier",
                    f"resolve:{record['game_id']}",
                    ok=False,
                    detail=resolution.reason,
                )
                continue
            summary.games_matched += 1
            summary.aliases_written += store.insert_team_aliases(
                conn, _alias_rows(record, resolution)
            )
            resolved.append((record, resolution))
        conn.commit()

        if with_odds:
            _ingest_odds(conn, client, resolved, captured_utc, summary, run)
        if with_injuries:
            _ingest_injuries(conn, client, resolved, captured_utc, summary, run)
        if with_props:
            _ingest_team_props(conn, client, resolved, captured_utc, summary, run)

        run.add_rows(summary.odds_rows + summary.injury_rows + summary.team_prop_rows)
        conn.commit()

    return summary


def _ingest_odds(conn, client, resolved, captured_utc, summary, run) -> None:
    for record, resolution in resolved:
        event_id = record["game_id"]  # Outlier's id: what the API wants.
        canonical_id = resolution.game_id  # cfbd:<id>: what the store wants.
        try:
            markets = client.fetch_event_markets(event_id, "GAMELINE")
        except SourceError as exc:
            summary.market_failures.append(f"{event_id}: {exc}")
            run.record_health("outlier", f"markets:{event_id}", ok=False, detail=str(exc))
            continue
        rows = parse_odds_rows(canonical_id, markets, captured_utc)
        written = store.insert_odds(conn, rows)
        summary.odds_rows += written
        summary.books.update(row.book for row in rows)
        run.record_health("outlier", f"markets:{event_id}", ok=True, rows=written)
    conn.commit()


def _ingest_injuries(conn, client, resolved, captured_utc, summary, run) -> None:
    for record, resolution in resolved:
        canonical_game = resolution.game_id
        sides = (
            ("home", resolution.home_team_id),
            ("away", resolution.away_team_id),
        )
        for side, canonical_team in sides:
            feed_team_id = record[side]["team_id"]  # Outlier's id: for the API.
            try:
                players = client.fetch_team_injuries(feed_team_id)
            except SourceError as exc:
                summary.injury_failures.append(f"{feed_team_id}: {exc}")
                run.record_health("outlier", f"injuries:{feed_team_id}", ok=False, detail=str(exc))
                continue
            rows = parse_injury_rows(canonical_game, canonical_team, players, captured_utc)
            summary.injury_rows += store.insert_availability(conn, rows)
            run.record_health("outlier", f"injuries:{feed_team_id}", ok=True, rows=len(rows))
    conn.commit()


def _ingest_team_props(conn, client, resolved, captured_utc, summary, run) -> None:
    canonical_games: list[str] = []
    for record, resolution in resolved:
        event_id = record["game_id"]
        canonical_game = resolution.game_id
        team_id_map = {
            record["home"]["team_id"]: resolution.home_team_id,
            record["away"]["team_id"]: resolution.away_team_id,
        }
        team_id_map = {key: value for key, value in team_id_map.items() if value}
        try:
            markets = client.fetch_event_markets(event_id, "TEAM_PROP")
        except SourceError as exc:
            summary.market_failures.append(f"{event_id} TEAM_PROP: {exc}")
            run.record_health("outlier", f"team_props:{event_id}", ok=False, detail=str(exc))
            continue
        rows = parse_team_prop_rows(canonical_game, markets, captured_utc, team_id_map)
        written = store.insert_team_prop_odds(conn, rows)
        summary.team_prop_rows += written
        summary.books.update(row.book for row in rows)
        canonical_games.append(canonical_game)
        run.record_health("outlier", f"team_props:{event_id}", ok=True, rows=written)
    summary.team_prop_consensus_rows = build_team_prop_consensus(
        conn, canonical_games, as_of_utc=captured_utc
    )
    conn.commit()


def _alias_rows(record: dict, resolution: EventResolution) -> list[dict[str, str]]:
    """Keep Outlier's naming as aliases on the canonical team.

    This ingest no longer writes ``teams`` rows, so without this the feed's
    vocabulary would be discarded and every run would re-derive the same
    mapping. Only ``market`` ("Washington") and ``alias`` ("WASH") are kept --
    deliberately NOT the nickname, because "Huskies" denotes three programmes
    and storing it as a resolving name would eventually let one of them win.
    """
    rows: list[dict[str, str]] = []
    sides = (
        ("home", resolution.home_team_id),
        ("away", resolution.away_team_id),
    )
    for side, canonical_team in sides:
        if canonical_team is None:
            continue
        team = record[side]
        for field_name, alias_type in (("market", "market"), ("alias", "abbreviation")):
            value = team.get(field_name)
            if value:
                rows.append(
                    {
                        "team_id": canonical_team,
                        "source": "outlier",
                        "alias": str(value),
                        "alias_type": alias_type,
                    }
                )
    return rows
