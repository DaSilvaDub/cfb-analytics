"""Cross-source identity: which canonical row is this feed talking about?

**CFBD is canonical.** Not because it is better data, but because it is the
only source with a complete history: 11,355 games over twelve seasons, and
every rating, Elo, feature and backtest row already keys on ``cfbd:<id>``.
Outlier covers the current slate window only (93 games against CFBD's 888 for
one season) and is a best-effort odds overlay whose token expires every 24
hours. Re-pointing the twelve-season backbone at a feed that goes dark on day
two would be the wrong way round.

The two ingests used to mint primary keys in **disjoint namespaces** --
``cfbd:401`` against Outlier's ``eventId`` -- with nothing mapping between
them, so one real game was two rows. The board listed every side twice and
consensus was computed over a split book set: on the 2026-09-06 Washington
game, 1478 consensus rows sat on the Outlier row and 4 on the CFBD one, each
blind to the other's books. That is the same error as reading a market from
one of several rows instead of their union.

Resolution here **never guesses**. A name that identifies two teams resolves to
nothing, exactly as an ambiguous venue name resolves to a NULL ``venue_id``
rather than to whichever row sorts first. The caller reports the refusal; it
does not invent a row to hold the data.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

CANONICAL_SOURCE = "cfbd"


def name_key(value: Any) -> str:
    """Normalise a team name for comparison across feeds."""
    return str(value or "").strip().upper()


@dataclass(frozen=True)
class GameMatch:
    game_id: str
    # Whether the feed and the canonical row agree on which team is home.
    # Odds sides are stored as HOME/AWAY, so a disagreement is not cosmetic:
    # re-keying across it would relabel every price onto the wrong team.
    orientation_agrees: bool


@dataclass(frozen=True)
class EventResolution:
    """The outcome of resolving one feed event, including why it failed."""

    game_id: str | None = None
    home_team_id: str | None = None
    away_team_id: str | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.game_id is not None


class CanonicalResolver:
    """Maps a foreign feed's teams and events onto canonical CFBD rows.

    Both indexes are built once per instance, so an ingest resolving thirty
    events does two queries rather than sixty.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._teams = self._build_team_index(conn)
        self._games = self._build_game_index(conn)

    @property
    def canonical_teams(self) -> int:
        return len({team for ids in self._teams.values() for team in ids})

    @staticmethod
    def _build_team_index(conn: sqlite3.Connection) -> dict[str, set[str]]:
        """Every name that could denote a canonical team, to the teams it denotes.

        A name is kept even when it is ambiguous; ambiguity is decided at lookup
        time so that "this matched three teams" stays distinguishable from
        "this matched nothing".
        """
        index: dict[str, set[str]] = defaultdict(set)
        for row in conn.execute(
            """SELECT team_id, school, alias, market FROM teams
               WHERE cfbd_id IS NOT NULL"""
        ):
            for field in ("school", "alias", "market"):
                key = name_key(row[field])
                if key:
                    index[key].add(str(row["team_id"]))
        for row in conn.execute(
            """SELECT a.team_id, a.alias FROM team_aliases a
               JOIN teams t ON t.team_id = a.team_id
               WHERE t.cfbd_id IS NOT NULL"""
        ):
            key = name_key(row["alias"])
            if key:
                index[key].add(str(row["team_id"]))
        return dict(index)

    @staticmethod
    def _build_game_index(
        conn: sqlite3.Connection,
    ) -> dict[tuple[frozenset[str], str], list[tuple[str, str]]]:
        """Canonical games by (unordered team pair, slate date).

        The pair is unordered so that a feed disagreeing about which side is
        home still finds the game -- the disagreement is then *reported* rather
        than silently absorbed.
        """
        index: dict[tuple[frozenset[str], str], list[tuple[str, str]]] = defaultdict(list)
        for row in conn.execute(
            """SELECT game_id, home_team_id, away_team_id, football_date
               FROM games WHERE source = ? AND football_date IS NOT NULL""",
            (CANONICAL_SOURCE,),
        ):
            home, away = str(row["home_team_id"]), str(row["away_team_id"])
            key = (frozenset((home, away)), str(row["football_date"]))
            index[key].append((str(row["game_id"]), home))
        return dict(index)

    def team_id(self, team: Mapping[str, Any]) -> str | None:
        """Canonical team_id for a feed team, or None when not uniquely identified.

        Probed most-reliable first. Outlier's ``market`` is the school
        ("Washington"), its ``school`` is the *nickname* ("Huskies") -- and
        three FBS programmes are the Huskies, so the nickname is tried last and
        only ever accepted when it happens to be unique.
        """
        for field in ("market", "alias", "school"):
            candidates = self._teams.get(name_key(team.get(field)))
            if candidates and len(candidates) == 1:
                return next(iter(candidates))
        return None

    def game(
        self, *, home_team_id: str, away_team_id: str, football_date: str
    ) -> GameMatch | None:
        """Canonical game for two canonical team ids on a slate date."""
        if home_team_id == away_team_id:
            return None
        pair = frozenset((home_team_id, away_team_id))
        for slate in self._candidate_dates(football_date):
            hits = self._games.get((pair, slate))
            if not hits:
                continue
            if len(hits) > 1:
                return None
            game_id, canonical_home = hits[0]
            return GameMatch(game_id, canonical_home == home_team_id)
        return None

    @staticmethod
    def _candidate_dates(football_date: str) -> list[str]:
        """The slate date, then its neighbours.

        A 02:30Z Sunday kickoff belongs to Saturday's slate. If the two feeds
        round that boundary differently the same game carries dates one day
        apart, so the exact date is tried first and the neighbours only as a
        fallback.
        """
        try:
            anchor = date.fromisoformat(football_date)
        except (TypeError, ValueError):
            return [str(football_date)]
        return [
            football_date,
            (anchor - timedelta(days=1)).isoformat(),
            (anchor + timedelta(days=1)).isoformat(),
        ]

    def event(self, record: Mapping[str, Any]) -> EventResolution:
        """Resolve one parsed feed event end to end.

        Returns the canonical ids, or a human-readable reason naming the teams
        involved -- a caller reporting "1 event unresolved" without saying
        which is the silent-partial-run failure the daily job exists to avoid.
        """
        home = record.get("home") or {}
        away = record.get("away") or {}
        label = (
            f"{away.get('market') or away.get('school') or '?'} @ "
            f"{home.get('market') or home.get('school') or '?'} "
            f"on {record.get('football_date')}"
        )

        home_id = self.team_id(home)
        away_id = self.team_id(away)
        unknown = [
            side_name
            for side_name, resolved in (
                (away.get("market") or away.get("school") or "?", away_id),
                (home.get("market") or home.get("school") or "?", home_id),
            )
            if resolved is None
        ]
        if unknown:
            return EventResolution(
                reason=f"{label}: no unique canonical team for {', '.join(unknown)}"
            )

        assert home_id is not None and away_id is not None  # narrowed by `unknown`
        match = self.game(
            home_team_id=home_id,
            away_team_id=away_id,
            football_date=str(record.get("football_date")),
        )
        if match is None:
            return EventResolution(
                reason=f"{label}: no canonical CFBD game. Run `backfill-cfbd` "
                       f"(or `daily`, which bootstraps the season) first"
            )
        if not match.orientation_agrees:
            return EventResolution(
                reason=f"{label}: feed and CFBD disagree on which team is home; "
                       f"refusing to re-key HOME/AWAY prices across the swap"
            )
        return EventResolution(
            game_id=match.game_id, home_team_id=home_id, away_team_id=away_id
        )
