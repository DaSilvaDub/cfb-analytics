"""The leakage guard.

Every feature read for a game goes through an ``AsOfReader`` bound to that
game's kickoff. The reader refuses any row timestamped at or after kickoff, and
refuses any feature whose declared availability class does not match the data
being handed to it.

This is deliberately fatal rather than a warning. A backtest that leaks looks
*excellent* -- better than a real model -- and is worthless. The only safe
failure mode is a crash during development, not an optimistic number in a
report six weeks later.

Three availability classes, declared per feature in ``spec.FEATURE_SPEC``:

* ``preseason`` -- fixed before week 1 and legitimately known all season
  (returning production, recruiting talent, coach tenure). Safe at any kickoff
  in that season, but **must not** be sourced from a season-final aggregate.
* ``weekly``    -- recomputed as the season progresses (efficiency ratings,
  Elo, records). Only rows strictly before kickoff are admissible.
* ``pregame``   -- settles hours before kickoff (odds, injury designations,
  weather forecast). Same rule, but additionally carries a staleness measure,
  because a 3-day-old "current" price is not a current price.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from cfb_analytics.errors import LeakageError

AVAILABILITY_CLASSES = ("preseason", "weekly", "pregame")

T = TypeVar("T")


def _parse(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise LeakageError(
            f"Cannot enforce the as-of rule: {field}={value!r} is not a parseable timestamp. "
            "An unparseable timestamp is treated as a leak, not as missing data."
        ) from exc


@dataclass(frozen=True)
class AsOfReader:
    """Reads rows for one game, refusing anything not knowable before kickoff.

    ``season`` is required so that a ``preseason`` feature can be checked
    against the right season -- last season's returning production is fine for
    *this* season's week 1 only if it is labelled with this season.
    """

    game_id: str
    kickoff_utc: str
    season: int | None = None

    @property
    def kickoff(self) -> datetime:
        return _parse(self.kickoff_utc, field="kickoff_utc")

    def check(self, as_of: Any, *, what: str) -> datetime:
        """Assert a single timestamp precedes kickoff. Returns it parsed."""
        stamp = _parse(as_of, field=f"{what}.as_of_utc")
        if stamp >= self.kickoff:
            raise LeakageError(
                f"Leak in {what} for game {self.game_id}: row is stamped "
                f"{stamp.isoformat()}, at or after kickoff {self.kickoff.isoformat()}. "
                "Only information available strictly before kickoff may be used."
            )
        return stamp

    def admissible(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        what: str,
        as_of_field: str = "as_of_utc",
    ) -> list[Mapping[str, Any]]:
        """Filter rows to those knowable before kickoff.

        Rows at or after kickoff are **dropped**, not raised on -- a store
        legitimately holds post-kickoff rows (settlement, closing lines) and
        filtering them is this method's whole job. Use ``check`` when a row is
        supposed to be pre-kickoff and its being late indicates a bug.
        """
        kickoff = self.kickoff
        kept = []
        for row in rows:
            raw = row.get(as_of_field)
            if raw is None:
                raise LeakageError(
                    f"Row in {what} for game {self.game_id} has no {as_of_field}; "
                    "a row without a timestamp cannot be proven pre-kickoff."
                )
            if _parse(raw, field=f"{what}.{as_of_field}") < kickoff:
                kept.append(row)
        return kept

    def latest(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        what: str,
        as_of_field: str = "as_of_utc",
    ) -> Mapping[str, Any] | None:
        """The most recent admissible row, or None."""
        kept = self.admissible(rows, what=what, as_of_field=as_of_field)
        if not kept:
            return None
        return max(kept, key=lambda row: _parse(row[as_of_field], field=what))

    def staleness_minutes(self, as_of: Any) -> float:
        """Minutes between a row's timestamp and kickoff. Larger = staler."""
        stamp = _parse(as_of, field="as_of_utc")
        return (self.kickoff - stamp).total_seconds() / 60.0

    def check_availability_class(self, availability: str, *, feature: str) -> None:
        if availability not in AVAILABILITY_CLASSES:
            raise LeakageError(
                f"Feature {feature!r} declares availability {availability!r}, which is not "
                f"one of {AVAILABILITY_CLASSES}. An undeclared class cannot be checked, "
                "so it is refused."
            )

    def check_preseason_season(self, row_season: Any, *, feature: str) -> None:
        """A preseason feature must be labelled with this game's season.

        The classic leak: joining a *season-final* rating onto week 3 of that
        same season. Labelled 'preseason' but computed with hindsight.
        """
        if self.season is None:
            raise LeakageError(
                f"Feature {feature!r} is preseason but the reader for game {self.game_id} "
                "has no season, so the label cannot be verified."
            )
        if row_season is None or int(row_season) != int(self.season):
            raise LeakageError(
                f"Leak in {feature!r} for game {self.game_id}: preseason row is labelled "
                f"season {row_season!r} but the game is in season {self.season}."
            )


def reader_for_game(game: Mapping[str, Any]) -> AsOfReader:
    missing = [k for k in ("game_id", "kickoff_utc") if not game.get(k)]
    if missing:
        raise LeakageError(f"Cannot build an AsOfReader; game is missing {missing}")
    return AsOfReader(
        game_id=str(game["game_id"]),
        kickoff_utc=str(game["kickoff_utc"]),
        season=game.get("season"),
    )
