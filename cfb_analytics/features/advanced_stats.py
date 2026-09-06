"""Leakage-safe net offense-minus-defense diffs from CFBD's cumulative
through-week advanced stats (``team_season_advanced``; plan section 5's T2
"EPA/success-rate/explosiveness" and T3 "line yards/stuff rate/havoc" tiers).

``team_season_advanced`` stores one row per (team, week, side) where
``side`` is ``'off'`` or ``'def'`` and each row is a CUMULATIVE snapshot
through that week -- exactly the same "weekly" availability class and
as-of-cutoff shape as CFBD's own weekly Elo (see ``features/elo_ratings.py``),
so the same ``AsOfReader.latest`` leakage guard applies here: predicting
week W must use the most recent row strictly before week W's own cutoff,
never that week's own in-progress snapshot.

CFBD's own sign convention for these fields is not symmetric between "off"
and "def", so a blind ``off - def`` is wrong for two of the seven stats:

* ``ppa``, ``success_rate``, ``explosiveness``, ``points_per_opportunity``,
  ``line_yards`` -- the ``off`` row is what this team's OFFENSE produced
  (higher is better) and the ``def`` row is what this team's DEFENSE
  ALLOWED (higher is worse). Net team quality on the stat is therefore
  ``off - def``.
* ``stuff_rate``, ``havoc`` -- the ``off`` row is what happened TO this
  team's offense (getting stuffed / suffering havoc plays -- higher is
  worse) and the ``def`` row is what this team's defense INFLICTED on
  opponents (higher is better). Net team quality is therefore the other
  way round: ``def - off``.

A missing snapshot (no admissible row for a side, or CFBD omitted a
nullable field like ``explosiveness`` for a small sample) contributes 0 to
that stat -- the same "missing means no signal, not a guess" convention
``models/shrinkage.py`` and ``features/ensemble.py`` already use for
talent/returning-production, not a fabricated league-average value.

Only 2015 onward is populated (see ``ingest/cfbd_fundamentals.py``); 2014
games simply get an all-zero (no-signal) vector here, reported honestly
rather than guessed.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from cfb_analytics.features.asof import AsOfReader

# Stats where a higher "off" value is good and a higher "def" (allowed) value is bad.
_OFF_MINUS_DEF = ("ppa", "success_rate", "explosiveness", "points_per_opportunity", "line_yards")
# Stats where a higher "off" (suffered) value is bad and a higher "def" (inflicted) value is good.
_DEF_MINUS_OFF = ("stuff_rate", "havoc")

ADVANCED_STAT_KEYS = _OFF_MINUS_DEF + _DEF_MINUS_OFF

AdvancedStatsCache = dict[str, dict[str, list[Mapping[str, Any]]]]


def load_advanced_stats(conn: sqlite3.Connection, season: int) -> AdvancedStatsCache:
    """Every ``team_season_advanced`` row for one season, grouped for
    in-memory leakage-safe lookups -- one query per season rather than one
    per game, which matters once P_logit's training set spans a decade of
    pooled cross-season history (see ``features/ensemble.py``).
    """
    rows = conn.execute(
        """SELECT team_id, side, as_of_utc, ppa, success_rate, explosiveness,
                  points_per_opportunity, havoc, line_yards, stuff_rate
           FROM team_season_advanced WHERE season = ?""",
        (season,),
    ).fetchall()
    cache: AdvancedStatsCache = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cache[row["team_id"]][row["side"]].append(dict(row))
    return cache


def team_advanced_nets(
    cache: AdvancedStatsCache,
    team_id: str,
    as_of_utc: str,
    *,
    season: int | None = None,
    reader_game_id: str = "advanced-stats-lookup",
) -> dict[str, float]:
    """Net offense-minus-defense quality on each of ``ADVANCED_STAT_KEYS``,
    as of strictly before ``as_of_utc``. Always returns all seven keys.
    """
    reader = AsOfReader(game_id=reader_game_id, kickoff_utc=as_of_utc, season=season)
    rows_by_side = cache.get(team_id, {})
    off = reader.latest(rows_by_side.get("off", []), what="team_season_advanced.off")
    defense = reader.latest(rows_by_side.get("def", []), what="team_season_advanced.def")

    def value(row: Mapping[str, Any] | None, key: str) -> float:
        if row is None or row.get(key) is None:
            return 0.0
        return float(row[key])

    nets: dict[str, float] = {}
    for key in _OFF_MINUS_DEF:
        nets[key] = value(off, key) - value(defense, key)
    for key in _DEF_MINUS_OFF:
        nets[key] = value(defense, key) - value(off, key)
    return nets
