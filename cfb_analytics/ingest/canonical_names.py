"""Resolve a CFBD school name onto a canonical ``cfbd:<id>`` team.

Polls and team boxes used to build that index from every ``teams`` row. A
legacy Outlier row shares the school's market name and abbreviation, so the
name either bound to the non-``cfbd:`` id or was dropped as ambiguous. Open
PR #20 applies the same predicate inside ``cfbd_rankings._team_name_index``.
This module is the copy the poll and box ingests call. That function's body
is left as it is on master so the two diffs do not edit the same lines.
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.ingest.cfbd_fundamentals import _name_key


def canonical_team_name_index(conn: sqlite3.Connection) -> dict[str, str]:
    """Normalised name to ``cfbd:<id>``.

    A name that still matches two canonical teams resolves to nothing. A name
    that matches only a legacy non-``cfbd:`` row also resolves to nothing.
    """
    rows = conn.execute(
        """SELECT team_id, school, alias, market FROM teams
           WHERE cfbd_id IS NOT NULL AND team_id = 'cfbd:' || cfbd_id"""
    ).fetchall()
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
    aliases = conn.execute(
        """SELECT a.team_id, a.alias FROM team_aliases a
           JOIN teams t ON t.team_id = a.team_id
           WHERE t.cfbd_id IS NOT NULL AND t.team_id = 'cfbd:' || t.cfbd_id"""
    ).fetchall()
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
