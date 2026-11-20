"""Rows stored as inferred_mark must not stay playable.

Migration 018 added ``publish_status`` with default ``'playable'``. A snapshot
whose ``hold_reason`` is ``inferred_mark`` kept that default. Count the rows
this migration will rewrite before it runs:

    SELECT COUNT(*) FROM over_board_snapshots
    WHERE hold_reason = 'inferred_mark' AND publish_status != 'held';
"""

from __future__ import annotations

import sqlite3

from cfb_analytics.db import MIGRATIONS, _connect, applied_versions, migrate
from cfb_analytics.ingest import store

_THROUGH = 19


def _apply_through(conn: sqlite3.Connection, version: int) -> None:
    applied_versions(conn)
    for ver, name, sql, hook in MIGRATIONS:
        if ver > version:
            break
        conn.executescript(sql)
        if hook is not None:
            hook(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_utc) VALUES (?, ?, 'x')",
            (ver, name),
        )
    conn.commit()


def _snapshot(
    conn: sqlite3.Connection,
    *,
    rank: int,
    pick: str,
    publish_status: str,
    hold_reason: str | None,
) -> None:
    conn.execute(
        """INSERT INTO over_board_snapshots
           (slate_date, rank, game_id, market, pick, team_id, side, line, projected,
            over_prob, confidence, tier, family, kickoff_utc, kickoff_et, game_label,
            flags, inclusion_reasons, published_utc, publish_status, hold_reason,
            actionable)
           VALUES ('2026-09-19', ?, 'g-hold', 'team_rushing_yards', ?, 'cfbd:1', 'OVER',
                   170.5, 200.0, 0.62, 6.5, 'MEDIUM', 'favorite_team_rush',
                   '2026-09-19T20:00:00+00:00', '4:00 p.m.', 'Away at Home',
                   '[]', '[]', '2026-09-19T16:00:00+00:00', ?, ?, 0)""",
        (rank, pick, publish_status, hold_reason),
    )


def test_inferred_mark_rows_are_held_and_other_rows_are_not(tmp_path) -> None:
    conn = _connect(tmp_path / "pre.sqlite3")
    _apply_through(conn, _THROUGH)
    store.upsert_team(
        conn, {"team_id": "cfbd:1", "school": "Home U", "alias": "HOM", "market": "Home U"}
    )
    store.upsert_team(
        conn, {"team_id": "cfbd:2", "school": "Away U", "alias": "AWY", "market": "Away U"}
    )
    store.upsert_cfbd_game(
        conn,
        {
            "game_id": "g-hold",
            "season": 2026,
            "football_date": "2026-09-19",
            "kickoff_utc": "2026-09-19T20:00:00+00:00",
            "home_team_id": "cfbd:1",
            "away_team_id": "cfbd:2",
            "completed": 0,
            "status": "scheduled",
        },
    )
    _snapshot(conn, rank=1, pick="inferred", publish_status="playable", hold_reason="inferred_mark")
    _snapshot(
        conn, rank=2, pick="no spread", publish_status="playable", hold_reason="no_spread_consensus"
    )
    _snapshot(conn, rank=3, pick="posted", publish_status="playable", hold_reason=None)
    _snapshot(conn, rank=4, pick="already", publish_status="held", hold_reason="inferred_mark")
    conn.commit()

    pending = conn.execute(
        """SELECT COUNT(*) AS n FROM over_board_snapshots
           WHERE hold_reason = 'inferred_mark' AND publish_status != 'held'"""
    ).fetchone()["n"]
    assert pending == 1

    applied = migrate(conn)
    assert 20 in applied

    rows = {
        row["pick"]: row["publish_status"]
        for row in conn.execute("SELECT pick, publish_status FROM over_board_snapshots")
    }
    assert rows == {
        "inferred": "held",
        "no spread": "playable",
        "posted": "playable",
        "already": "held",
    }
    conn.close()
