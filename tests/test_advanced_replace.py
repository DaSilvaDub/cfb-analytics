"""A second advanced-stat fetch for the same week replaces the stored snapshot."""

from __future__ import annotations

from cfb_analytics.ingest import store


def _row(*, ppa: float, side: str = "off", week: int = 11, garbage_excluded: int = 1) -> dict:
    return {
        "season": 2026,
        "week": week,
        "team_id": "cfbd:1",
        "side": side,
        "as_of_utc": "2026-11-08T23:00:00+00:00",
        "provenance_mode": "reconstructed",
        "garbage_excluded": garbage_excluded,
        "ppa": ppa,
        "success_rate": 0.4,
    }


def test_fetching_the_same_week_twice_keeps_one_row_with_the_latest_stats(conn) -> None:
    store.upsert_team(
        conn, {"team_id": "cfbd:1", "school": "Home U", "alias": "HOM", "market": "Home U"}
    )
    store.insert_team_advanced(conn, [_row(ppa=0.1)])
    store.insert_team_advanced(
        conn,
        [
            _row(ppa=0.2, side="def"),
            _row(ppa=0.3, week=12),
            _row(ppa=0.05, garbage_excluded=0),
        ],
    )
    store.insert_team_advanced(conn, [_row(ppa=0.5)])

    stored = conn.execute(
        """SELECT week, side, garbage_excluded, ppa FROM team_season_advanced
           WHERE team_id = 'cfbd:1' AND season = 2026
           ORDER BY week, side, garbage_excluded"""
    ).fetchall()
    assert [tuple(row) for row in stored] == [
        (11, "def", 1, 0.2),
        (11, "off", 0, 0.05),
        (11, "off", 1, 0.5),
        (12, "off", 1, 0.3),
    ]
