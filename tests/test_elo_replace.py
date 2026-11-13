"""A second Elo fetch for the same week replaces the stored rating."""

from __future__ import annotations

from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import parse_elo_rating


def _rating(conn, elo: float) -> None:
    store.upsert_team(
        conn, {"team_id": "cfbd:333", "school": "Alabama", "alias": "ALA", "market": "Alabama"}
    )
    row = parse_elo_rating(
        {"year": 2026, "team": "Alabama", "elo": elo},
        week=10,
        as_of_utc="2026-11-09T03:30:00+00:00",
    )
    row = {**row, "team_id": "cfbd:333", "provenance_mode": "reconstructed"}
    del row["team_name"]
    store.insert_team_ratings(conn, [row])


def test_fetching_the_same_week_twice_keeps_one_row_with_the_latest_rating(conn) -> None:
    _rating(conn, 1863)
    _rating(conn, 1939)
    stored = conn.execute(
        """SELECT rating FROM team_ratings
           WHERE team_id = 'cfbd:333' AND season = 2026 AND week = 10
             AND source = 'elo_cfbd' AND snapshot_scope = 'weekly'"""
    ).fetchall()
    assert len(stored) == 1
    assert stored[0]["rating"] == 1939
