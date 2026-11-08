"""The Elo dedupe script prints a plan and does not delete unless --apply."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import parse_elo_rating


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "dedupe_elo.py"
    spec = importlib.util.spec_from_file_location("dedupe_elo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(conn, *, week: int, elo: float, ingested: str) -> None:
    store.upsert_team(
        conn, {"team_id": "cfbd:333", "school": "Alabama", "alias": "ALA", "market": None}
    )
    row = parse_elo_rating(
        {"year": 2026, "team": "Alabama", "elo": elo},
        week=week,
        as_of_utc="2026-11-01T00:00:00+00:00",
    )
    row = {**row, "team_id": "cfbd:333", "provenance_mode": "reconstructed"}
    del row["team_name"]
    store.insert_team_ratings(conn, [row])
    conn.execute(
        """UPDATE team_ratings SET ingested_utc = ?
           WHERE team_id = 'cfbd:333' AND week = ? AND rating = ?""",
        (ingested, week, elo),
    )


def test_dry_run_prints_counts_and_deletes_nothing(conn, capsys) -> None:
    _write(conn, week=10, elo=1939, ingested="2026-11-05T00:00:00+00:00")
    current = conn.execute(
        "SELECT * FROM team_ratings WHERE team_id = 'cfbd:333' AND week = 10"
    ).fetchone()
    conn.execute(
        """INSERT INTO team_ratings
           (snapshot_id, season, period, week, season_type, team_id, source, snapshot_scope,
            provenance_mode, as_of_utc, ingested_utc, rating)
           VALUES ('older-snap', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            current["season"],
            current["period"],
            current["week"],
            current["season_type"],
            current["team_id"],
            current["source"],
            current["snapshot_scope"],
            current["provenance_mode"],
            current["as_of_utc"],
            "2026-10-01T00:00:00+00:00",
            1800,
        ),
    )
    _write(conn, week=11, elo=1492.88, ingested="2026-10-25T00:00:00+00:00")
    conn.commit()
    path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    before = conn.execute("SELECT COUNT(*) AS n FROM team_ratings").fetchone()["n"]

    module = _load()
    assert module.main(["--db", str(path), "--season", "2026"]) == 0
    text = capsys.readouterr().out
    assert "DRY RUN — deletes nothing" in text
    assert "future week 11: before=1 after=0 would_delete=1" in text
    assert "week 10 duplicates: before=2 after=1 would_delete=1" in text
    after = conn.execute("SELECT COUNT(*) AS n FROM team_ratings").fetchone()["n"]
    assert after == before
