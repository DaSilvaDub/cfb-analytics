"""Poll and team-box names resolve only onto canonical ``cfbd:`` team ids.

On master, ``_team_name_index`` reads every ``teams`` row. A legacy Outlier
row whose market is "Penn State" (team_id not ``cfbd:``) makes that name
either bind to the legacy id or drop out as ambiguous. Open PR #20 restricts
``cfbd_rankings._team_name_index`` and is not merged; this test covers the
poll and box call sites without editing that function body.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.canonical_names import canonical_team_name_index
from cfb_analytics.ingest.cfbd_boxes import ingest_team_boxes
from cfb_analytics.ingest.cfbd_rankings import ingest_rankings
from tests.conftest import seed_canonical_game, seed_canonical_team

LEGACY_ID = "1883100bf4e7fbd3e5233730985cb9c84a889228"


def _legacy_penn_state(conn, team_id: str = LEGACY_ID) -> None:
    store.upsert_team(
        conn,
        {
            "team_id": team_id,
            "school": "Nittany Lions",
            "alias": "PSU",
            "market": "Penn State",
        },
    )
    store.insert_team_aliases(
        conn,
        [
            {
                "team_id": team_id,
                "source": "outlier",
                "alias": "Penn State",
                "alias_type": "market",
            }
        ],
    )


def _ap_payload(*schools: tuple[str, int]) -> list[dict]:
    return [
        {
            "season": 2026,
            "week": 5,
            "polls": [
                {
                    "poll": "AP Top 25",
                    "ranks": [{"school": school, "rank": rank} for school, rank in schools],
                }
            ],
        }
    ]


class _RankClient:
    def __init__(self, payload):
        self.payload = payload

    def fetch_rankings(self, *args, **kwargs):
        return self.payload


class _BoxClient:
    def fetch_game_teams(self, *args, **kwargs):
        return [
            {
                "id": 1,
                "teams": [
                    {
                        "team": "Penn State",
                        "stats": [{"category": "rushingYards", "stat": 200}],
                    },
                    {"team": "Northwestern", "stats": []},
                ],
            }
        ]


def test_collision_resolves_poll_and_box_to_the_canonical_id(conn):
    penn = seed_canonical_team(conn, 213, "Penn State", "PSU")
    northwestern = seed_canonical_team(conn, 77, "Northwestern", "NU")
    _legacy_penn_state(conn)
    seed_canonical_game(conn, "cfbd:1", northwestern, penn, week=6)
    before = conn.execute("SELECT COUNT(*) AS n FROM teams").fetchone()["n"]

    assert canonical_team_name_index(conn)["penn state"] == penn

    summary = ingest_rankings(conn, _RankClient(_ap_payload(("Penn State", 8))), 2026, week=5)
    assert summary.rows == 1
    assert summary.unresolved == 0
    assert [row["team_id"] for row in conn.execute("SELECT team_id FROM team_polls")] == [penn]

    boxes = ingest_team_boxes(conn, _BoxClient(), 2026, 6)
    assert boxes.rows == 2
    assert boxes.unresolved == 0
    assert {row["team_id"] for row in conn.execute("SELECT team_id FROM team_game_box")} == {
        penn,
        northwestern,
    }
    assert conn.execute("SELECT COUNT(*) AS n FROM teams").fetchone()["n"] == before


def test_legacy_only_name_does_not_bind_or_create_a_team(conn):
    northwestern = seed_canonical_team(conn, 77, "Northwestern", "NU")
    seed_canonical_team(conn, 84, "Indiana", "IU")
    _legacy_penn_state(conn, team_id="legacy-only")
    seed_canonical_game(conn, "cfbd:1", northwestern, "cfbd:84", week=6)
    before = {row["team_id"] for row in conn.execute("SELECT team_id FROM teams")}

    assert "penn state" not in canonical_team_name_index(conn)

    summary = ingest_rankings(
        conn,
        _RankClient(_ap_payload(("Penn State", 8), ("Northwestern", 12))),
        2026,
        week=5,
    )
    assert summary.rows == 1
    assert summary.unresolved == 1
    assert [row["team_id"] for row in conn.execute("SELECT team_id FROM team_polls")] == [
        northwestern
    ]

    boxes = ingest_team_boxes(conn, _BoxClient(), 2026, 6)
    assert boxes.unresolved == 1
    stored = {row["team_id"] for row in conn.execute("SELECT team_id FROM team_game_box")}
    assert stored == {northwestern}
    assert "legacy-only" not in stored
    after = {row["team_id"] for row in conn.execute("SELECT team_id FROM teams")}
    assert after == before


def test_a_week_of_only_legacy_names_writes_no_poll_row(conn):
    _legacy_penn_state(conn, team_id="legacy-only")
    before = conn.execute("SELECT COUNT(*) AS n FROM teams").fetchone()["n"]
    with pytest.raises(SchemaError, match="resolved zero teams"):
        ingest_rankings(conn, _RankClient(_ap_payload(("Penn State", 8))), 2026, week=5)
    assert conn.execute("SELECT COUNT(*) AS n FROM team_polls").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM teams").fetchone()["n"] == before


def test_two_canonical_teams_sharing_a_name_stay_unresolved(conn):
    penn = seed_canonical_team(conn, 213, "Penn State", "PSU")
    other = seed_canonical_team(conn, 999, "Other University", "OTHER")
    store.insert_team_aliases(
        conn,
        [
            {
                "team_id": other,
                "source": "cfbd",
                "alias": "Penn State",
                "alias_type": "alternate_name",
            }
        ],
    )
    index = canonical_team_name_index(conn)
    assert "penn state" not in index
    assert index["psu"] == penn


def test_ap_week_dry_run_lists_weeks_5_through_9_and_does_not_ingest(capsys, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "reingest_ap_weeks.py"
    spec = importlib.util.spec_from_file_location("reingest_ap_weeks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.commands(2026) == [
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 5",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 6",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 7",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 8",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 9",
    ]
    assert module.main([]) == 0
    captured = capsys.readouterr()
    assert "DRY RUN — not executed" in captured.out
    assert "week 5" in captured.out and "week 9" in captured.out
    assert not (tmp_path / "data").exists() or not list((tmp_path / "data").rglob("*"))
