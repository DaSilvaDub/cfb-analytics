"""CFBD names resolve canonically even with legacy Outlier rows in the store."""

from cfb_analytics.features.ranked_slate import select_ranked_slate
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_boxes import ingest_team_boxes
from cfb_analytics.ingest.cfbd_rankings import _team_name_index, ingest_rankings
from tests.conftest import seed_canonical_game, seed_canonical_team


def _seed_collision(conn):
    penn = seed_canonical_team(conn, 213, "Penn State", "PSU")
    northwestern = seed_canonical_team(conn, 77, "Northwestern", "NU")
    store.upsert_team(
        conn,
        {
            "team_id": "legacy-outlier",
            "school": "Nittany Lions",
            "alias": "PSU",
            "market": "Penn State",
        },
    )
    store.insert_team_aliases(
        conn,
        [
            {
                "team_id": "legacy-outlier",
                "source": "outlier",
                "alias": "Penn State",
                "alias_type": "market",
            },
        ],
    )
    seed_canonical_game(
        conn,
        "cfbd:1",
        northwestern,
        penn,
        football_date="2026-10-02",
        kickoff="2026-10-03T00:00:00+00:00",
        week=6,
    )
    conn.commit()
    return penn, northwestern


def test_poll_ingestion_ignores_legacy_names_and_restores_ranked_game(conn):
    penn, _ = _seed_collision(conn)

    class Client:
        def fetch_rankings(self, *args, **kwargs):
            return [
                {
                    "season": 2026,
                    "week": 5,
                    "polls": [
                        {"poll": "AP Top 25", "ranks": [{"school": "Penn State", "rank": 8}]},
                    ],
                }
            ]

    summary = ingest_rankings(conn, Client(), 2026, week=5)
    assert summary.rows == 1
    assert summary.unresolved == 0
    assert _team_name_index(conn)["penn state"] == penn
    assert [g.game_id for g in select_ranked_slate(conn, "2026-10-02")] == ["cfbd:1"]


def test_boxes_ignore_legacy_names_and_attach_to_canonical_team(conn):
    penn, northwestern = _seed_collision(conn)

    class Client:
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

    summary = ingest_team_boxes(conn, Client(), 2026, 6)
    assert summary.rows == 2
    assert summary.unresolved == 0
    assert {r["team_id"] for r in conn.execute("SELECT team_id FROM team_game_box")} == {
        penn,
        northwestern,
    }


def test_true_canonical_name_ambiguity_remains_unresolved(conn):
    penn, _ = _seed_collision(conn)
    other = seed_canonical_team(conn, 999, "Other University", "OTHER")
    store.insert_team_aliases(
        conn,
        [
            {
                "team_id": other,
                "source": "cfbd",
                "alias": "Penn State",
                "alias_type": "alternate_name",
            },
        ],
    )
    index = _team_name_index(conn)
    assert "penn state" not in index
    assert index["psu"] == penn
