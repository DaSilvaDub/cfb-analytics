"""Board snapshots, team boxes, posted-line fallback, family settlement."""

from __future__ import annotations

import logging

from cfb_analytics.errors import SourceError
from cfb_analytics.features.over_confidence import (
    classify_family,
    load_published_board,
    persist_over_board,
    settle_over_board,
)
from cfb_analytics.ingest.cfbd_boxes import parse_team_boxes
from cfb_analytics.ingest.store import insert_team_game_boxes
from cfb_analytics.models.team_props import default_team_props_inputs
from tests.conftest import seed_canonical_game, seed_canonical_team
from tests.test_over_confidence import _insert_spread
from tests.test_ranked_slate import _seed_conference, _seed_poll


def test_classify_family() -> None:
    assert classify_family("team_rushing_yards", -16.5) == "favorite_team_rush"
    assert classify_family("team_receiving_yards", 1.5) == "team_rec"
    assert classify_family("game_rushing_yards", -3.0) == "game_yards"


def test_parse_team_boxes_reads_rush_and_pass() -> None:
    rows = parse_team_boxes(
        {
            "id": 401856685,
            "teams": [
                {
                    "team": "Alabama",
                    "points": 50,
                    "stats": [
                        {"category": "rushingYards", "stat": "249"},
                        {"category": "netPassingYards", "stat": "320"},
                        {"category": "totalYards", "stat": "569"},
                    ],
                },
                {
                    "team": "Florida State",
                    "points": 36,
                    "stats": [
                        {"category": "rushingYards", "stat": "94"},
                        {"category": "netPassingYards", "stat": "377"},
                        {"category": "totalYards", "stat": "471"},
                    ],
                },
            ],
        }
    )
    by_name = {row["team_name"]: row for row in rows}
    assert by_name["Alabama"]["rushing_yards"] == 249
    assert by_name["Alabama"]["net_passing_yards"] == 320
    assert by_name["Florida State"]["game_id"] == "cfbd:401856685"


def test_passing_yards_without_net_passing_yards_is_null() -> None:
    rows = parse_team_boxes(
        {
            "id": 1,
            "teams": [
                {
                    "team": "Alabama",
                    "stats": [
                        {"category": "rushingYards", "stat": "100"},
                        {"category": "passingYards", "stat": "250"},
                    ],
                }
            ],
        }
    )
    assert rows[0]["rushing_yards"] == 100
    assert rows[0]["net_passing_yards"] is None


def test_snapshot_is_source_of_truth_for_settle(conn) -> None:
    indiana = seed_canonical_team(conn, 84, "Indiana", "IU")
    wku = seed_canonical_team(conn, 98, "Western Kentucky", "WKU")
    _seed_conference(conn, indiana, "Big Ten")
    _seed_poll(conn, indiana, 4)
    seed_canonical_game(
        conn,
        "cfbd:iu",
        indiana,
        wku,
        football_date="2026-09-19",
        kickoff="2026-09-19T20:00:00+00:00",
        week=3,
    )
    _insert_spread(conn, "cfbd:iu", -44.5)
    persist_over_board(
        conn,
        "2026-09-19",
        [
            {
                "rank": 1,
                "game_id": "cfbd:iu",
                "game_label": "Western Kentucky at Indiana",
                "kickoff_et": "4:00 p.m.",
                "kickoff_utc": "2026-09-19T20:00:00+00:00",
                "market": "team_rushing_yards",
                "pick": "IU rush",
                "team_id": indiana,
                "side": "OVER",
                "line": 170.5,
                "projected": 242.0,
                "over_prob": 0.84,
                "confidence": 9.2,
                "tier": "HIGH",
                "family": "favorite_team_rush",
                "inclusion_reasons": ("ap_top_25",),
                "flags": ("inferred_mark", "blowout_script"),
            },
            {
                "rank": 2,
                "game_id": "cfbd:iu",
                "game_label": "Western Kentucky at Indiana",
                "kickoff_et": "4:00 p.m.",
                "kickoff_utc": "2026-09-19T20:00:00+00:00",
                "market": "game_rushing_yards",
                "pick": "WKU/IU game rush",
                "team_id": None,
                "side": "OVER",
                "line": 310.5,
                "projected": 360.0,
                "over_prob": 0.72,
                "confidence": 7.8,
                "tier": "HIGH",
                "family": "game_yards",
                "inclusion_reasons": ("ap_top_25",),
                "flags": ("inferred_mark",),
            },
        ],
        as_of_utc="2026-09-19T16:00:00+00:00",
    )
    insert_team_game_boxes(
        conn,
        [
            {
                "game_id": "cfbd:iu",
                "team_id": indiana,
                "rushing_yards": 233.0,
                "net_passing_yards": 280.0,
                "total_yards": 513.0,
                "points": 38,
            },
            {
                "game_id": "cfbd:iu",
                "team_id": wku,
                "rushing_yards": 90.0,
                "net_passing_yards": 210.0,
                "total_yards": 300.0,
                "points": 10,
            },
        ],
    )
    conn.commit()
    published = load_published_board(conn, "2026-09-19")
    assert len(published) == 2
    assert published[0].line == 170.5
    report = settle_over_board(conn, "2026-09-19")
    assert report.evaluated == 2
    assert report.cash == 2
    assert report.by_family["favorite_team_rush"] == (1, 1)
    assert report.by_family["game_yards"] == (1, 1)
    assert report.source == "snapshot"
    # Empty rebuild after kickoff must not wipe the freeze.
    persist_over_board(conn, "2026-09-19", [], as_of_utc="2026-09-20T04:00:00+00:00")
    persist_over_board(
        conn,
        "2026-09-19",
        [
            {
                "rank": 1,
                "game_id": "cfbd:iu",
                "game_label": "Western Kentucky at Indiana",
                "kickoff_et": "4:00 p.m.",
                "kickoff_utc": "2026-09-19T20:00:00+00:00",
                "market": "team_rushing_yards",
                "pick": "IU rush",
                "team_id": indiana,
                "side": "OVER",
                "line": 99.5,
                "projected": 100.0,
                "over_prob": 0.51,
                "confidence": 5.6,
                "tier": "LEAN",
                "family": "favorite_team_rush",
                "inclusion_reasons": ("ap_top_25",),
                "flags": (),
            }
        ],
        as_of_utc="2026-09-20T04:00:00+00:00",
    )
    frozen = load_published_board(conn, "2026-09-19")
    assert frozen[0].line == 170.5
    assert len(frozen) == 2


def test_posted_rush_line_from_two_book_snapshots(conn) -> None:
    indiana = seed_canonical_team(conn, 84, "Indiana", "IU")
    wku = seed_canonical_team(conn, 98, "Western Kentucky", "WKU")
    _seed_conference(conn, indiana, "Big Ten")
    _seed_poll(conn, indiana, 4)
    seed_canonical_game(
        conn,
        "cfbd:iu",
        indiana,
        wku,
        football_date="2026-09-19",
        kickoff="2026-09-19T20:00:00+00:00",
        week=3,
    )
    _insert_spread(conn, "cfbd:iu", -16.5)
    for book, snapshot_id in (("DK", "s-dk"), ("PINNACLE", "s-pin")):
        conn.execute(
            """INSERT INTO team_prop_odds_snapshots
               (snapshot_id, game_id, team_id, market_id, book, captured_utc,
                market, side, line, price_american, price_decimal, is_primary, source)
               VALUES (?, 'cfbd:iu', ?, NULL, ?, '2026-09-19T12:00:00+00:00',
                       'team_rushing_yards', 'OVER', 170.5, -110, 1.91, 1, 'outlier')""",
            (snapshot_id, indiana, book),
        )
    conn.execute(
        """INSERT INTO team_prop_odds_snapshots
           (snapshot_id, game_id, team_id, market_id, book, captured_utc,
            market, side, line, price_american, price_decimal, is_primary, source)
           VALUES ('s-late', 'cfbd:iu', ?, NULL, 'FD', '2026-09-19T21:00:00+00:00',
                   'team_rushing_yards', 'OVER', 99.5, -110, 1.91, 1, 'outlier')""",
        (indiana,),
    )
    conn.commit()
    from cfb_analytics.features.over_confidence import build_over_confidence_board

    board = build_over_confidence_board(
        conn,
        "2026-09-19",
        inputs_by_team={
            indiana: default_team_props_inputs(
                expected_rushing_attempts=48.0,
                expected_pass_attempts=24.0,
                data_quality_score=100.0,
            ),
            wku: default_team_props_inputs(data_quality_score=100.0),
        },
    )
    iu_rush = next(row for row in board if row.pick == "IU rush")
    assert iu_rush.line == 170.5
    assert "inferred_mark" not in iu_rush.flags
    assert iu_rush.family == "favorite_team_rush"


def test_ingest_completed_boxes_stores_rush_and_pass(conn) -> None:
    alabama = seed_canonical_team(conn, 333, "Alabama", "ALA")
    fsu = seed_canonical_team(conn, 52, "Florida State", "FSU")
    seed_canonical_game(
        conn,
        "cfbd:401856685",
        alabama,
        fsu,
        football_date="2026-09-19",
        kickoff="2026-09-19T00:00:00+00:00",
        week=3,
    )
    conn.execute("UPDATE games SET completed = 1 WHERE game_id = 'cfbd:401856685'")
    conn.commit()

    class FakeBoxClient:
        def fetch_game_teams(self, year, week, *, season_type="regular"):
            assert year == 2026
            assert week == 3
            return [
                {
                    "id": 401856685,
                    "teams": [
                        {
                            "team": "Alabama",
                            "points": 50,
                            "stats": [
                                {"category": "rushingYards", "stat": "249"},
                                {"category": "netPassingYards", "stat": "320"},
                            ],
                        },
                        {
                            "team": "Florida State",
                            "points": 36,
                            "stats": [
                                {"category": "rushingYards", "stat": "94"},
                                {"category": "netPassingYards", "stat": "377"},
                            ],
                        },
                    ],
                }
            ]

    from cfb_analytics.ingest.cfbd_boxes import ingest_completed_boxes

    summary = ingest_completed_boxes(conn, FakeBoxClient(), 2026)
    assert summary.rows == 2
    ala = conn.execute(
        "SELECT rushing_yards, net_passing_yards FROM team_game_box "
        "WHERE game_id = 'cfbd:401856685' AND team_id = ?",
        (alabama,),
    ).fetchone()
    assert ala["rushing_yards"] == 249
    assert ala["net_passing_yards"] == 320


def _seed_completed_weeks(conn):
    alabama = seed_canonical_team(conn, 333, "Alabama", "ALA")
    fsu = seed_canonical_team(conn, 52, "Florida State", "FSU")
    seed_canonical_game(
        conn,
        "cfbd:1",
        alabama,
        fsu,
        football_date="2026-09-05",
        kickoff="2026-09-05T23:00:00+00:00",
        week=1,
    )
    seed_canonical_game(
        conn,
        "cfbd:12",
        alabama,
        fsu,
        football_date="2026-11-21",
        kickoff="2026-11-21T23:00:00+00:00",
        week=12,
    )
    conn.execute("UPDATE games SET completed = 1")
    conn.commit()
    return alabama, fsu


def _week_payload(raw_id: int) -> list[dict]:
    return [
        {
            "id": raw_id,
            "teams": [
                {
                    "team": "Alabama",
                    "points": 30,
                    "stats": [
                        {"category": "rushingYards", "stat": "180"},
                        {"category": "netPassingYards", "stat": "210"},
                    ],
                },
                {
                    "team": "Florida State",
                    "points": 10,
                    "stats": [
                        {"category": "rushingYards", "stat": "40"},
                        {"category": "netPassingYards", "stat": "90"},
                    ],
                },
            ],
        }
    ]


def test_weeks_with_stored_boxes_are_not_refetched(conn) -> None:
    alabama, fsu = _seed_completed_weeks(conn)
    insert_team_game_boxes(
        conn,
        [
            {
                "game_id": "cfbd:1",
                "team_id": alabama,
                "rushing_yards": 10.0,
                "net_passing_yards": 20.0,
                "total_yards": 30.0,
                "points": 7,
            },
            {
                "game_id": "cfbd:1",
                "team_id": fsu,
                "rushing_yards": 5.0,
                "net_passing_yards": 15.0,
                "total_yards": 20.0,
                "points": 3,
            },
        ],
    )

    calls: list[int] = []

    class Client:
        def fetch_game_teams(self, year, week, *, season_type="regular"):
            calls.append(week)
            return _week_payload(1 if week == 1 else 12)

    from cfb_analytics.ingest.cfbd_boxes import ingest_completed_boxes

    summary = ingest_completed_boxes(conn, Client(), 2026)
    assert calls == [12]
    assert summary.rows == 2
    stored = conn.execute(
        "SELECT rushing_yards FROM team_game_box WHERE game_id = 'cfbd:1' AND team_id = ?",
        (alabama,),
    ).fetchone()
    assert stored["rushing_yards"] == 10.0


def test_a_week_429_does_not_block_a_later_week(conn, caplog) -> None:
    _seed_completed_weeks(conn)

    class Client:
        def fetch_game_teams(self, year, week, *, season_type="regular"):
            if week == 1:
                raise SourceError(
                    "[cfbd] HTTP 429 for https://api.collegefootballdata.com/games/teams"
                )
            return _week_payload(12)

    from cfb_analytics.ingest.cfbd_boxes import ingest_completed_boxes

    with caplog.at_level(logging.WARNING, logger="cfb_analytics.ingest.cfbd_boxes"):
        summary = ingest_completed_boxes(conn, Client(), 2026)
    assert summary.rows == 2
    assert any("429" in record.getMessage() for record in caplog.records)
    landed = conn.execute(
        "SELECT COUNT(*) AS n FROM team_game_box WHERE game_id = 'cfbd:12'"
    ).fetchone()
    assert landed["n"] == 2
