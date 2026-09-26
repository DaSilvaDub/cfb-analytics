"""Ranked-slate selection: AP Top 25 games plus each conference's top two."""

from __future__ import annotations

from cfb_analytics.features.ranked_slate import select_ranked_slate
from cfb_analytics.ingest import store
from cfb_analytics.sources.cfbd import parse_rankings
from tests.conftest import seed_canonical_game, seed_canonical_team


def _seed_conference(
    conn,
    team_id: str,
    conference: str,
    *,
    season: int = 2026,
    classification: str = "fbs",
) -> None:
    store.upsert_team_season(
        conn,
        {
            "team_id": team_id,
            "season": season,
            "source": "cfbd",
            "conference": conference,
            "division": None,
            "classification": classification,
            "venue_id": None,
        },
    )


def _seed_poll(conn, team_id: str, rank: int, *, week: int = 3, poll: str = "AP Top 25") -> None:
    store.insert_team_polls(
        conn,
        [
            {
                "season": 2026,
                "week": week,
                "season_type": "regular",
                "poll": poll,
                "rank": rank,
                "team_id": team_id,
                "points": 1600 - rank,
                "first_place_votes": 1 if rank == 1 else 0,
                "as_of_utc": "2026-09-14T16:00:00+00:00",
            }
        ],
    )


def test_selects_ap_top_25_game_and_skips_unranked_pair(conn) -> None:
    indiana = seed_canonical_team(conn, 84, "Indiana", "IU")
    wku = seed_canonical_team(conn, 98, "Western Kentucky", "WKU")
    tulsa = seed_canonical_team(conn, 202, "Tulsa", "TLSA")
    rice = seed_canonical_team(conn, 242, "Rice", "RICE")
    usf = seed_canonical_team(conn, 58, "South Florida", "USF")
    utsa = seed_canonical_team(conn, 2636, "UTSA", "UTSA")
    _seed_conference(conn, indiana, "Big Ten")
    _seed_conference(conn, wku, "Conference USA")
    _seed_conference(conn, tulsa, "American Athletic")
    _seed_conference(conn, rice, "American Athletic")
    _seed_conference(conn, usf, "American Athletic")
    _seed_conference(conn, utsa, "American Athletic")
    _seed_poll(conn, usf, 1)
    _seed_poll(conn, utsa, 2)
    _seed_poll(conn, indiana, 4)
    seed_canonical_game(
        conn,
        "cfbd:ranked",
        indiana,
        wku,
        football_date="2026-09-19",
        kickoff="2026-09-19T20:00:00+00:00",
        week=3,
    )
    seed_canonical_game(
        conn,
        "cfbd:unranked",
        tulsa,
        rice,
        football_date="2026-09-19",
        kickoff="2026-09-19T17:00:00+00:00",
        week=3,
    )
    conn.commit()

    selected = select_ranked_slate(conn, "2026-09-19")
    ids = {game.game_id for game in selected}
    assert "cfbd:ranked" in ids
    assert "cfbd:unranked" not in ids
    reasons = next(game.inclusion_reasons for game in selected if game.game_id == "cfbd:ranked")
    assert "ap_top_25" in reasons


def test_includes_conference_top_two_even_when_unranked(conn) -> None:
    toledo = seed_canonical_team(conn, 2649, "Toledo", "TOL")
    temple = seed_canonical_team(conn, 218, "Temple", "TEM")
    ohio = seed_canonical_team(conn, 195, "Ohio", "OHIO")
    _seed_conference(conn, toledo, "Mid-American")
    _seed_conference(conn, temple, "American Athletic")
    _seed_conference(conn, ohio, "Mid-American")
    store.insert_team_ratings(
        conn,
        [
            {
                "season": 2026,
                "period": "regular:week:02",
                "week": 2,
                "season_type": "regular",
                "team_id": toledo,
                "source": "elo_cfbd",
                "snapshot_scope": "weekly",
                "as_of_utc": "2026-09-14T00:00:00+00:00",
                "provenance_mode": "reconstructed",
                "rating": 1600.0,
                "ranking": None,
                "off_rating": None,
                "def_rating": None,
                "st_rating": None,
                "sos": None,
                "second_order_wins": None,
            },
            {
                "season": 2026,
                "period": "regular:week:02",
                "week": 2,
                "season_type": "regular",
                "team_id": ohio,
                "source": "elo_cfbd",
                "snapshot_scope": "weekly",
                "as_of_utc": "2026-09-14T00:00:00+00:00",
                "provenance_mode": "reconstructed",
                "rating": 1400.0,
                "ranking": None,
                "off_rating": None,
                "def_rating": None,
                "st_rating": None,
                "sos": None,
                "second_order_wins": None,
            },
        ],
    )
    seed_canonical_game(
        conn,
        "cfbd:mac",
        toledo,
        temple,
        football_date="2026-09-19",
        kickoff="2026-09-19T19:00:00+00:00",
        week=3,
    )
    conn.commit()

    selected = select_ranked_slate(conn, "2026-09-19")
    assert any(game.game_id == "cfbd:mac" for game in selected)
    mac = next(game for game in selected if game.game_id == "cfbd:mac")
    assert "conference_top_2" in mac.inclusion_reasons


def test_parse_rankings_flattens_nested_polls() -> None:
    rows = parse_rankings(
        {
            "season": 2026,
            "seasonType": "regular",
            "week": 3,
            "polls": [
                {
                    "poll": "AP Top 25",
                    "ranks": [
                        {
                            "rank": 1,
                            "school": "Texas",
                            "points": 1678,
                            "firstPlaceVotes": 56,
                        }
                    ],
                }
            ],
        },
        as_of_utc="2026-09-14T16:00:00+00:00",
    )
    assert rows == [
        {
            "season": 2026,
            "week": 3,
            "season_type": "regular",
            "poll": "AP Top 25",
            "rank": 1,
            "team_name": "Texas",
            "points": 1678,
            "first_place_votes": 56,
            "as_of_utc": "2026-09-14T16:00:00+00:00",
        }
    ]


def test_dedupes_ranked_conference_leader_game(conn) -> None:
    alabama = seed_canonical_team(conn, 333, "Alabama", "ALA")
    fsu = seed_canonical_team(conn, 52, "Florida State", "FSU")
    _seed_conference(conn, alabama, "SEC")
    _seed_conference(conn, fsu, "ACC")
    _seed_poll(conn, alabama, 10)
    seed_canonical_game(
        conn,
        "cfbd:both",
        alabama,
        fsu,
        football_date="2026-09-19",
        kickoff="2026-09-19T19:30:00+00:00",
        week=3,
    )
    conn.commit()

    selected = select_ranked_slate(conn, "2026-09-19")
    matches = [game for game in selected if game.game_id == "cfbd:both"]
    assert len(matches) == 1
    assert "ap_top_25" in matches[0].inclusion_reasons
    assert "conference_top_2" in matches[0].inclusion_reasons


def test_fcs_conference_leader_is_not_admitted(conn) -> None:
    nebraska = seed_canonical_team(conn, 158, "Nebraska", "NEB")
    ndsu = seed_canonical_team(conn, 155, "North Dakota State", "NDSU")
    _seed_conference(conn, nebraska, "Big Ten")
    _seed_conference(conn, ndsu, "Missouri Valley", classification="fcs")
    store.insert_team_ratings(
        conn,
        [
            {
                "season": 2026,
                "period": "regular:week:02",
                "week": 2,
                "season_type": "regular",
                "team_id": ndsu,
                "source": "elo_cfbd",
                "snapshot_scope": "weekly",
                "as_of_utc": "2026-09-14T00:00:00+00:00",
                "provenance_mode": "reconstructed",
                "rating": 1800.0,
                "ranking": None,
                "off_rating": None,
                "def_rating": None,
                "st_rating": None,
                "sos": None,
                "second_order_wins": None,
            }
        ],
    )
    seed_canonical_game(
        conn,
        "cfbd:fcs",
        nebraska,
        ndsu,
        football_date="2026-09-19",
        kickoff="2026-09-19T19:00:00+00:00",
        week=3,
    )
    conn.commit()
    selected = select_ranked_slate(conn, "2026-09-19")
    assert all(game.game_id != "cfbd:fcs" for game in selected)


def test_unrated_conference_does_not_pick_lexicographic_leaders(conn) -> None:
    tulsa = seed_canonical_team(conn, 202, "Tulsa", "TLSA")
    rice = seed_canonical_team(conn, 242, "Rice", "RICE")
    _seed_conference(conn, tulsa, "American Athletic")
    _seed_conference(conn, rice, "American Athletic")
    seed_canonical_game(
        conn,
        "cfbd:unrated",
        tulsa,
        rice,
        football_date="2026-09-19",
        kickoff="2026-09-19T17:00:00+00:00",
        week=3,
    )
    conn.commit()
    assert select_ranked_slate(conn, "2026-09-19") == []


def test_week_two_slate_ignores_later_ap_poll(conn) -> None:
    texas = seed_canonical_team(conn, 251, "Texas", "TEX")
    utsa = seed_canonical_team(conn, 2636, "UTSA", "UTSA")
    _seed_conference(conn, texas, "SEC")
    _seed_conference(conn, utsa, "American Athletic")
    _seed_poll(conn, texas, 1, week=10)
    seed_canonical_game(
        conn,
        "cfbd:early",
        texas,
        utsa,
        football_date="2026-09-12",
        kickoff="2026-09-12T16:00:00+00:00",
        week=2,
    )
    conn.commit()
    selected = select_ranked_slate(conn, "2026-09-12")
    assert selected == []
