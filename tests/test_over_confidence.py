"""OVER confidence board for team/game rushing and receiving."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.over_confidence import (
    apply_script_adjustment,
    build_over_confidence_board,
    format_kickoff_et,
    inferred_line,
    over_confidence_from_prob,
    settle_over_board,
)
from cfb_analytics.models.team_props import default_team_props_inputs
from tests.conftest import seed_canonical_game, seed_canonical_team
from tests.test_ranked_slate import _seed_conference, _seed_poll


def test_format_kickoff_et_is_eastern() -> None:
    # 20:00 UTC Saturday is 4:00 p.m. ET.
    assert format_kickoff_et("2026-09-19T20:00:00+00:00") == "4:00 p.m."
    assert format_kickoff_et("2026-09-19T16:00:00+00:00") == "12:00 p.m."


def test_blowout_favorite_shifts_volume_to_rush() -> None:
    baseline = default_team_props_inputs(data_quality_score=100.0)
    adjusted = apply_script_adjustment(baseline, spread=-44.5)
    assert adjusted.expected_rushing_attempts > baseline.expected_rushing_attempts
    assert adjusted.expected_pass_attempts < baseline.expected_pass_attempts


def test_missing_spread_does_not_apply_toss_up_script() -> None:
    baseline = default_team_props_inputs(data_quality_score=100.0)
    assert apply_script_adjustment(baseline, spread=None) == baseline


def test_trailing_dog_shifts_volume_to_pass() -> None:
    baseline = default_team_props_inputs(data_quality_score=100.0)
    adjusted = apply_script_adjustment(baseline, spread=44.5)
    assert adjusted.expected_pass_attempts > baseline.expected_pass_attempts
    assert adjusted.expected_rushing_attempts < baseline.expected_rushing_attempts


def test_close_game_does_not_inflate_pass_without_leaky_defense() -> None:
    baseline = default_team_props_inputs(
        expected_pass_attempts=32.0,
        expected_rushing_attempts=36.0,
        data_quality_score=100.0,
    )
    adjusted = apply_script_adjustment(baseline, spread=1.5, opp_pass_allowed=180.0)
    assert adjusted.expected_pass_attempts == baseline.expected_pass_attempts
    assert adjusted.expected_rushing_attempts >= baseline.expected_rushing_attempts


def test_close_game_inflates_pass_only_when_pass_heavy_and_leaky() -> None:
    baseline = default_team_props_inputs(
        expected_pass_attempts=42.0,
        expected_rushing_attempts=22.0,
        data_quality_score=100.0,
    )
    adjusted = apply_script_adjustment(baseline, spread=1.5, opp_pass_allowed=280.0)
    assert adjusted.expected_pass_attempts > baseline.expected_pass_attempts


def test_calibrated_over_prob_never_prints_ninety_nine() -> None:
    from cfb_analytics.features.over_confidence import calibrated_over_prob

    p = calibrated_over_prob("team_receiving_yards", 448.0, 290.5, posted=False)
    assert p <= 0.84
    assert p >= 0.70


def test_inferred_line_blends_opponent_defense() -> None:
    league_mark = inferred_line(350.0, market="team_receiving_yards")
    stout_mark = inferred_line(350.0, market="team_receiving_yards", opp_allowed=120.0)
    leaky_mark = inferred_line(350.0, market="team_receiving_yards", opp_allowed=320.0)
    assert stout_mark < league_mark < leaky_mark


def test_inferred_line_is_book_like_not_the_projection() -> None:
    """Books regress pass-happy cupcake tape toward league average.

    A 442-yard receiving projection must not post a 440.5 mark, or every OVER
    collapses to a coin flip and SMU rec cannot rank as a high-prob outcome.
    """
    assert inferred_line(187.0, market="team_receiving_yards") == 185.5
    mark = inferred_line(442.0, market="team_receiving_yards")
    assert mark <= 320.5
    assert mark < 442.0 - 50.0


def _insert_spread(conn, game_id: str, home_line: float, *, n_books: int = 14) -> None:
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'SPREAD', ?, 'HOME', '2026-09-19T12:00:00+00:00', ?, -110,
                   -110, 'DK', 0.04, 'all_books', 0.52, 0.52, 0.52, 0.01, '[]')""",
        (game_id, home_line, n_books),
    )


def test_pass_happy_close_game_receiving_over_ranks_high(conn) -> None:
    smu = seed_canonical_team(conn, 2567, "SMU", "SMU")
    louisville = seed_canonical_team(conn, 97, "Louisville", "LOU")
    _seed_conference(conn, smu, "ACC")
    _seed_conference(conn, louisville, "ACC")
    _seed_poll(conn, smu, 16)
    seed_canonical_game(
        conn,
        "cfbd:smu",
        louisville,
        smu,
        football_date="2026-09-19",
        kickoff="2026-09-19T19:30:00+00:00",
        week=3,
    )
    _insert_spread(conn, "cfbd:smu", -1.5, n_books=14)
    conn.commit()
    rec_inputs = default_team_props_inputs(
        expected_pass_attempts=42.0,
        expected_rushing_attempts=22.0,
        completion_probability=0.78,
        yards_per_completion=13.5,
        data_quality_score=50.0,
    )
    board = build_over_confidence_board(
        conn,
        "2026-09-19",
        inputs_by_team={
            smu: rec_inputs,
            louisville: replace(rec_inputs, expected_pass_attempts=32.0, yards_per_completion=12.0),
        },
        min_prob=0.50,
    )
    smu_rec = next(
        row for row in board if row.market == "team_receiving_yards" and row.team_id == smu
    )
    assert smu_rec.over_prob >= 0.70
    assert smu_rec.over_prob <= 0.84
    assert smu_rec.tier == "HIGH"
    assert smu_rec.rank <= 3
    assert smu_rec.projected - smu_rec.line >= 40.0
    assert "inferred_mark" in smu_rec.flags


def test_blowout_favorite_receiving_is_vetoed(conn) -> None:
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
    _insert_spread(conn, "cfbd:iu", -44.5, n_books=12)
    conn.commit()
    inputs = default_team_props_inputs(
        expected_pass_attempts=28.0,
        expected_rushing_attempts=48.0,
        data_quality_score=50.0,
    )
    board = build_over_confidence_board(
        conn,
        "2026-09-19",
        inputs_by_team={indiana: inputs, wku: inputs},
        min_prob=0.50,
    )
    recs = [row.pick for row in board if row.market == "team_receiving_yards"]
    assert "IU rec" not in recs
    assert "WKU rec" not in recs
    assert any(row.pick == "IU rush" for row in board)
    assert all(row.market != "game_receiving_yards" for row in board)
    assert all(row.market != "game_rushing_yards" for row in board)


def test_confidence_scale_matches_session_card() -> None:
    assert over_confidence_from_prob(0.76) == 8.3


def test_board_ranks_overs_with_kickoff_and_rejects_player_props(conn) -> None:
    indiana = seed_canonical_team(conn, 84, "Indiana", "IU")
    wku = seed_canonical_team(conn, 98, "Western Kentucky", "WKU")
    smu = seed_canonical_team(conn, 2567, "SMU", "SMU")
    louisville = seed_canonical_team(conn, 97, "Louisville", "LOU")
    _seed_conference(conn, indiana, "Big Ten")
    _seed_conference(conn, wku, "Conference USA")
    _seed_conference(conn, smu, "ACC")
    _seed_conference(conn, louisville, "ACC")
    _seed_poll(conn, indiana, 4)
    _seed_poll(conn, smu, 16)
    seed_canonical_game(
        conn,
        "cfbd:iu",
        indiana,
        wku,
        football_date="2026-09-19",
        kickoff="2026-09-19T20:00:00+00:00",
        week=3,
    )
    seed_canonical_game(
        conn,
        "cfbd:smu",
        louisville,
        smu,
        football_date="2026-09-19",
        kickoff="2026-09-19T19:30:00+00:00",
        week=3,
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'SPREAD', ?, 'HOME', '2026-09-19T12:00:00+00:00', 8, -110,
                   -110, 'DK', 0.04, 'all_books', 0.55, 0.55, 0.55, 0.01, '[]')""",
        ("cfbd:iu", -44.5),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'SPREAD', ?, 'HOME', '2026-09-19T12:00:00+00:00', 8, -110,
                   -110, 'DK', 0.04, 'all_books', 0.52, 0.52, 0.52, 0.01, '[]')""",
        ("cfbd:smu", -1.5),
    )
    conn.commit()

    rush_inputs = default_team_props_inputs(
        expected_pass_attempts=18.0,
        expected_rushing_attempts=48.0,
        yards_before_contact=3.2,
        yards_after_contact=2.4,
        data_quality_score=100.0,
    )
    rec_inputs = default_team_props_inputs(
        expected_pass_attempts=42.0,
        expected_rushing_attempts=22.0,
        completion_probability=0.78,
        yards_per_completion=13.5,
        data_quality_score=100.0,
    )
    inputs = {
        indiana: rush_inputs,
        wku: replace(rush_inputs, expected_rushing_attempts=28.0, expected_pass_attempts=32.0),
        smu: rec_inputs,
        louisville: replace(rec_inputs, expected_pass_attempts=36.0, yards_per_completion=12.0),
    }

    board = build_over_confidence_board(
        conn,
        "2026-09-19",
        inputs_by_team=inputs,
        min_prob=0.50,
    )
    assert board
    assert board == sorted(
        board, key=lambda row: (-row.over_prob, row.game_label, row.market, row.pick)
    )
    assert all(row.side == "OVER" for row in board)
    assert {row.market for row in board} <= {
        "team_rushing_yards",
        "team_receiving_yards",
        "game_rushing_yards",
        "game_receiving_yards",
    }
    assert all(":" in row.kickoff_et and "m." in row.kickoff_et for row in board)
    indiana_rush = next(
        row for row in board if row.market == "team_rushing_yards" and row.team_id == indiana
    )
    assert indiana_rush.kickoff_et == "4:00 p.m."
    assert "Indiana" in indiana_rush.game_label
    assert indiana_rush.over_prob >= 0.50
    assert "player" not in indiana_rush.market


def test_over_board_cli_prints_kickoff(capsys) -> None:
    from cfb_analytics import cli, db, paths

    with db.open_db() as conn:
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
        conn.commit()

    assert paths.database_path().exists()
    assert cli.main(["over-board", "--date", "2026-09-19"]) == 0
    out = capsys.readouterr().out
    assert "OVER CONFIDENCE BOARD" in out
    assert "4:00 p.m." in out
    assert "UNPROMOTED" in out


def test_board_empty_when_no_ranked_games(conn) -> None:
    tulsa = seed_canonical_team(conn, 202, "Tulsa", "TLSA")
    rice = seed_canonical_team(conn, 242, "Rice", "RICE")
    seed_canonical_game(
        conn,
        "cfbd:none",
        tulsa,
        rice,
        football_date="2026-09-19",
        kickoff="2026-09-19T17:00:00+00:00",
        week=3,
    )
    conn.commit()
    assert build_over_confidence_board(conn, "2026-09-19") == []


def test_settle_over_board_requires_snapshot(conn) -> None:
    report = settle_over_board(conn, "2026-09-19")
    assert report.source == "missing"
    assert report.evaluated == 0
    assert "OVER BOARD SETTLEMENT" in report.as_text()


def test_missing_team_inputs_do_not_reuse_another_offense(conn) -> None:
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
    conn.commit()
    with pytest.raises(SchemaError, match="No team-prop inputs"):
        build_over_confidence_board(
            conn,
            "2026-09-19",
            inputs_by_team={indiana: default_team_props_inputs(data_quality_score=100.0)},
        )
