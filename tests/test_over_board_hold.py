"""Unpriced OVER rows stay on the snapshot and are not playable.

CFBD spreads have no American price, so market_consensus has no SPREAD row
and the sit-QB gates never see a number. An inferred mark is not a two-sided
book line. Those rows are stored with publish_status held.
"""

from __future__ import annotations

from datetime import datetime

from cfb_analytics.daily import run_daily
from cfb_analytics.features.over_confidence import (
    HOLD_INFERRED,
    HOLD_NO_SPREAD,
    PUBLISH_HELD,
    PUBLISH_PLAYABLE,
    OverConfidencePick,
    build_over_confidence_board,
    load_published_board,
    persist_over_board,
)
from cfb_analytics.models.team_props import default_team_props_inputs
from cfb_analytics.reporting.slate import build_slate_report, render_slate_report, slate_report_json
from cfb_analytics.utils import FOOTBALL_TZ
from tests.conftest import seed_canonical_game, seed_canonical_team
from tests.test_over_confidence import _insert_spread
from tests.test_ranked_slate import _seed_conference, _seed_poll

KICKOFF = "2026-09-19T20:00:00+00:00"
AS_OF = "2026-09-19T12:00:00+00:00"
POSTED_LINE = 155.5


def _ranked_game(conn, game_id: str = "cfbd:iu") -> tuple[str, str]:
    home = seed_canonical_team(conn, 84, "Indiana", "IU")
    away = seed_canonical_team(conn, 98, "Western Kentucky", "WKU")
    _seed_conference(conn, home, "Big Ten")
    _seed_poll(conn, home, 4)
    seed_canonical_game(
        conn,
        game_id,
        home,
        away,
        football_date="2026-09-19",
        kickoff=KICKOFF,
        week=3,
    )
    return home, away


def _inputs():
    return default_team_props_inputs(
        expected_pass_attempts=18.0,
        expected_rushing_attempts=48.0,
        yards_before_contact=3.2,
        yards_after_contact=2.4,
        data_quality_score=100.0,
    )


def _board(conn, home: str, away: str):
    inputs = _inputs()
    return build_over_confidence_board(
        conn,
        "2026-09-19",
        inputs_by_team={home: inputs, away: inputs},
        min_prob=0.50,
        as_of_utc=AS_OF,
    )


def _two_sided_rush(conn, game_id: str, team_id: str) -> None:
    for side in ("OVER", "UNDER"):
        conn.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, best_price, best_book, hold,
                prob_multiplicative, prob_shin, prob_power, flags)
               VALUES (?, ?, 'team_rushing_yards', ?, ?, ?, 4, -110, -105, 'DK', 0.045,
                       0.52, 0.52, 0.52, '[]')""",
            (game_id, team_id, POSTED_LINE, side, AS_OF),
        )


def test_missing_spread_consensus_holds_every_row(conn) -> None:
    home, away = _ranked_game(conn)
    _two_sided_rush(conn, "cfbd:iu", home)
    conn.commit()
    board = _board(conn, home, away)
    assert board
    assert all(row.publish_status == PUBLISH_HELD for row in board)
    assert all(row.hold_reason == HOLD_NO_SPREAD for row in board)


def test_inferred_mark_is_held_when_the_spread_exists(conn) -> None:
    home, away = _ranked_game(conn)
    _insert_spread(conn, "cfbd:iu", -6.5, as_of=AS_OF)
    conn.commit()
    board = _board(conn, home, away)
    inferred = [row for row in board if "inferred_mark" in row.flags]
    assert inferred
    assert all(row.publish_status == PUBLISH_HELD for row in inferred)
    assert all(row.hold_reason == HOLD_INFERRED for row in inferred)
    rush = next(
        row for row in inferred if row.market == "team_rushing_yards" and row.team_id == home
    )
    assert rush.tier in {"HIGH", "MEDIUM", "LEAN"}


def test_priced_two_sided_row_with_a_spread_stays_playable(conn) -> None:
    home, away = _ranked_game(conn)
    _insert_spread(conn, "cfbd:iu", -6.5, as_of=AS_OF)
    _two_sided_rush(conn, "cfbd:iu", home)
    conn.commit()
    board = _board(conn, home, away)
    rush = next(row for row in board if row.market == "team_rushing_yards" and row.team_id == home)
    assert rush.line == POSTED_LINE
    assert "inferred_mark" not in rush.flags
    assert rush.publish_status == PUBLISH_PLAYABLE
    assert rush.hold_reason is None
    persist_over_board(conn, "2026-09-19", [rush], as_of_utc=AS_OF)
    stored = load_published_board(conn, "2026-09-19")
    assert len(stored) == 1
    assert stored[0].line == POSTED_LINE
    assert stored[0].publish_status == PUBLISH_PLAYABLE
    assert stored[0].hold_reason is None


def test_run_daily_and_slate_report_name_the_hold(conn, monkeypatch) -> None:
    monkeypatch.setattr("cfb_analytics.config.has_cfbd_key", lambda: False)
    home, _away = _ranked_game(conn)
    conn.commit()
    kickoff = KICKOFF

    def fake_build(_conn, _slate, **_kwargs):
        return [
            OverConfidencePick(
                rank=1,
                game_id="cfbd:iu",
                game_label="WKU at IU",
                kickoff_et="4:00 p.m.",
                market="team_rushing_yards",
                pick="IU rush",
                team_id=home,
                side="OVER",
                line=160.5,
                projected=202.8,
                over_prob=0.78,
                confidence=8.5,
                tier="HIGH",
                inclusion_reasons=("ap_top_25",),
                flags=("ap_top_25", "inferred_mark"),
                family="favorite_team_rush",
                kickoff_utc=kickoff,
                publish_status=PUBLISH_HELD,
                hold_reason=HOLD_NO_SPREAD,
            )
        ]

    monkeypatch.setattr(
        "cfb_analytics.features.over_confidence.build_over_confidence_board", fake_build
    )
    report = run_daily(
        conn,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=datetime(2026, 9, 19, 12, tzinfo=FOOTBALL_TZ),
    )
    outcome = next(item for item in report.outcomes if item.name == "over_board")
    assert "held=1" in outcome.detail
    assert "no_spread_consensus=1" in outcome.detail
    assert "held rows are not playable" in outcome.detail
    stored = load_published_board(conn, "2026-09-19")
    assert stored[0].publish_status == PUBLISH_HELD
    assert stored[0].hold_reason == HOLD_NO_SPREAD

    slate = build_slate_report(conn, "2026-09-19")
    text = render_slate_report(slate)
    assert "held=1" in text
    assert "no_spread_consensus=1" in text
    payload = slate_report_json(slate)
    assert "no_spread_consensus=1" in payload["yardage_overs"]["hold"]
    assert payload["yardage_overs"]["picks"][0]["publish_status"] == PUBLISH_HELD
    assert payload["yardage_overs"]["picks"][0]["hold_reason"] == HOLD_NO_SPREAD
