"""OVER-board snapshots freeze before kickoff.

Reproduced on fa60cfd before the fix: ``persist_over_board`` replaced a
pre-kickoff snapshot (line 170.5) with a later pre-kickoff write (line 99.5),
and a first write stamped exactly at kickoff was stored. ``run_daily`` did not
call ``persist_over_board`` at all.
"""

from __future__ import annotations

from datetime import datetime

from cfb_analytics.daily import run_daily
from cfb_analytics.features.over_confidence import (
    OverConfidencePick,
    load_published_board,
    persist_over_board,
)
from cfb_analytics.utils import FOOTBALL_TZ
from tests.conftest import seed_canonical_game, seed_canonical_team


def _pick(
    game_id: str, kickoff: str, line: float, *, market: str = "team_rushing_yards"
) -> OverConfidencePick:
    return OverConfidencePick(
        rank=1,
        game_id=game_id,
        game_label="WKU at IU",
        kickoff_et="4:00 p.m.",
        kickoff_utc=kickoff,
        market=market,
        pick="IU rush" if market == "team_rushing_yards" else "IU rec",
        team_id="cfbd:84",
        side="OVER",
        line=line,
        projected=242.0,
        over_prob=0.84,
        confidence=9.2,
        tier="HIGH",
        family="favorite_team_rush",
        inclusion_reasons=("ap_top_25",),
        flags=(),
    )


def _seed(conn) -> str:
    home = seed_canonical_team(conn, 84, "Indiana", "IU")
    away = seed_canonical_team(conn, 98, "Western Kentucky", "WKU")
    seed_canonical_game(
        conn,
        "cfbd:iu",
        home,
        away,
        football_date="2026-09-19",
        kickoff="2026-09-19T20:00:00+00:00",
        week=3,
    )
    seed_canonical_game(
        conn,
        "cfbd:other",
        home,
        away,
        football_date="2026-09-19",
        kickoff="2026-09-19T23:00:00+00:00",
        week=3,
    )
    return home


def test_pre_kickoff_rerun_does_not_overwrite(conn) -> None:
    _seed(conn)
    kickoff = "2026-09-19T20:00:00+00:00"
    first = persist_over_board(
        conn,
        "2026-09-19",
        [_pick("cfbd:iu", kickoff, 170.5)],
        as_of_utc="2026-09-19T12:00:00+00:00",
    )
    assert first == 1
    rejected: list[str] = []
    second = persist_over_board(
        conn,
        "2026-09-19",
        [
            _pick("cfbd:iu", kickoff, 99.5),
            _pick("cfbd:iu", kickoff, 210.5, market="team_receiving_yards"),
        ],
        as_of_utc="2026-09-19T15:00:00+00:00",
        rejected=rejected,
    )
    assert second == 0
    assert rejected == []
    board = load_published_board(conn, "2026-09-19")
    assert len(board) == 1
    assert board[0].line == 170.5
    assert board[0].market == "team_rushing_yards"


def test_later_pre_kickoff_game_can_still_be_added(conn) -> None:
    """A freeze is per game. A game with no snapshot can still be published."""
    _seed(conn)
    persist_over_board(
        conn,
        "2026-09-19",
        [_pick("cfbd:iu", "2026-09-19T20:00:00+00:00", 170.5)],
        as_of_utc="2026-09-19T12:00:00+00:00",
    )
    persist_over_board(
        conn,
        "2026-09-19",
        [
            _pick("cfbd:iu", "2026-09-19T20:00:00+00:00", 99.5),
            _pick("cfbd:other", "2026-09-19T23:00:00+00:00", 140.5),
        ],
        as_of_utc="2026-09-19T16:00:00+00:00",
    )
    board = {row.game_id: row for row in load_published_board(conn, "2026-09-19")}
    assert board["cfbd:iu"].line == 170.5
    assert board["cfbd:other"].line == 140.5


def test_post_kickoff_write_is_rejected(conn) -> None:
    _seed(conn)
    kickoff = "2026-09-19T20:00:00+00:00"
    rejected: list[str] = []
    at_kickoff = persist_over_board(
        conn,
        "2026-09-19",
        [_pick("cfbd:iu", kickoff, 170.5)],
        as_of_utc=kickoff,
        rejected=rejected,
    )
    assert at_kickoff == 0
    assert rejected == ["cfbd:iu"]
    assert load_published_board(conn, "2026-09-19") == []

    rejected_after: list[str] = []
    after = persist_over_board(
        conn,
        "2026-09-19",
        [_pick("cfbd:iu", kickoff, 170.5)],
        as_of_utc="2026-09-19T21:00:00+00:00",
        rejected=rejected_after,
    )
    assert after == 0
    assert rejected_after == ["cfbd:iu"]
    assert load_published_board(conn, "2026-09-19") == []


def test_post_kickoff_rerun_leaves_the_pregame_snapshot(conn) -> None:
    _seed(conn)
    kickoff = "2026-09-19T20:00:00+00:00"
    persist_over_board(
        conn,
        "2026-09-19",
        [_pick("cfbd:iu", kickoff, 170.5)],
        as_of_utc="2026-09-19T12:00:00+00:00",
    )
    rejected: list[str] = []
    persist_over_board(
        conn,
        "2026-09-19",
        [],
        as_of_utc="2026-09-20T04:00:00+00:00",
        rejected=rejected,
    )
    persist_over_board(
        conn,
        "2026-09-19",
        [_pick("cfbd:iu", kickoff, 1.5)],
        as_of_utc="2026-09-20T04:00:00+00:00",
        rejected=rejected,
    )
    board = load_published_board(conn, "2026-09-19")
    assert len(board) == 1
    assert board[0].line == 170.5


def test_run_daily_freeze_is_idempotent_and_rejects_post_kickoff(conn, monkeypatch) -> None:
    monkeypatch.setattr("cfb_analytics.config.has_cfbd_key", lambda: False)
    home = _seed(conn)
    assert home == "cfbd:84"
    kickoff = "2026-09-19T20:00:00+00:00"
    calls = {"n": 0}

    def fake_build(_conn, _slate, **_kwargs):
        calls["n"] += 1
        line = 170.5 if calls["n"] == 1 else 99.5
        return [_pick("cfbd:iu", kickoff, line)]

    monkeypatch.setattr(
        "cfb_analytics.features.over_confidence.build_over_confidence_board", fake_build
    )
    before = datetime(2026, 9, 19, 12, tzinfo=FOOTBALL_TZ)
    first = run_daily(
        conn,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=before,
    )
    outcome = next(item for item in first.outcomes if item.name == "over_board")
    assert outcome.status == "ok"
    assert "new_rows=1" in outcome.detail
    assert load_published_board(conn, "2026-09-19")[0].line == 170.5

    second = run_daily(
        conn,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=datetime(2026, 9, 19, 15, tzinfo=FOOTBALL_TZ),
    )
    again = next(item for item in second.outcomes if item.name == "over_board")
    assert "new_rows=0" in again.detail
    assert calls["n"] == 2
    assert load_published_board(conn, "2026-09-19")[0].line == 170.5

    conn.execute("DELETE FROM over_board_snapshots")
    after = run_daily(
        conn,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=datetime(2026, 9, 19, 18, tzinfo=FOOTBALL_TZ),
    )
    rejected = next(item for item in after.outcomes if item.name == "over_board")
    assert "post-kickoff rejected=cfbd:iu" in rejected.detail
    assert load_published_board(conn, "2026-09-19") == []
