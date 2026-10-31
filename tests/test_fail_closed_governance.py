"""Fail-closed governance and unknown-QB confirmation.

Reproduced on fa60cfd before the fix:
* ``mispriced`` caught an exception from ``GovernanceGate.evaluate_candidate``
  and stored ``GovernanceAction.APPROVE`` (cli.py).
* ``_load_qb_status`` started at ``qb_confirmed = True`` and left that value
  when no starter or no availability row existed (reasoning/context.py).
"""

from __future__ import annotations

import json
import logging

import pytest

from cfb_analytics import cli
from cfb_analytics.governance.gate import GovernanceGate
from cfb_analytics.governance.models import GovernanceAction
from cfb_analytics.governance.rules import CandidateWager
from cfb_analytics.ingest import store
from cfb_analytics.reasoning.context import _load_qb_status, load_situational_context_for_game
from tests.conftest import seed_canonical_game, seed_canonical_team
from tests.test_cli_reporting import _seed_game_and_consensus


@pytest.fixture
def cli_db():
    cli.main(["init-db"])
    from cfb_analytics import db

    with db.open_db() as connection:
        yield connection


class _ExplodingRules:
    def evaluate_rules(self, *_args, **_kwargs):
        raise RuntimeError("rules exploded")


def _total_candidate() -> CandidateWager:
    return CandidateWager(
        candidate_id="cfbd:1:TOTAL:OVER",
        game_id="cfbd:1",
        market_type="TOTAL",
        market="TOTAL",
        side="OVER",
        line=55.0,
        odds_american=-110,
        confidence=8.0,
        edge_pct=0.05,
        fair_prob=0.55,
    )


def test_completed_evaluation_is_not_failed_closed() -> None:
    """The wrapper must not turn a finished evaluation into a veto."""
    verdict = GovernanceGate().evaluate_candidate(_total_candidate())
    assert verdict.action == GovernanceAction.APPROVE
    assert verdict.is_approved is True


def test_exception_yields_no_pass_and_is_not_approved(caplog: pytest.LogCaptureFixture) -> None:
    gate = GovernanceGate(rules_engine=_ExplodingRules())  # type: ignore[arg-type]
    with caplog.at_level(logging.ERROR):
        verdict = gate.evaluate_candidate(_total_candidate(), context=object())  # type: ignore[arg-type]

    assert verdict.action != GovernanceAction.PASS
    assert verdict.action != GovernanceAction.APPROVE
    assert verdict.is_pass is False
    assert verdict.is_approved is False
    assert verdict.action == GovernanceAction.VETO
    assert verdict.parlay_eligible is False
    assert "failed closed" in verdict.notes
    assert "rules exploded" in verdict.notes
    assert any("rules exploded" in rec.message for rec in caplog.records)
    assert any(rec.exc_info is not None for rec in caplog.records)


def test_mispriced_cli_exception_does_not_approve(capsys, cli_db, monkeypatch, caplog) -> None:
    """mispriced emits the gate's fail-closed VETO. The command has no handler of its own."""
    game_id = _seed_game_and_consensus(cli_db, "2026-09-05")
    for side, prob in (("OVER", 0.5), ("UNDER", 0.5)):
        cli_db.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, best_price, best_book, hold,
                prob_multiplicative, prob_shin, prob_power, flags)
               VALUES (?, 'cfbd:100', 'team_rushing_yards', 40.5, ?,
                       '2026-09-05T12:00:00+00:00', 4, -110, -105, 'DK', 0.045,
                       ?, ?, ?, '[]')""",
            (game_id, side, prob, prob, prob),
        )
    cli_db.commit()

    def boom(self, *_args, **_kwargs):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(
        "cfb_analytics.governance.gate.GovernanceGate._evaluate_candidate", boom
    )
    capsys.readouterr()
    with caplog.at_level(logging.ERROR):
        rc = cli.main(["mispriced", "--date", "2026-09-05", "--min-edge", "0.0", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidates"], "the seeded slate must reach the gate"
    for record in payload["candidates"]:
        verdict = record["governance_verdict"]
        assert verdict["action"] != "PASS"
        assert verdict["action"] != "APPROVE"
        assert verdict["action"] == "VETO"
        assert verdict["parlay_eligible"] is False
        assert "failed closed" in verdict["notes"]
    assert any("gate exploded" in rec.message for rec in caplog.records)


def _passing(conn, game_id, team_id, player_id, attempts, week, kickoff) -> None:
    store.upsert_player(conn, player_id, "Starter")
    store.upsert_player_game_passing(
        conn,
        {
            "game_id": game_id,
            "team_id": team_id,
            "player_id": player_id,
            "season": 2026,
            "week": week,
            "completions": 20,
            "attempts": attempts,
            "yards": 250,
            "avg_yards": 12.5,
            "touchdowns": 2,
            "interceptions": 0,
            "qbr": None,
            "source": "cfbd",
        },
    )


def _availability(conn, game_id, team_id, as_of, designation) -> None:
    store.insert_availability(
        conn,
        [
            {
                "game_id": game_id,
                "team_id": team_id,
                "player_id": "qb-news",
                "as_of_utc": as_of,
                "position": "QB",
                "position_group": "QB",
                "designation": designation,
                "injury_type": None,
                "return_date": None,
                "last_updated_utc": as_of,
                "has_news": 1,
                "source": "outlier",
            }
        ],
    )


def _two_games(conn):
    home = seed_canonical_team(conn, 100, "Georgia", "UGA")
    away = seed_canonical_team(conn, 200, "Vanderbilt", "VAN")
    seed_canonical_game(
        conn,
        "cfbd:prior",
        home,
        away,
        football_date="2026-08-29",
        kickoff="2026-08-29T23:30:00+00:00",
        week=1,
    )
    seed_canonical_game(
        conn,
        "cfbd:next",
        home,
        away,
        football_date="2026-09-05",
        kickoff="2026-09-05T19:30:00+00:00",
        week=2,
    )
    return home


def test_missing_starter_or_injury_row_is_not_confirmed(conn) -> None:
    home = _two_games(conn)
    kickoff = "2026-09-05T19:30:00+00:00"

    confirmed, _name, _share = _load_qb_status(conn, home, 2026, kickoff, "cfbd:next")
    assert confirmed is False

    _availability(conn, "cfbd:next", home, "2026-09-05T12:00:00+00:00", "Probable")
    confirmed, _name, _share = _load_qb_status(conn, home, 2026, kickoff, "cfbd:next")
    assert confirmed is False

    conn.execute("DELETE FROM availability")
    _passing(conn, "cfbd:prior", home, "cfbd:qb1", 30, 1, "2026-08-29T23:30:00+00:00")
    confirmed, name, share = _load_qb_status(conn, home, 2026, kickoff, "cfbd:next")
    assert confirmed is False
    assert name == "Starter"
    assert share == pytest.approx(1.0)

    ctx = load_situational_context_for_game(conn, "cfbd:next", "2026-09-05")
    assert ctx.qb_home_confirmed is False
    assert ctx.qb_away_confirmed is False


def test_starter_and_fresh_probable_row_confirm(conn) -> None:
    home = _two_games(conn)
    kickoff = "2026-09-05T19:30:00+00:00"
    _passing(conn, "cfbd:prior", home, "cfbd:qb1", 30, 1, "2026-08-29T23:30:00+00:00")
    _availability(conn, "cfbd:next", home, "2026-09-04T12:00:00+00:00", "Out")
    _availability(conn, "cfbd:next", home, "2026-09-05T12:00:00+00:00", "Probable")

    confirmed, name, _share = _load_qb_status(conn, home, 2026, kickoff, "cfbd:next")
    assert confirmed is True
    assert name == "Starter"

    ctx = load_situational_context_for_game(conn, "cfbd:next", "2026-09-05")
    assert ctx.qb_home_confirmed is True
    assert ctx.qb_away_confirmed is False


def test_out_or_post_kickoff_row_does_not_confirm(conn) -> None:
    home = _two_games(conn)
    kickoff = "2026-09-05T19:30:00+00:00"
    _passing(conn, "cfbd:prior", home, "cfbd:qb1", 30, 1, "2026-08-29T23:30:00+00:00")

    _availability(conn, "cfbd:next", home, "2026-09-05T12:00:00+00:00", "Questionable")
    confirmed, _name, _share = _load_qb_status(conn, home, 2026, kickoff, "cfbd:next")
    assert confirmed is False

    conn.execute("DELETE FROM availability")
    _availability(conn, "cfbd:next", home, "2026-09-05T19:30:00+00:00", "Probable")
    confirmed, _name, _share = _load_qb_status(conn, home, 2026, kickoff, "cfbd:next")
    assert confirmed is False


def test_usage_bar_is_unchanged(conn) -> None:
    """attempt_share >= 0.70 and games >= 1 still required; the bar did not move."""
    home = _two_games(conn)
    kickoff = "2026-09-05T19:30:00+00:00"
    _passing(conn, "cfbd:prior", home, "cfbd:qb1", 10, 1, "2026-08-29T23:30:00+00:00")
    _passing(conn, "cfbd:prior", home, "cfbd:qb2", 10, 1, "2026-08-29T23:30:00+00:00")
    _availability(conn, "cfbd:next", home, "2026-09-05T12:00:00+00:00", "Probable")
    confirmed, _name, share = _load_qb_status(conn, home, 2026, kickoff, "cfbd:next")
    assert share == pytest.approx(0.5)
    assert confirmed is False


def test_starter_lookup_error_stays_unconfirmed(conn, monkeypatch, caplog) -> None:
    home = _two_games(conn)
    _availability(conn, "cfbd:next", home, "2026-09-05T12:00:00+00:00", "Probable")

    def boom(*_args, **_kwargs):
        raise RuntimeError("passing table unreadable")

    monkeypatch.setattr("cfb_analytics.reasoning.context.presumptive_starter_as_of", boom)
    with caplog.at_level(logging.ERROR):
        confirmed, name, _share = _load_qb_status(
            conn, home, 2026, "2026-09-05T19:30:00+00:00", "cfbd:next"
        )
    assert confirmed is False
    assert name is None
    assert any("passing table unreadable" in rec.message for rec in caplog.records)
