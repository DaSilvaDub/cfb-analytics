"""Tests for the CLI entry points, focused on 'futures' and 'live' subcommands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cfb_analytics import cli
from cfb_analytics.models.futures import (
    AdjustedPowerRating,
    GameWinProjection,
    SeasonFuturesProjection,
    WinTotalLineEvaluation,
)


def _make_dummy_futures_projection(
    team_id: str = "cfbd:100",
    season: int = 2026,
    posted_lines: list[float] | None = None,
) -> SeasonFuturesProjection:
    rating = AdjustedPowerRating(
        team_id=team_id,
        base_talent_rating=5.0,
        portal_adjustment=0.5,
        nil_adjustment=0.0,
        returning_production_adjustment=0.2,
        qb_adjustment=1.0,
        adjusted_power_rating=6.7,
        elo_equivalent=1667.5,
        true_talent_composite=750.0,
    )
    schedule = [
        GameWinProjection(
            opponent_id=f"cfbd:{200 + i}",
            projected_spread=-7.0 if i % 2 == 0 else -3.5,
            win_probability=0.72 if i % 2 == 0 else 0.61,
            is_home=i % 2 == 0,
            is_neutral=False,
            is_conference=True,
            game_week=i + 1,
        )
        for i in range(12)
    ]
    evals: dict[float, WinTotalLineEvaluation] = {}
    lines = posted_lines if posted_lines is not None else [8.5]
    for line in lines:
        evals[line] = WinTotalLineEvaluation(
            line=line,
            prob_over=0.642,
            prob_under=0.358,
            prob_push=0.0,
            fair_over_american=-179,
            fair_under_american=179,
            recommended_side="OVER",
            edge=0.142,
            expected_value=0.21,
            model_status="uncalibrated_shadow",
            is_actionable=False,
        )

    return SeasonFuturesProjection(
        team_id=team_id,
        conference="Big Ten",
        adjusted_rating=rating,
        schedule_projections=schedule,
        expected_wins=9.32,
        win_variance=2.15,
        win_stdev=1.47,
        win_distribution=[0.05, 0.15, 0.30, 0.35, 0.15],
        expected_conference_wins=6.5,
        conf_win_distribution=[0.1, 0.2, 0.4, 0.3],
        prob_reach_conference_championship=0.325,
        prob_win_conference_championship=0.184,
        prob_cfp_appearance=0.412,
        win_total_evaluations=evals,
    )


class TestCliFuturesSubcommand:
    def test_futures_summary_human_readable(self, capsys, monkeypatch):
        dummy = _make_dummy_futures_projection("cfbd:100", 2026, [8.5])
        monkeypatch.setattr(
            "cfb_analytics.features.futures.project_team_futures_from_db",
            lambda *a, **kw: dummy,
        )
        monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())

        rc = cli.main(
            [
                "futures",
                "--team",
                "cfbd:100",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00+00:00",
                "--lines",
                "8.5",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "SEASON FUTURES PROJECTION - cfbd:100 (2026)" in out
        assert "True Talent Composite : 750.0" in out
        assert "Adjusted Power Rating : +6.70" in out
        assert "Expected Wins         : 9.32" in out
        assert "Conference (Big Ten)" in out
        assert "CCG Reach: 32.5%" in out
        assert "CCG Win: 18.4%" in out
        assert "CFP Appearance (12-tm): 41.2%" in out
        assert "Win Total Line Evaluations:" in out
        assert "8.5" in out
        assert "64.2%" in out
        assert "OVER" in out

    def test_futures_json_output(self, capsys, monkeypatch):
        dummy = _make_dummy_futures_projection("cfbd:100", 2026, [8.5])
        monkeypatch.setattr(
            "cfb_analytics.features.futures.project_team_futures_from_db",
            lambda *a, **kw: dummy,
        )
        monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())

        rc = cli.main(
            [
                "futures",
                "--team",
                "cfbd:100",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
                "--lines",
                "8.5",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == 1
        assert payload["model"] == "season_futures_v1"
        assert payload["model_status"] == "uncalibrated_shadow"
        assert payload["is_actionable"] is False
        assert payload["season"] == 2026
        assert payload["projection"]["expected_wins"] == 9.32
        assert payload["projection"]["prob_cfp_appearance"] == 0.412
        assert "8.5" in payload["projection"]["win_total_evaluations"]

    def test_futures_with_mc_sims(self, capsys, monkeypatch):
        dummy = _make_dummy_futures_projection("cfbd:100", 2026, [8.5])
        monkeypatch.setattr(
            "cfb_analytics.features.futures.project_team_futures_from_db",
            lambda *a, **kw: dummy,
        )
        monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())

        rc = cli.main(
            [
                "futures",
                "--team-id",
                "cfbd:100",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
                "--lines",
                "8.5",
                "--mc-sims",
                "200",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Monte Carlo Simulation (200 trials):" in out
        assert "Mean Wins" in out
        assert "Median Wins" in out

    def test_futures_with_mc_sims_json(self, capsys, monkeypatch):
        dummy = _make_dummy_futures_projection("cfbd:100", 2026, [8.5])
        monkeypatch.setattr(
            "cfb_analytics.features.futures.project_team_futures_from_db",
            lambda *a, **kw: dummy,
        )
        monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())

        rc = cli.main(
            [
                "futures",
                "--team",
                "cfbd:100",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
                "--lines",
                "8.5",
                "--mc-sims",
                "100",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "simulation" in payload
        assert payload["simulation"]["n_simulations"] == 100
        assert "mean_wins" in payload["simulation"]

    def test_futures_errors_missing_required_args(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["futures"])

        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["futures", "--season", "2026", "--as-of", "2026-08-15T00:00:00Z"]
            )

        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["futures", "--team", "cfbd:1", "--as-of", "2026-08-15T00:00:00Z"]
            )

    def test_futures_invalid_as_of_timestamp(self, capsys):
        rc = cli.main(
            ["futures", "--team", "cfbd:1", "--season", "2026", "--as-of", "invalid-date"]
        )
        assert rc == 2
        assert "must be a parseable ISO timestamp" in capsys.readouterr().err

    def test_futures_invalid_mc_sims(self, capsys, monkeypatch):
        dummy = _make_dummy_futures_projection("cfbd:100", 2026, [8.5])
        monkeypatch.setattr(
            "cfb_analytics.features.futures.project_team_futures_from_db",
            lambda *a, **kw: dummy,
        )
        monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())
        rc = cli.main(
            [
                "futures",
                "--team",
                "cfbd:100",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
                "--mc-sims",
                "0",
            ]
        )
        assert rc == 2
        assert "--mc-sims must be a positive integer" in capsys.readouterr().err

    def test_futures_space_separated_and_comma_separated_lines(self, capsys, monkeypatch):
        captured_lines = []

        def fake_project(*args, **kwargs):
            captured_lines.extend(kwargs.get("posted_lines", []))
            return _make_dummy_futures_projection("cfbd:100", 2026, kwargs.get("posted_lines"))

        monkeypatch.setattr(
            "cfb_analytics.features.futures.project_team_futures_from_db",
            fake_project,
        )
        monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())

        rc = cli.main(
            [
                "futures",
                "--team",
                "cfbd:100",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
                "--lines",
                "7.5",
                "8.5",
                "--lines",
                "9.5,10.5",
            ]
        )
        assert rc == 0
        assert captured_lines == [7.5, 8.5, 9.5, 10.5]

    def test_futures_negative_line_rejected(self, capsys):
        rc = cli.main(
            [
                "futures",
                "--team",
                "cfbd:100",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
                "--lines",
                "-2.5",
            ]
        )
        assert rc == 2
        assert "cannot be negative" in capsys.readouterr().err


class TestCliLiveSubcommand:
    def test_live_manual_simulation_summary(self, capsys):
        rc = cli.main(
            [
                "live",
                "--home",
                "cfbd:100",
                "--away",
                "cfbd:200",
                "--quarter",
                "2",
                "--clock",
                "10:30",
                "--down",
                "2",
                "--distance",
                "6",
                "--yardline",
                "45",
                "--home-score",
                "14",
                "--away-score",
                "7",
                "--possession",
                "cfbd:100",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "LIVE GAME STATE & MICRO-MARKETS" in out
        assert "cfbd:100 vs cfbd:200" in out
        assert "Q2 10:30 | Down 2 & 6 at yardline 45 (cfbd:100 ball)" in out
        assert "Score      : cfbd:100 14 - 7 cfbd:200" in out
        assert "Win Probability:" in out
        assert "cfbd:100" in out
        assert "cfbd:200" in out
        assert "Next Drive Outcome Distribution (cfbd:100 at yardline 45):" in out
        assert "Touchdown (TD)" in out
        assert "Field Goal (FG)" in out
        assert "Punt" in out
        assert "Live Projected Totals:" in out
        assert "Projected Game Total" in out

    def test_live_manual_simulation_json(self, capsys):
        rc = cli.main(
            [
                "live",
                "--home",
                "cfbd:100",
                "--away",
                "cfbd:200",
                "--quarter",
                "3",
                "--clock",
                "05:00",
                "--down",
                "1",
                "--distance",
                "10",
                "--yardline",
                "75",
                "--home-score",
                "21",
                "--away-score",
                "17",
                "--possession",
                "cfbd:200",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == 1
        assert payload["model"] == "live_micro_markets_v1"
        assert payload["model_status"] == "uncalibrated_shadow"
        assert payload["is_actionable"] is False

        state = payload["state"]
        assert state["home_team_id"] == "cfbd:100"
        assert state["away_team_id"] == "cfbd:200"
        assert state["possession_team_id"] == "cfbd:200"
        assert state["quarter"] == 3
        assert state["clock_seconds"] == 300
        assert state["clock_display"] == "5:00"
        assert state["home_score"] == 21
        assert state["away_score"] == 17

        win_prob = payload["win_probability"]
        assert 0.0 < win_prob["home_win_prob"] < 1.0
        assert 0.0 < win_prob["away_win_prob"] < 1.0
        assert round(win_prob["home_win_prob"] + win_prob["away_win_prob"], 2) == 1.0

        drive = payload["next_drive_outcome"]
        assert drive["possession_team_id"] == "cfbd:200"
        assert drive["start_yardline"] == 75
        assert drive["touchdown"] > 0
        assert drive["punt"] > 0

        totals = payload["projected_totals"]
        assert totals["home"]["current_score"] == 21
        assert totals["away"]["current_score"] == 17
        assert totals["projected_game_total"] > 38.0

    def test_live_from_db_game(self, capsys):
        cli.main(["init-db"])
        capsys.readouterr()

        from cfb_analytics import db
        from cfb_analytics.ingest import store

        with db.open_db() as conn:
            store.upsert_team(
                conn, {"team_id": "home_t", "school": "HT", "alias": "HT", "market": "HT"}
            )
            store.upsert_team(
                conn, {"team_id": "away_t", "school": "AT", "alias": "AT", "market": "AT"}
            )
            store.upsert_game(
                conn,
                {
                    "game_id": "test_live_game_1",
                    "season": 2026,
                    "kickoff_utc": "2026-09-12T16:00:00+00:00",
                    "football_date": "2026-09-12",
                    "home_team_id": "home_t",
                    "away_team_id": "away_t",
                    "home_points": 10,
                    "away_points": 7,
                    "completed": 0,
                    "status": "in_progress",
                },
            )
            conn.execute(
                "UPDATE games SET home_points = 10, away_points = 7 "
                "WHERE game_id = 'test_live_game_1'"
            )

        rc = cli.main(
            [
                "live",
                "--game",
                "test_live_game_1",
                "--possession",
                "home_t",
                "--quarter",
                "2",
                "--clock",
                "07:30",
                "--down",
                "1",
                "--distance",
                "10",
                "--yardline",
                "50",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "home_t vs away_t" in out
        assert "Score      : home_t 10 - 7 away_t" in out

    def test_live_clock_seconds_input(self, capsys):
        rc = cli.main(
            [
                "live",
                "--home",
                "cfbd:1",
                "--away",
                "cfbd:2",
                "--clock",
                "450",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Q1 7:30" in out

    def test_live_errors_missing_game_and_teams(self, capsys):
        rc = cli.main(["live"])
        assert rc == 2
        assert (
            "Either --game or both --home and --away must be specified" in capsys.readouterr().err
        )

    def test_live_errors_same_home_and_away(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:1", "--away", "cfbd:1"])
        assert rc == 2
        assert "Home and away team IDs must differ" in capsys.readouterr().err

    def test_live_errors_invalid_possession(self, capsys):
        rc = cli.main(
            ["live", "--home", "cfbd:1", "--away", "cfbd:2", "--possession", "cfbd:other"]
        )
        assert rc == 2
        assert "must be either home" in capsys.readouterr().err

    def test_live_errors_negative_score(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:1", "--away", "cfbd:2", "--home-score", "-5"])
        assert rc == 2
        assert "Scores cannot be negative" in capsys.readouterr().err

    def test_live_errors_invalid_clock(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:1", "--away", "cfbd:2", "--clock", "20:00"])
        assert rc == 2
        assert "Clock must be between 0 and 900 seconds" in capsys.readouterr().err

    def test_live_errors_unparseable_clock(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:1", "--away", "cfbd:2", "--clock", "invalid"])
        assert rc == 2
        assert "--clock format must be MM:SS or integer seconds" in capsys.readouterr().err

    def test_live_errors_invalid_down(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:1", "--away", "cfbd:2", "--down", "5"])
        assert rc == 2
        assert "Down must be in [1, 4]" in capsys.readouterr().err

    def test_live_errors_invalid_yardline(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:1", "--away", "cfbd:2", "--yardline", "120"])
        assert rc == 2
        assert "Yardline must be in [1, 99]" in capsys.readouterr().err

    def test_live_errors_game_not_found(self, capsys):
        cli.main(["init-db"])
        capsys.readouterr()
        rc = cli.main(["live", "--game", "nonexistent_game_id"])
        assert rc == 2
        assert "not found in database" in capsys.readouterr().err

    def test_live_regulation_expired_auto_final(self, capsys):
        rc = cli.main(
            [
                "live",
                "--home",
                "cfbd:100",
                "--away",
                "cfbd:200",
                "--quarter",
                "4",
                "--clock",
                "00:00",
                "--home-score",
                "24",
                "--away-score",
                "17",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "[FINAL]" in out
        assert "100.0% (N/A)" in out
        assert "0.0% (N/A)" in out

    def test_live_completed_db_game(self, capsys):
        cli.main(["init-db"])
        capsys.readouterr()

        from cfb_analytics import db
        from cfb_analytics.ingest import store

        with db.open_db() as conn:
            store.upsert_team(
                conn, {"team_id": "ht_fin", "school": "HTF", "alias": "HTF", "market": "HTF"}
            )
            store.upsert_team(
                conn, {"team_id": "at_fin", "school": "ATF", "alias": "ATF", "market": "ATF"}
            )
            store.upsert_game(
                conn,
                {
                    "game_id": "test_completed_game_1",
                    "season": 2026,
                    "kickoff_utc": "2026-09-12T16:00:00+00:00",
                    "football_date": "2026-09-12",
                    "home_team_id": "ht_fin",
                    "away_team_id": "at_fin",
                    "status": "completed",
                },
            )
            conn.execute(
                "UPDATE games SET home_points = 35, away_points = 21, "
                "completed = 1, status = 'completed' "
                "WHERE game_id = 'test_completed_game_1'"
            )

        rc = cli.main(["live", "--game", "test_completed_game_1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ht_fin vs at_fin" in out
        assert "[FINAL]" in out
        assert "Score      : ht_fin 35 - 21 at_fin" in out
        assert "100.0% (N/A)" in out

    def test_live_final_flag(self, capsys):
        rc = cli.main(
            [
                "live",
                "--home",
                "cfbd:100",
                "--away",
                "cfbd:200",
                "--quarter",
                "5",
                "--clock",
                "00:00",
                "--home-score",
                "31",
                "--away-score",
                "28",
                "--final",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "[FINAL]" in out
        assert "100.0% (N/A)" in out

    def test_live_tied_final_rejected(self, capsys):
        rc = cli.main(
            [
                "live",
                "--home",
                "cfbd:100",
                "--away",
                "cfbd:200",
                "--quarter",
                "4",
                "--clock",
                "00:00",
                "--home-score",
                "20",
                "--away-score",
                "20",
                "--final",
            ]
        )
        assert rc == 2
        assert "cannot have a tied score" in capsys.readouterr().err

    def test_live_errors_invalid_clock_seconds_range(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:1", "--away", "cfbd:2", "--clock", "12:65"])
        assert rc == 2
        assert "MM:SS with 0-59 seconds" in capsys.readouterr().err

    def test_live_errors_same_teams_case_insensitive(self, capsys):
        rc = cli.main(["live", "--home", "cfbd:abc", "--away", "CFBD:ABC"])
        assert rc == 2
        assert "Home and away team IDs must differ" in capsys.readouterr().err


@pytest.mark.parametrize("line", ["nan", "inf", ",", " "])
def test_futures_invalid_lines_before_database(line, capsys, monkeypatch):
    def no_db():
        pytest.fail("Invalid input must be rejected before opening the database")

    monkeypatch.setattr("cfb_analytics.db.open_db", no_db)
    assert (
        cli.main(
            [
                "futures",
                "--team",
                "cfbd:1",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
                "--lines",
                line,
            ]
        )
        == 2
    )
    assert capsys.readouterr().err


def test_futures_does_not_invent_missing_inputs(monkeypatch, capsys):
    from cfb_analytics.errors import SchemaError

    def project(*args, **kwargs):
        assert kwargs["qb_tier"] is None
        assert kwargs["qb_continuity"] is None
        assert kwargs["portal_composite"] is None
        assert kwargs["nil_tier"] is None
        assert kwargs["nil_budget_millions"] is None
        raise SchemaError("Missing caller inputs")

    monkeypatch.setattr("cfb_analytics.features.futures.project_team_futures_from_db", project)
    monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())
    assert (
        cli.main(
            [
                "futures",
                "--team",
                "cfbd:1",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
            ]
        )
        == 2
    )
    assert "Missing caller inputs" in capsys.readouterr().err


def test_live_unresolved_overtime_is_not_final(capsys):
    assert (
        cli.main(
            [
                "live",
                "--home",
                "H",
                "--away",
                "A",
                "--quarter",
                "5",
                "--clock",
                "0",
                "--home-score",
                "31",
                "--away-score",
                "28",
                "--json",
            ]
        )
        == 2
    )
    assert "Unresolved overtime" in capsys.readouterr().err


def test_live_final_has_no_next_drive(capsys):
    assert (
        cli.main(
            [
                "live",
                "--home",
                "H",
                "--away",
                "A",
                "--quarter",
                "4",
                "--clock",
                "0",
                "--home-score",
                "31",
                "--away-score",
                "28",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["next_drive_outcome"] is None
    assert payload["projected_totals"]["projected_game_total"] == 59


def test_live_shadow_warning_survives_global_promotion(monkeypatch, capsys):
    monkeypatch.setattr("cfb_analytics.config.is_shadow_mode", lambda: False)
    assert cli.main(["live", "--home", "H", "--away", "A"]) == 0
    assert "uncalibrated shadow output" in capsys.readouterr().out


def test_futures_shadow_warning_survives_global_promotion(monkeypatch, capsys):
    monkeypatch.setattr("cfb_analytics.config.is_shadow_mode", lambda: False)
    monkeypatch.setattr("cfb_analytics.db.open_db", MagicMock())
    monkeypatch.setattr(
        "cfb_analytics.features.futures.project_team_futures_from_db",
        lambda *a, **kw: _make_dummy_futures_projection(),
    )
    assert (
        cli.main(
            [
                "futures",
                "--team",
                "cfbd:1",
                "--season",
                "2026",
                "--as-of",
                "2026-08-15T00:00:00Z",
            ]
        )
        == 0
    )
    assert "uncalibrated shadow output" in capsys.readouterr().out


@pytest.mark.parametrize(
    "case", ["missing_situation", "identity_override", "missing_final_score", "missing_live_scores"]
)
def test_live_database_rejects_incomplete_or_mismatched_state(case, capsys):
    from cfb_analytics import db
    from cfb_analytics.ingest import store

    cli.main(["init-db"])
    capsys.readouterr()
    with db.open_db() as conn:
        for team in ("H", "A"):
            store.upsert_team(conn, {"team_id": team, "school": team})
        store.upsert_game(
            conn,
            {
                "game_id": "g",
                "season": 2026,
                "kickoff_utc": "2026-09-12T16:00:00+00:00",
                "football_date": "2026-09-12",
                "home_team_id": "H",
                "away_team_id": "A",
                "status": "in_progress",
            },
        )
        if case == "missing_final_score":
            conn.execute("UPDATE games SET completed = 1, home_points = 7 WHERE game_id = 'g'")
    extra = ["--home", "other"] if case == "identity_override" else []
    if case == "missing_live_scores":
        extra = [
            "--quarter",
            "3",
            "--clock",
            "450",
            "--down",
            "1",
            "--distance",
            "10",
            "--yardline",
            "50",
            "--possession",
            "H",
        ]
    assert cli.main(["live", "--game", "g", *extra]) == 2
    errors = {
        "missing_live_scores": "Database scores are missing",
        "missing_situation": "Database has no live situation",
        "identity_override": "cannot override a stored game identity",
        "missing_final_score": "missing final scores",
    }
    assert errors[case] in capsys.readouterr().err
