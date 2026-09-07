from __future__ import annotations

import json

import pytest

from cfb_analytics import cli, paths
from cfb_analytics.errors import AuthRequiredError, UnknownLeagueError
from cfb_analytics.sources.cfbd import CFBDClient
from cfb_analytics.sources.outlier import LEAGUE_TOKEN, OutlierClient


class StubHttp:
    """Stands in for HttpClient, keyed by URL substring."""

    def __init__(self, responses=None, errors=None):
        self.responses = responses or {}
        self.errors = errors or {}
        self.calls: list[str] = []

    def get_json(self, url):
        self.calls.append(url)
        for fragment, exc in self.errors.items():
            if fragment in url:
                raise exc
        for fragment, payload in self.responses.items():
            if fragment in url:
                return payload
        raise UnknownLeagueError(f"no stub for {url}")

    def get_payload(self, url):
        return self.get_json(url)


class TestOutlierClientRouting:
    def test_schedule_returns_event_dicts(self, schedule_event):
        http = StubHttp({"/schedule": {"events": [schedule_event, "junk"]}})
        assert OutlierClient(http=http).fetch_schedule() == [schedule_event]

    def test_markets_uses_query_parameter_not_path_segment(self, moneyline_market):
        http = StubHttp({"/markets": {"markets": [moneyline_market]}})
        OutlierClient(http=http).fetch_event_markets("evt-1", "GAMELINE")
        assert "markets?marketType=GAMELINE" in http.calls[0]

    def test_injuries_hits_the_league_scoped_team_path(self):
        http = StubHttp({"/injuries": {"players": [{"playerId": "p"}]}})
        rows = OutlierClient(http=http).fetch_team_injuries("t-1")
        assert rows == [{"playerId": "p"}]
        assert f"/leagues/{LEAGUE_TOKEN}/teams/t-1/injuries" in http.calls[0]

    def test_missing_markets_key_yields_empty_list(self):
        http = StubHttp({"/markets": {}})
        assert OutlierClient(http=http).fetch_event_markets("evt-1") == []

    def test_schedule_without_events_raises(self):
        from cfb_analytics.errors import SchemaError

        http = StubHttp({"/schedule": {"nope": []}})
        with pytest.raises(SchemaError, match="no 'events' list"):
            OutlierClient(http=http).fetch_schedule()


class TestUnknownLeagueDisambiguation:
    """Outlier answers an unknown league with 502, same as a real outage."""

    def test_names_the_correct_token_when_the_control_league_resolves(self):
        http = StubHttp(
            responses={"/NFL/schedule": {"events": []}},
            errors={"/BOGUS/schedule": UnknownLeagueError("HTTP 502")},
        )
        with pytest.raises(UnknownLeagueError, match="NCAAFB"):
            OutlierClient(http=http, league="BOGUS").fetch_schedule()

    def test_reports_an_outage_when_the_control_league_also_fails(self):
        http = StubHttp(errors={"schedule": UnknownLeagueError("HTTP 502")})
        with pytest.raises(UnknownLeagueError, match="API outage"):
            OutlierClient(http=http, league="BOGUS").fetch_schedule()

    def test_auth_failure_is_not_mistaken_for_an_unknown_league(self):
        http = StubHttp(errors={"schedule": AuthRequiredError("HTTP 401")})
        with pytest.raises(AuthRequiredError):
            OutlierClient(http=http).fetch_schedule()


class TestCliParser:
    def test_registers_only_implemented_commands(self):
        parser = cli.build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        commands = set(actions[0].choices)
        assert commands == {"init-db", "doctor", "schedule", "status", "ingest",
                            "backfill-cfbd", "coverage", "market", "board", "daily",
                            "backfill-fundamentals", "backfill-roster", "backfill-passing",
                            "backfill-elo", "fit-ratings", "fit-elo", "backtest", "futures"}

    def test_unimplemented_phases_are_absent(self):
        """`--help` must not advertise anything that does not run."""
        parser = cli.build_parser()
        commands = set([a for a in parser._actions if getattr(a, "choices", None)][0].choices)
        assert not commands & {"features", "train", "slate", "parlay", "settle"}

    def test_ingest_requires_a_date(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["ingest"])

    def test_futures_requires_an_explicit_as_of_cutoff(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                [
                    "futures",
                    "--team-id", "cfbd:1",
                    "--season", "2026",
                    "--portal-net-composite", "0",
                    "--nil-budget-millions", "10",
                    "--qb-tier", "unknown",
                    "--qb-continuity", "unknown",
                ]
            )


class TestCliCommands:
    def test_futures_rejects_a_timezone_naive_cutoff(self, capsys):
        result = cli.main(
            [
                "futures",
                "--team-id", "cfbd:1",
                "--season", "2026",
                "--as-of", "2026-08-15T00:00:00",
                "--portal-net-composite", "0",
                "--nil-budget-millions", "10",
                "--qb-tier", "unknown",
                "--qb-continuity", "unknown",
            ]
        )

        assert result == 2
        assert "must include a UTC offset" in capsys.readouterr().err

    def test_futures_emits_versioned_non_actionable_json(self, capsys, monkeypatch):
        from cfb_analytics.models.futures import (
            NILTier,
            RosterTalentInputs,
            ScheduledOpponent,
            project_season_futures,
        )

        captured = {}

        def fake_project(conn, team_id, season, **kwargs):
            captured.update(team_id=team_id, season=season, **kwargs)
            kwargs["input_manifest"]["fixture"] = {"snapshot_id": "fixture:1"}
            inputs = RosterTalentInputs(
                team_id=team_id,
                recruiting_composite=700.0,
                portal_composite=kwargs["portal_composite"],
                nil_tier=NILTier.from_budget(kwargs["nil_budget_millions"]),
                returning_production=0.60,
                qb_tier=kwargs["qb_tier"],
                qb_continuity=kwargs["qb_continuity"],
                conference="Big Ten",
            )
            return project_season_futures(
                inputs,
                [ScheduledOpponent("cfbd:2", 0.0)],
                posted_lines=kwargs["posted_lines"],
            )

        monkeypatch.setattr(
            "cfb_analytics.features.futures.project_team_futures_from_db", fake_project
        )
        cli.main(["init-db"])
        capsys.readouterr()

        result = cli.main(
            [
                "futures",
                "--team-id", "cfbd:1",
                "--season", "2026",
                "--as-of", "2026-08-15T00:00:00+00:00",
                "--portal-net-composite", "2.5",
                "--nil-budget-millions", "15",
                "--qb-tier", "tier_2_quality_starter",
                "--qb-continuity", "returning_starter_same_system",
                "--posted-line", "0.5",
            ]
        )

        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == 1
        assert payload["model_status"] == "uncalibrated_shadow"
        assert payload["is_actionable"] is False
        assert payload["as_of_utc"] == "2026-08-15T00:00:00+00:00"
        assert payload["inputs"]["provenance"] == "caller_supplied_unverified"
        assert payload["inputs"]["portal_net_composite"] == 2.5
        assert payload["inputs"]["nil_budget_millions"] == 15.0
        assert payload["database_inputs"] == {"fixture": {"snapshot_id": "fixture:1"}}
        assert payload["projection"]["model_status"] == "uncalibrated_shadow"
        assert payload["projection"]["win_total_evaluations"]["0.5"]["is_actionable"] is False
        assert captured == {
            "team_id": "cfbd:1",
            "season": 2026,
            "as_of_utc": "2026-08-15T00:00:00+00:00",
            "portal_composite": 2.5,
            "nil_tier": None,
            "nil_budget_millions": 15.0,
            "qb_tier": "tier_2_quality_starter",
            "qb_continuity": "returning_starter_same_system",
            "posted_lines": [0.5],
            "input_manifest": {"fixture": {"snapshot_id": "fixture:1"}},
        }

        captured.clear()
        result = cli.main(
            [
                "futures",
                "--team-id", "cfbd:1",
                "--season", "2026",
                "--as-of", "2026-08-14T20:00:00-04:00",
                "--portal-net-composite", "2.5",
                "--nil-budget-millions", "15",
                "--qb-tier", "tier_2_quality_starter",
                "--qb-continuity", "returning_starter_same_system",
            ]
        )

        assert result == 0
        no_line_payload = json.loads(capsys.readouterr().out)
        assert no_line_payload["as_of_utc"] == "2026-08-15T00:00:00+00:00"
        assert no_line_payload["inputs"]["posted_lines"] == []
        assert no_line_payload["projection"]["win_total_evaluations"] == {}
        assert captured["posted_lines"] == []

    def test_init_db_creates_the_store_and_reports_what_it_applied(self, capsys):
        assert cli.main(["init-db"]) == 0
        assert paths.database_path().exists()
        from cfb_analytics.db import MIGRATIONS

        expected = [version for version, _, _, _ in MIGRATIONS]
        assert f"applied migrations {expected}" in capsys.readouterr().out

    def test_init_db_is_idempotent(self, capsys):
        cli.main(["init-db"])
        capsys.readouterr()
        assert cli.main(["init-db"]) == 0
        assert "already current" in capsys.readouterr().out

    def test_status_without_a_database_explains_what_to_run(self, capsys):
        assert cli.main(["status"]) == 1
        assert "init-db" in capsys.readouterr().out

    def test_status_reports_counts(self, capsys):
        cli.main(["init-db"])
        assert cli.main(["status"]) == 0
        assert "odds_snapshots" in capsys.readouterr().out

    def test_coverage_without_odds_says_so(self, capsys):
        cli.main(["init-db"])
        assert cli.main(["coverage"]) == 0
        assert "No odds captured yet" in capsys.readouterr().out

    def test_coverage_flags_absent_sharp_books(self, capsys, moneyline_market):
        from cfb_analytics import db
        from cfb_analytics.ingest import store
        from cfb_analytics.sources.outlier import parse_odds_rows

        cli.main(["init-db"])
        with db.open_db() as conn:
            store.upsert_team(conn, {"team_id": "h", "school": "H", "alias": "H", "market": "H"})
            store.upsert_team(conn, {"team_id": "a", "school": "A", "alias": "A", "market": "A"})
            store.upsert_game(conn, {
                "game_id": "evt-1", "season": 2026,
                "kickoff_utc": "2026-09-05T23:30:00+00:00",
                "football_date": "2026-09-05", "day_of_week": 5,
                "home_team_id": "h", "away_team_id": "a", "venue_name": None,
                "network": None, "status": "pregame"})
            store.insert_odds(
                conn,
                parse_odds_rows("evt-1", [moneyline_market], "2026-09-01T00:00:00+00:00"),
            )

        assert cli.main(["coverage"]) == 0
        out = capsys.readouterr().out
        assert "NONE" in out, "fixture has no sharp books, so it must report NONE"
        assert "PS3838" in out, "the tracked sharp set is named"

    def test_doctor_reports_blocked_cfbd_without_printing_a_key(self, capsys, monkeypatch):
        monkeypatch.setattr(paths, "outlier_session_dir", lambda: paths.data_dir() / "nope")
        assert cli.main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "cfbd      : BLOCKED - set CFBD_API_KEY" in out
        assert "SHADOW" in out

    def test_source_error_becomes_exit_code_2(self, capsys, monkeypatch):
        monkeypatch.setattr(paths, "outlier_session_dir", lambda: paths.data_dir() / "missing")
        assert cli.main(["schedule"]) == 2
        assert "error:" in capsys.readouterr().err

    def test_backfill_rejects_reverse_year_range_before_loading_credentials(self, capsys):
        assert cli.main(
            ["backfill-cfbd", "--start-year", "2025", "--end-year", "2024"]
        ) == 2
        assert "start-year" in capsys.readouterr().err


class TestCfbdClient:
    def test_games_use_query_parameters_from_the_docs(self):
        http = StubHttp({"/games": []})
        CFBDClient(http=http).fetch_games(2024, season_type="both", classification="fbs")
        assert "/games?year=2024&seasonType=both&classification=fbs" in http.calls[0]

    def test_teams_and_venues_are_array_endpoints(self):
        http = StubHttp({"/teams/fbs": [{"id": 1, "school": "Ohio State", "location": {}}],
                         "/venues": [{"id": 1}]})
        assert len(CFBDClient(http=http).fetch_fbs_teams(2024)) == 1
        assert len(CFBDClient(http=http).fetch_venues()) == 1
