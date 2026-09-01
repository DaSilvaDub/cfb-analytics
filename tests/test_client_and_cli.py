from __future__ import annotations

import pytest

from cfb_analytics import cli, paths
from cfb_analytics.errors import AuthRequiredError, UnknownLeagueError
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
        assert commands == {"init-db", "doctor", "schedule", "status", "ingest", "coverage"}

    def test_unimplemented_phases_are_absent(self):
        """`--help` must not advertise anything that does not run."""
        parser = cli.build_parser()
        commands = set([a for a in parser._actions if getattr(a, "choices", None)][0].choices)
        assert not commands & {"features", "train", "backtest", "slate", "parlay", "settle"}

    def test_ingest_requires_a_date(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["ingest"])


class TestCliCommands:
    def test_init_db_creates_the_store_and_reports_what_it_applied(self, capsys):
        assert cli.main(["init-db"]) == 0
        assert paths.database_path().exists()
        assert "applied migrations [1, 2]" in capsys.readouterr().out

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
        monkeypatch.delenv("CFBD_API_KEY", raising=False)
        monkeypatch.setattr(paths, "outlier_session_dir", lambda: paths.data_dir() / "nope")
        assert cli.main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "cfbd      : BLOCKED - set CFBD_API_KEY" in out
        assert "SHADOW" in out

    def test_source_error_becomes_exit_code_2(self, capsys, monkeypatch):
        monkeypatch.setattr(paths, "outlier_session_dir", lambda: paths.data_dir() / "missing")
        assert cli.main(["schedule"]) == 2
        assert "error:" in capsys.readouterr().err
