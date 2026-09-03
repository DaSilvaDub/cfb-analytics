from __future__ import annotations

import pytest

from cfb_analytics import cli, db
from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest.cfbd_ingest import backfill_years
from cfb_analytics.sources.cfbd import (
    parse_game,
    parse_team,
    parse_team_aliases,
    parse_team_season,
    parse_venue,
)


class StubCfbdClient:
    def __init__(self, *, venues, teams_by_year, games_by_year):
        self.venues = venues
        self.teams_by_year = teams_by_year
        self.games_by_year = games_by_year

    def fetch_venues(self):
        return self.venues

    def fetch_fbs_teams(self, year: int):
        return self.teams_by_year[year]

    def fetch_games(self, year: int, *, season_type: str = "both", classification: str = "fbs"):
        assert season_type == "both"
        assert classification == "fbs"
        return self.games_by_year[year]


def _team_row():
    return {
        "id": 200,
        "school": "Ohio State",
        "mascot": "Buckeyes",
        "abbreviation": "OSU",
        "alternateNames": ["Ohio State", "OSU"],
        "conference": "Big Ten",
        "division": "East",
        "classification": "fbs",
        "location": {
            "id": 10,
            "name": "Ohio Stadium",
            "city": "Columbus",
            "state": "OH",
            "timezone": "America/New_York",
            "latitude": 40.0,
            "longitude": -83.0,
            "elevation": "250",
            "capacity": 100000,
            "grass": True,
            "dome": False,
        },
    }


def _game_row():
    return {
        "id": 999,
        "season": 2024,
        "week": 1,
        "seasonType": "regular",
        "startDate": "2024-08-31T19:30:00-04:00",
        "startTimeTBD": False,
        "completed": True,
        "neutralSite": False,
        "conferenceGame": False,
        "attendance": 100000,
        "venueId": 10,
        "venue": "Ohio Stadium",
        "homeId": 200,
        "homeTeam": "Ohio State",
        "homeConference": "Big Ten",
        "homeClassification": "fbs",
        "homePoints": 31,
        "homeLineScores": [7, 7, 10, 7],
        "homePostgameWinProbability": 0.9,
        "homePregameElo": 1750,
        "homePostgameElo": 1760,
        "awayId": 201,
        "awayTeam": "Akron",
        "awayConference": "MAC",
        "awayClassification": "fbs",
        "awayPoints": 10,
        "awayLineScores": [3, 0, 0, 7],
        "awayPostgameWinProbability": 0.1,
        "awayPregameElo": 1450,
        "awayPostgameElo": 1440,
        "excitementIndex": 1.2,
        "highlights": None,
        "notes": None,
        "playoff": {},
    }


class TestCfbdParsing:
    def test_parse_team_and_season(self):
        team = parse_team(_team_row())
        season = parse_team_season(_team_row(), year=2024)
        assert team["team_id"] == "cfbd:200"
        assert team["cfbd_id"] == 200
        assert team["alias"] == "OSU"
        assert season["season"] == 2024
        assert season["conference"] == "Big Ten"
        assert season["division"] == "East"

    def test_parse_team_aliases(self):
        aliases = parse_team_aliases(_team_row())
        assert {row["alias"] for row in aliases} == {"Ohio State", "OSU"}

    def test_parse_venue(self):
        venue = parse_venue(_team_row()["location"])
        assert venue is not None
        assert venue["venue_id"] == "10"
        assert venue["surface"] == "grass"
        assert venue["dome"] == 0
        assert venue["elevation_m"] is None, "CFBD does not document the elevation unit"

    def test_parse_venue_rejects_non_boolean_grass_or_dome(self):
        with pytest.raises(SchemaError, match="grass"):
            parse_venue({"id": 10, "name": "X", "grass": "false"})
        with pytest.raises(SchemaError, match="dome"):
            parse_venue({"id": 10, "name": "X", "dome": "true"})

    def test_parse_venue_without_name_is_not_storable(self):
        assert parse_venue({"id": 10, "name": None}) is None

    def test_parse_game(self):
        game = parse_game(_game_row())
        assert game["game_id"] == "cfbd:999"
        assert game["home_team_id"] == "cfbd:200"
        assert game["away_team_id"] == "cfbd:201"
        assert game["completed"] == 1
        assert game["football_date"] == "2024-08-31"

    def test_parse_game_requires_start_date(self):
        row = _game_row()
        row["startDate"] = None
        try:
            parse_game(row)
        except SchemaError as exc:
            assert "startDate" in str(exc)
        else:
            raise AssertionError("expected SchemaError")


class TestCfbdBackfill:
    def test_backfill_writes_historical_rows(self):
        cli.main(["init-db"])
        with db.open_db() as conn:
            summary = backfill_years(
                conn,
                StubCfbdClient(
                    venues=[_team_row()["location"]],
                    teams_by_year={
                        2024: [
                            _team_row(),
                            {
                                **_team_row(),
                                "id": 201,
                                "school": "Akron",
                                "abbreviation": "AKR",
                                "alternateNames": ["Akron"],
                                "conference": "MAC",
                            },
                        ]
                    },
                    games_by_year={2024: [_game_row()]},
                ),
                start_year=2024,
                end_year=2024,
            )
            assert summary.seasons == 1
            assert conn.execute("SELECT COUNT(*) AS n FROM venues").fetchone()["n"] == 1
            assert conn.execute("SELECT COUNT(*) AS n FROM team_seasons").fetchone()["n"] == 2
            assert conn.execute("SELECT COUNT(*) AS n FROM team_aliases").fetchone()["n"] >= 2
            game = conn.execute(
                "SELECT source, completed, home_points FROM games WHERE game_id = 'cfbd:999'"
            ).fetchone()
            assert game["source"] == "cfbd"
            assert game["completed"] == 1
            assert game["home_points"] == 31

            run = conn.execute(
                "SELECT status, rows_written FROM runs WHERE command LIKE 'backfill-cfbd%'"
            ).fetchone()
            assert run["status"] == "ok"
            assert run["rows_written"] > 0
            endpoints = {
                row["endpoint"]
                for row in conn.execute(
                    "SELECT endpoint FROM source_health WHERE source = 'cfbd'"
                )
            }
            assert endpoints == {"venues", "teams/fbs:2024", "games:2024"}
            season = conn.execute(
                "SELECT conference, division "
                "FROM team_seasons "
                "WHERE team_id = 'cfbd:200' AND season = 2024"
            ).fetchone()
            assert dict(season) == {"conference": "Big Ten", "division": "East"}

    def test_game_only_opponent_is_materialized_before_foreign_key(self):
        """FBS-filtered games can still contain an FCS opponent absent from teams/fbs."""
        fbs_only = [_team_row()]
        game = {**_game_row(), "awayId": 9001, "awayTeam": "FCS Visitor",
                "awayConference": "Big Sky", "awayClassification": "fcs"}
        with db.open_db() as conn:
            backfill_years(
                conn,
                StubCfbdClient(
                    venues=[_team_row()["location"]],
                    teams_by_year={2024: fbs_only},
                    games_by_year={2024: [game]},
                ),
                start_year=2024,
                end_year=2024,
            )
            opponent = conn.execute(
                "SELECT school, classification FROM teams WHERE team_id = 'cfbd:9001'"
            ).fetchone()
            assert dict(opponent) == {"school": "FCS Visitor", "classification": "fcs"}
            assert conn.execute(
                "SELECT away_team_id FROM games WHERE game_id = 'cfbd:999'"
            ).fetchone()["away_team_id"] == "cfbd:9001"

    def test_reverse_year_range_fails_before_fetching(self):
        class NeverFetch:
            def fetch_venues(self):
                raise AssertionError("must validate before source access")

        with db.open_db() as conn, pytest.raises(SchemaError, match="start_year"):
            backfill_years(conn, NeverFetch(), start_year=2025, end_year=2024)

    def test_schema_drift_marks_run_and_source_failed(self):
        broken = {**_game_row(), "startDate": None}
        with db.open_db() as conn, pytest.raises(SchemaError, match="startDate"):
            backfill_years(
                conn,
                StubCfbdClient(
                    venues=[_team_row()["location"]],
                    teams_by_year={2024: [_team_row()]},
                    games_by_year={2024: [broken]},
                ),
                start_year=2024,
                end_year=2024,
            )
        with db.open_db() as conn:
            run = conn.execute(
                "SELECT status, error FROM runs WHERE command LIKE 'backfill-cfbd%'"
            ).fetchone()
            assert run["status"] == "failed"
            assert "startDate" in run["error"]
            health = conn.execute(
                "SELECT ok, detail FROM source_health WHERE endpoint = 'games:2024'"
            ).fetchone()
            assert health["ok"] == 0
            assert "startDate" in health["detail"]

    def test_cli_backfill_requires_cfbd_key(self, capsys, monkeypatch):
        monkeypatch.delenv("CFBD_API_KEY", raising=False)
        assert cli.main(["backfill-cfbd", "--start-year", "2024", "--end-year", "2024"]) == 2
        assert "CFBD_API_KEY" in capsys.readouterr().err
