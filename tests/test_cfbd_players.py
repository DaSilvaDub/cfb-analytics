"""CFBD roster and per-game passing: parsers, ingest, and the leakage-safe
presumptive-starter derivation."""

from __future__ import annotations

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.qb import presumptive_starter_as_of
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_players import (
    completed_weeks,
    ingest_game_passing,
    ingest_roster,
    weeks_missing_passing,
)
from cfb_analytics.sources.cfbd import parse_game_player_passing, parse_roster_row

# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

def _roster_row(**overrides):
    row = {
        "id": "4685522", "firstName": "Ty", "lastName": "Simpson", "team": "Alabama",
        "weight": 208, "height": 74, "jersey": 15, "year": 3, "position": "QB",
        "homeCity": "Martin", "homeState": "TN", "homeCountry": "USA",
    }
    row.update(overrides)
    return row


class TestParseRosterRow:
    def test_parses_a_normal_row(self):
        parsed = parse_roster_row(_roster_row())
        assert parsed == {
            "player_id": "cfbd:4685522", "name": "Ty Simpson", "team_name": "Alabama",
            "position": "QB", "class_year": 3, "height_in": 74, "weight_lb": 208,
            "home_state": "TN",
        }

    def test_missing_id_returns_none(self):
        assert parse_roster_row(_roster_row(id=None)) is None

    def test_missing_year_is_none_not_zero(self):
        """A missing years-in-program count must not silently read as 0
        (true freshman) -- that is a real, different value."""
        row = _roster_row()
        del row["year"]
        assert parse_roster_row(row)["class_year"] is None

    def test_missing_last_name_still_returns_first_name(self):
        row = _roster_row()
        del row["lastName"]
        assert parse_roster_row(row)["name"] == "Ty"

    def test_missing_both_names_is_none(self):
        row = _roster_row()
        del row["firstName"]
        del row["lastName"]
        assert parse_roster_row(row)["name"] is None

    def test_missing_team_raises(self):
        row = _roster_row()
        del row["team"]
        with pytest.raises(SchemaError):
            parse_roster_row(row)


def _game_players_row(**overrides):
    row = {
        "id": 401628319,
        "teams": [
            {
                "homeAway": "home",
                "categories": [
                    {
                        "name": "passing",
                        "types": [
                            {"name": "C/ATT", "athletes": [
                                {"id": "4432734", "name": "Jalen Milroe", "stat": "7/9"}]},
                            {"name": "YDS", "athletes": [
                                {"id": "4432734", "name": "Jalen Milroe", "stat": "200"}]},
                            {"name": "TD", "athletes": [
                                {"id": "4432734", "name": "Jalen Milroe", "stat": "3"}]},
                            {"name": "INT", "athletes": [
                                {"id": "4432734", "name": "Jalen Milroe", "stat": "0"}]},
                            {"name": "QBR", "athletes": [
                                {"id": "4432734", "name": "Jalen Milroe", "stat": "98.7"}]},
                        ],
                    }
                ],
            },
            {
                "homeAway": "away",
                "categories": [
                    {
                        "name": "passing",
                        "types": [
                            {"name": "C/ATT", "athletes": [
                                {"id": "4431948", "name": "TJ Finley", "stat": "18/31"}]},
                        ],
                    }
                ],
            },
        ],
    }
    row.update(overrides)
    return row


class TestParseGamePlayerPassing:
    def test_flattens_one_player_across_stat_types(self):
        rows = parse_game_player_passing(_game_players_row())
        milroe = next(r for r in rows if r["player_id"] == "cfbd:4432734")
        assert milroe["completions"] == 7
        assert milroe["attempts"] == 9
        assert milroe["yards"] == 200
        assert milroe["touchdowns"] == 3
        assert milroe["interceptions"] == 0
        assert milroe["qbr"] == pytest.approx(98.7)
        assert milroe["home_away"] == "home"
        assert milroe["game_id"] == "cfbd:401628319"

    def test_both_sides_are_returned(self):
        rows = parse_game_player_passing(_game_players_row())
        sides = {r["home_away"] for r in rows}
        assert sides == {"home", "away"}

    def test_side_with_no_passing_category_contributes_no_rows(self):
        row = _game_players_row()
        row["teams"][1]["categories"] = [{"name": "rushing", "types": []}]
        rows = parse_game_player_passing(row)
        assert not [r for r in rows if r["home_away"] == "away"]

    def test_drops_the_synthetic_team_pseudo_athlete(self):
        """CFBD emits a ' Team' entry with a negative id for unattributed
        plays. Verified live across a full week: always negative id, always
        named ' Team'. A real athlete id is never negative."""
        row = _game_players_row()
        row["teams"][1]["categories"][0]["types"].append(
            {"name": "C/ATT", "athletes": [{"id": "-5154", "name": " Team", "stat": "0/1"}]}
        )
        rows = parse_game_player_passing(row)
        assert "cfbd:-5154" not in {r["player_id"] for r in rows}

    def test_malformed_catt_yields_none_rather_than_a_guess(self):
        row = _game_players_row()
        row["teams"][0]["categories"][0]["types"][0]["athletes"][0]["stat"] = "garbled"
        rows = parse_game_player_passing(row)
        milroe = next(r for r in rows if r["player_id"] == "cfbd:4432734")
        assert milroe["completions"] is None
        assert milroe["attempts"] is None

    def test_side_missing_home_away_is_skipped(self):
        row = _game_players_row()
        del row["teams"][0]["homeAway"]
        rows = parse_game_player_passing(row)
        assert all(r["player_id"] != "cfbd:4432734" for r in rows)

    def test_no_teams_at_all_yields_no_rows(self):
        assert parse_game_player_passing({"id": 1, "teams": []}) == []


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

class FakeCfbdClient:
    def __init__(self, roster=None, game_players=None):
        self._roster = roster or []
        self._game_players = game_players or []

    def fetch_roster(self, year):
        return self._roster

    def fetch_game_players(self, year, week, *, season_type="regular"):
        return self._game_players


def _seed_cfbd_team(conn, team_id, school):
    store.upsert_team(conn, {"team_id": team_id, "school": school, "alias": None, "market": None})


def _seed_cfbd_game(conn, game_id, home_id, away_id, season=2024, week=1, completed=1):
    conn.execute(
        """INSERT INTO games (game_id, season, week, season_type, kickoff_utc, football_date,
                              home_team_id, away_team_id, completed, source, ingested_utc)
           VALUES (?, ?, ?, 'regular', '2024-08-31T23:30:00+00:00', '2024-08-31',
                   ?, ?, ?, 'cfbd', 'x')""",
        (game_id, season, week, home_id, away_id, completed),
    )


class TestIngestRoster:
    def test_resolves_team_by_exact_school_name(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        client = FakeCfbdClient(roster=[_roster_row()])
        summary = ingest_roster(conn, client, 2025)
        assert summary.written == 1
        row = conn.execute(
            "SELECT team_id FROM player_seasons WHERE player_id = 'cfbd:4685522'"
        ).fetchone()
        assert row["team_id"] == "cfbd:333"

    def test_unmatched_team_is_counted_not_guessed(self, conn):
        client = FakeCfbdClient(roster=[_roster_row(team="Nowhere State")])
        summary = ingest_roster(conn, client, 2025)
        assert summary.written == 0
        assert "Nowhere State" in summary.unmatched_teams

    def test_stores_class_year_and_position(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        ingest_roster(conn, FakeCfbdClient(roster=[_roster_row()]), 2025)
        row = conn.execute(
            "SELECT position, class_year FROM player_seasons WHERE player_id = 'cfbd:4685522'"
        ).fetchone()
        assert row["position"] == "QB"
        assert row["class_year"] == 3

    def test_rerun_updates_rather_than_duplicates(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        client = FakeCfbdClient(roster=[_roster_row()])
        ingest_roster(conn, client, 2025)
        ingest_roster(conn, client, 2025)
        n = conn.execute(
            "SELECT COUNT(*) n FROM player_seasons WHERE player_id = 'cfbd:4685522'"
        ).fetchone()["n"]
        assert n == 1

    def test_row_with_no_player_id_is_silently_skipped(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        summary = ingest_roster(conn, FakeCfbdClient(roster=[_roster_row(id=None)]), 2025)
        assert summary.written == 0
        assert summary.rows_seen == 1

    def test_summary_text_reports_unmatched_count(self, conn):
        client = FakeCfbdClient(roster=[_roster_row(team="Nowhere State")])
        text = ingest_roster(conn, client, 2025).as_text()
        assert "unmatched teams  : 1" in text


class TestIngestGamePassing:
    def test_resolves_team_via_game_home_away_not_by_name(self, conn):
        """No team name or id exists at the team level in this payload at
        all -- only homeAway. Resolution must go through the games table."""
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        _seed_cfbd_team(conn, "cfbd:99", "Western Kentucky")
        _seed_cfbd_game(conn, "cfbd:401628319", "cfbd:333", "cfbd:99")

        summary = ingest_game_passing(
            conn, FakeCfbdClient(game_players=[_game_players_row()]), 2024, 1
        )
        assert summary.rows_written == 2
        row = conn.execute(
            "SELECT team_id FROM player_game_passing WHERE player_id = 'cfbd:4432734'"
        ).fetchone()
        assert row["team_id"] == "cfbd:333"  # home side

    def test_game_not_yet_in_store_is_counted_not_guessed(self, conn):
        summary = ingest_game_passing(
            conn, FakeCfbdClient(game_players=[_game_players_row()]), 2024, 1
        )
        assert summary.games_unmatched == 1
        assert summary.rows_written == 0

    def test_rerun_replaces_rather_than_duplicates(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        _seed_cfbd_team(conn, "cfbd:99", "Western Kentucky")
        _seed_cfbd_game(conn, "cfbd:401628319", "cfbd:333", "cfbd:99")
        client = FakeCfbdClient(game_players=[_game_players_row()])
        ingest_game_passing(conn, client, 2024, 1)
        ingest_game_passing(conn, client, 2024, 1)
        n = conn.execute("SELECT COUNT(*) n FROM player_game_passing").fetchone()["n"]
        assert n == 2

    def test_stores_the_box_score_stats(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        _seed_cfbd_team(conn, "cfbd:99", "Western Kentucky")
        _seed_cfbd_game(conn, "cfbd:401628319", "cfbd:333", "cfbd:99")
        ingest_game_passing(
            conn, FakeCfbdClient(game_players=[_game_players_row()]), 2024, 1
        )
        row = conn.execute(
            "SELECT attempts, yards, touchdowns FROM player_game_passing "
            "WHERE player_id = 'cfbd:4432734'"
        ).fetchone()
        assert row["attempts"] == 9
        assert row["yards"] == 200
        assert row["touchdowns"] == 3


class TestCompletedWeeksAndMissing:
    def test_completed_weeks_lists_only_finished_weeks(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        _seed_cfbd_team(conn, "cfbd:99", "Western Kentucky")
        _seed_cfbd_game(conn, "cfbd:g1", "cfbd:333", "cfbd:99", week=1, completed=1)
        _seed_cfbd_game(conn, "cfbd:g2", "cfbd:333", "cfbd:99", week=2, completed=0)
        assert completed_weeks(conn, 2024) == [1]

    def test_raises_when_season_has_no_stored_games(self, conn):
        with pytest.raises(SchemaError, match="backfill games first"):
            completed_weeks(conn, 2024)

    def test_weeks_missing_passing_excludes_already_captured_weeks(self, conn):
        _seed_cfbd_team(conn, "cfbd:333", "Alabama")
        _seed_cfbd_team(conn, "cfbd:99", "Western Kentucky")
        _seed_cfbd_game(conn, "cfbd:401628319", "cfbd:333", "cfbd:99", week=1)
        _seed_cfbd_game(conn, "cfbd:g2", "cfbd:333", "cfbd:99", week=2)
        ingest_game_passing(
            conn, FakeCfbdClient(game_players=[_game_players_row()]), 2024, 1
        )
        assert weeks_missing_passing(conn, 2024) == [2]

    def test_weeks_missing_passing_raises_when_nothing_stored(self, conn):
        with pytest.raises(SchemaError):
            weeks_missing_passing(conn, 2024)


# --------------------------------------------------------------------------
# Presumptive starter (leakage-safe, honest about being a proxy)
# --------------------------------------------------------------------------

def _seed_cfbd_game_if_absent(conn, game_id, home_id, away_id, season, week, kickoff):
    exists = conn.execute("SELECT 1 FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if exists:
        return
    _seed_cfbd_team(conn, home_id, home_id)
    _seed_cfbd_team(conn, away_id, away_id)
    conn.execute(
        """INSERT INTO games (game_id, season, week, season_type, kickoff_utc, football_date,
                              home_team_id, away_team_id, completed, source, ingested_utc)
           VALUES (?, ?, ?, 'regular', ?, ?, ?, ?, 1, 'cfbd', 'x')""",
        (game_id, season, week, kickoff, kickoff[:10], home_id, away_id),
    )


def _passing_row(conn, game_id, team_id, player_id, name, attempts, week, kickoff, season=2024):
    _seed_cfbd_game_if_absent(conn, game_id, team_id, "cfbd:opponent", season, week, kickoff)
    store.upsert_player(conn, player_id, name)
    store.upsert_player_game_passing(conn, {
        "game_id": game_id, "team_id": team_id, "player_id": player_id,
        "season": season, "week": week, "completions": None, "attempts": attempts,
        "yards": None, "avg_yards": None, "touchdowns": None, "interceptions": None,
        "qbr": None, "source": "cfbd",
    })


class TestPresumptiveStarter:
    def test_none_when_no_prior_games_this_season(self, conn):
        """Week 1: the correct, honest answer is 'unknown', not a guess --
        exactly the case the CORE-tier QB-status blocker already covers."""
        assert presumptive_starter_as_of(
            conn, "cfbd:333", 2024, "2024-08-31T23:30:00+00:00"
        ) is None

    def test_identifies_the_clear_usage_leader(self, conn):
        _passing_row(conn, "cfbd:g1", "cfbd:333", "cfbd:qb1", "Starter", 30, 1,
                     "2024-08-31T23:30:00+00:00")
        _passing_row(conn, "cfbd:g1", "cfbd:333", "cfbd:qb2", "Backup", 2, 1,
                     "2024-08-31T23:30:00+00:00")
        result = presumptive_starter_as_of(conn, "cfbd:333", 2024, "2024-09-07T23:30:00+00:00")
        assert result.player_id == "cfbd:qb1"
        assert result.name == "Starter"
        assert result.attempts == 30
        assert result.attempt_share == pytest.approx(30 / 32)

    def test_excludes_games_at_or_after_the_given_kickoff(self, conn):
        """The leakage boundary: a game on the SAME slate must not count as
        prior information for another game on that slate."""
        _passing_row(conn, "cfbd:g1", "cfbd:333", "cfbd:qb1", "P1", 20, 1,
                     "2024-09-07T23:30:00+00:00")
        result = presumptive_starter_as_of(conn, "cfbd:333", 2024, "2024-09-07T23:30:00+00:00")
        assert result is None

    def test_lookback_window_limits_to_the_n_most_recent_games(self, conn):
        weeks = [2, 3, 4]
        _passing_row(conn, "cfbd:g1", "cfbd:333", "cfbd:old_qb", "Old", 40, 1,
                     "2024-08-31T23:30:00+00:00")
        for week in weeks:
            kickoff = f"2024-09-{week + 6:02d}T23:30:00+00:00"
            _passing_row(conn, f"cfbd:g{week}", "cfbd:333", "cfbd:new_qb", "New", 10,
                         week, kickoff)
        result = presumptive_starter_as_of(
            conn, "cfbd:333", 2024, "2024-10-01T23:30:00+00:00", lookback_games=3
        )
        # Only the 3 most recent games count; the week-1 40-attempt game is
        # outside the window, so New (10+10+10=30) beats Old (0 in-window).
        assert result.player_id == "cfbd:new_qb"

    def test_different_team_is_unaffected(self, conn):
        _passing_row(conn, "cfbd:g1", "cfbd:333", "cfbd:qb1", "QB", 20, 1,
                     "2024-08-31T23:30:00+00:00")
        assert presumptive_starter_as_of(
            conn, "cfbd:other", 2024, "2024-09-07T23:30:00+00:00"
        ) is None

    def test_result_is_not_a_confirmed_starter_by_construction(self, conn):
        """Regression guard for the documented contract: this module must
        never claim confidence beyond 'who has been throwing it lately'."""
        from cfb_analytics.features.qb import PresumptiveStarter
        assert not hasattr(PresumptiveStarter, "confirmed")
        assert not hasattr(PresumptiveStarter, "is_starter")
