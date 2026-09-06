from __future__ import annotations

from cfb_analytics.features.preseason import returning_ppa_zscores, talent_zscores
from cfb_analytics.ingest import store


def _seed_team(conn, team_id):
    store.upsert_team(conn, {"team_id": team_id, "school": team_id, "alias": None, "market": None})


def _seed_talent(conn, season, team_id, talent_composite):
    store.insert_team_talent(conn, [{
        "season": season, "team_id": team_id, "availability_class": "preseason",
        "talent_composite": talent_composite,
    }])


def _seed_returning(conn, season, team_id, percent_ppa):
    store.insert_returning_production(conn, [{
        "season": season, "team_id": team_id, "availability_class": "preseason",
        "total_ppa": None, "passing_ppa": None, "receiving_ppa": None, "rushing_ppa": None,
        "percent_ppa": percent_ppa, "percent_passing_ppa": None,
        "percent_receiving_ppa": None, "percent_rushing_ppa": None,
        "usage": None, "passing_usage": None, "receiving_usage": None, "rushing_usage": None,
    }])


class TestTalentZscores:
    def test_computes_a_real_zscore_across_teams(self, conn):
        for team_id in ("a", "b", "c"):
            _seed_team(conn, team_id)
        _seed_talent(conn, 2026, "a", 900.0)
        _seed_talent(conn, 2026, "b", 500.0)
        _seed_talent(conn, 2026, "c", 100.0)

        result = talent_zscores(conn, 2026)
        assert result["a"] > 0
        assert result["b"] == 0.0
        assert result["c"] < 0

    def test_a_different_season_is_not_mixed_in(self, conn):
        _seed_team(conn, "a")
        _seed_talent(conn, 2025, "a", 900.0)
        assert talent_zscores(conn, 2026) == {}

    def test_no_data_returns_an_empty_map(self, conn):
        assert talent_zscores(conn, 2026) == {}


class TestReturningPpaZscores:
    def test_computes_a_real_zscore_across_teams(self, conn):
        for team_id in ("a", "b"):
            _seed_team(conn, team_id)
        _seed_returning(conn, 2026, "a", 0.9)
        _seed_returning(conn, 2026, "b", 0.1)

        result = returning_ppa_zscores(conn, 2026)
        assert result["a"] > 0
        assert result["b"] < 0

    def test_no_data_returns_an_empty_map(self, conn):
        assert returning_ppa_zscores(conn, 2026) == {}
