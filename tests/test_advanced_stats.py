"""Tests for the net offense-minus-defense advanced-stat diffs."""

from __future__ import annotations

from collections import defaultdict

import pytest

from cfb_analytics.features.advanced_stats import (
    ADVANCED_STAT_KEYS,
    load_advanced_stats,
    team_advanced_nets,
)
from cfb_analytics.ingest import store


def _row(team_id, season, week, side, as_of_utc, **stats):
    base = {
        "season": season, "week": week, "team_id": team_id, "side": side,
        "as_of_utc": as_of_utc, "provenance_mode": "reconstructed", "garbage_excluded": 1,
        "plays": 60, "drives": 12, "total_ppa": 5.0,
        "ppa": None, "success_rate": None, "explosiveness": None,
        "points_per_opportunity": None, "havoc": None, "line_yards": None,
        "stuff_rate": None, "passing_ppa": None, "rushing_ppa": None,
        "passing_success_rate": None, "rushing_success_rate": None,
    }
    base.update(stats)
    return base


def _cache_from_rows(rows):
    """The same in-memory shape ``load_advanced_stats`` returns, without a
    real connection -- these tests are about the as-of/sign-convention math,
    not the SQL."""
    cache: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cache[row["team_id"]][row["side"]].append(row)
    return cache


class TestTeamAdvancedNets:
    def test_a_team_with_no_rows_gets_all_zero_nets(self):
        cache = _cache_from_rows([])
        nets = team_advanced_nets(cache, "nowhere", "2019-09-05T00:00:00+00:00")
        assert nets == dict.fromkeys(ADVANCED_STAT_KEYS, 0.0)

    def test_off_minus_def_stats_use_offense_produced_minus_defense_allowed(self):
        cache = _cache_from_rows([
            _row("h", 2019, 1, "off", "2019-09-02T00:00:00+00:00", ppa=0.3, success_rate=0.5),
            _row("h", 2019, 1, "def", "2019-09-02T00:00:00+00:00", ppa=0.1, success_rate=0.4),
        ])
        nets = team_advanced_nets(cache, "h", "2019-09-05T00:00:00+00:00")
        assert nets["ppa"] == pytest.approx(0.2)  # produced 0.3, allowed 0.1: net = +0.2
        assert nets["success_rate"] == pytest.approx(0.1)

    def test_def_minus_off_stats_flip_the_sign(self):
        """havoc/stuff_rate: high 'off' (suffered) is bad, high 'def' (inflicted) is good."""
        cache = _cache_from_rows([
            _row("h", 2019, 1, "off", "2019-09-02T00:00:00+00:00", havoc=0.10, stuff_rate=0.15),
            _row("h", 2019, 1, "def", "2019-09-02T00:00:00+00:00", havoc=0.20, stuff_rate=0.25),
        ])
        nets = team_advanced_nets(cache, "h", "2019-09-05T00:00:00+00:00")
        assert nets["havoc"] == pytest.approx(0.10)  # inflicted 0.20 minus suffered 0.10
        assert nets["stuff_rate"] == pytest.approx(0.10)

    def test_a_null_field_on_a_present_row_defaults_to_zero_not_a_crash(self):
        cache = _cache_from_rows([
            _row("h", 2019, 1, "off", "2019-09-02T00:00:00+00:00", ppa=0.3),
            _row("h", 2019, 1, "def", "2019-09-02T00:00:00+00:00", ppa=0.1),
            # explosiveness left None on both rows (CFBD sometimes omits it)
        ])
        nets = team_advanced_nets(cache, "h", "2019-09-05T00:00:00+00:00")
        assert nets["explosiveness"] == 0.0

    def test_only_an_admissible_row_strictly_before_as_of_utc_is_used(self):
        cache = _cache_from_rows([
            _row("h", 2019, 1, "off", "2019-09-02T00:00:00+00:00", ppa=0.1),
            _row("h", 2019, 1, "def", "2019-09-02T00:00:00+00:00", ppa=0.0),
            # Week 2's cumulative snapshot -- must NOT leak into a week-2 prediction.
            _row("h", 2019, 2, "off", "2019-09-09T00:00:00+00:00", ppa=0.9),
            _row("h", 2019, 2, "def", "2019-09-09T00:00:00+00:00", ppa=0.0),
        ])
        nets = team_advanced_nets(cache, "h", "2019-09-09T00:00:00+00:00")
        assert nets["ppa"] == pytest.approx(0.1)


class TestLoadAdvancedStats:
    def test_reads_back_what_was_stored(self, conn):
        store.upsert_team(conn, {
            "team_id": "h", "school": "Home U", "alias": None, "market": None,
        })
        store.insert_team_advanced(conn, [
            _row("h", 2019, 1, "off", "2019-09-02T00:00:00+00:00", ppa=0.3),
            _row("h", 2019, 1, "def", "2019-09-02T00:00:00+00:00", ppa=0.1),
        ])
        cache = load_advanced_stats(conn, 2019)
        nets = team_advanced_nets(cache, "h", "2019-09-05T00:00:00+00:00")
        assert nets["ppa"] == pytest.approx(0.2)

    def test_a_different_season_is_not_mixed_in(self, conn):
        store.upsert_team(conn, {
            "team_id": "h", "school": "Home U", "alias": None, "market": None,
        })
        store.insert_team_advanced(conn, [
            _row("h", 2018, 1, "off", "2018-09-02T00:00:00+00:00", ppa=9.9),
            _row("h", 2018, 1, "def", "2018-09-02T00:00:00+00:00", ppa=0.0),
        ])
        cache = load_advanced_stats(conn, 2019)
        nets = team_advanced_nets(cache, "h", "2019-09-05T00:00:00+00:00")
        assert nets["ppa"] == 0.0
