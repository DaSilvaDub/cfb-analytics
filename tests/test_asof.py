"""Leakage-guard tests.

The guard is the highest-risk component in the build: a backtest that leaks
looks better than a real model and is worthless. These tests encode the
specific leaks that are easy to write by accident.
"""

from __future__ import annotations

import pytest

from cfb_analytics.errors import LeakageError
from cfb_analytics.features.asof import AsOfReader, reader_for_game

KICKOFF = "2026-09-05T23:30:00+00:00"


@pytest.fixture
def reader():
    return AsOfReader(game_id="evt-1", kickoff_utc=KICKOFF, season=2026)


def row(as_of: str, **extra):
    return {"as_of_utc": as_of, **extra}


class TestCheck:
    def test_accepts_a_pre_kickoff_timestamp(self, reader):
        assert reader.check("2026-09-05T20:00:00+00:00", what="odds")

    def test_rejects_a_post_kickoff_timestamp(self, reader):
        with pytest.raises(LeakageError, match="at or after kickoff"):
            reader.check("2026-09-06T01:00:00+00:00", what="odds")

    def test_rejects_a_timestamp_exactly_at_kickoff(self, reader):
        """Kickoff itself is not 'before kickoff'."""
        with pytest.raises(LeakageError, match="at or after kickoff"):
            reader.check(KICKOFF, what="odds")

    def test_unparseable_timestamp_is_treated_as_a_leak_not_as_missing(self, reader):
        with pytest.raises(LeakageError, match="not a parseable timestamp"):
            reader.check("sometime last week", what="odds")

    def test_none_timestamp_is_a_leak(self, reader):
        with pytest.raises(LeakageError):
            reader.check(None, what="odds")


class TestAdmissible:
    def test_drops_post_kickoff_rows_and_keeps_the_rest(self, reader):
        rows = [
            row("2026-09-01T00:00:00+00:00", tag="early"),
            row("2026-09-05T23:00:00+00:00", tag="late-but-ok"),
            row("2026-09-06T02:00:00+00:00", tag="closing-line"),
        ]
        kept = reader.admissible(rows, what="odds")
        assert [r["tag"] for r in kept] == ["early", "late-but-ok"]

    def test_closing_lines_and_settlements_are_filtered_not_fatal(self, reader):
        """The store legitimately holds post-kickoff rows; filtering them is
        this method's job, so it must not raise."""
        assert reader.admissible([row("2026-09-09T00:00:00+00:00")], what="settle") == []

    def test_row_without_a_timestamp_is_refused(self, reader):
        with pytest.raises(LeakageError, match="cannot be proven pre-kickoff"):
            reader.admissible([{"book": "FANDUEL"}], what="odds")

    def test_empty_input_is_empty_output(self, reader):
        assert reader.admissible([], what="odds") == []


class TestLatest:
    def test_returns_the_most_recent_admissible_row(self, reader):
        rows = [
            row("2026-09-01T00:00:00+00:00", tag="old"),
            row("2026-09-05T22:00:00+00:00", tag="newest-pre-kick"),
            row("2026-09-06T05:00:00+00:00", tag="post-kick"),
        ]
        assert reader.latest(rows, what="odds")["tag"] == "newest-pre-kick"

    def test_returns_none_when_everything_is_post_kickoff(self, reader):
        assert reader.latest([row("2026-09-07T00:00:00+00:00")], what="odds") is None


class TestPreseasonSeasonCheck:
    """The classic leak: a season-FINAL rating joined onto a mid-season game
    and labelled 'preseason'."""

    def test_accepts_a_row_labelled_with_this_season(self, reader):
        reader.check_preseason_season(2026, feature="returning_production")

    def test_rejects_a_row_from_a_different_season(self, reader):
        with pytest.raises(LeakageError, match="labelled season"):
            reader.check_preseason_season(2025, feature="returning_production")

    def test_rejects_an_unlabelled_row(self, reader):
        with pytest.raises(LeakageError):
            reader.check_preseason_season(None, feature="talent")

    def test_reader_without_a_season_cannot_verify_and_refuses(self):
        bare = AsOfReader(game_id="evt-1", kickoff_utc=KICKOFF)
        with pytest.raises(LeakageError, match="no season"):
            bare.check_preseason_season(2026, feature="talent")


class TestAvailabilityClass:
    @pytest.mark.parametrize("cls", ["preseason", "weekly", "pregame"])
    def test_accepts_declared_classes(self, reader, cls):
        reader.check_availability_class(cls, feature="x")

    def test_refuses_an_undeclared_class(self, reader):
        with pytest.raises(LeakageError, match="not\n?.*one of|not one of"):
            reader.check_availability_class("whenever", feature="x")


class TestStaleness:
    def test_measures_minutes_before_kickoff(self, reader):
        assert reader.staleness_minutes("2026-09-05T22:30:00+00:00") == pytest.approx(60.0)

    def test_is_negative_for_post_kickoff_rows(self, reader):
        assert reader.staleness_minutes("2026-09-06T00:30:00+00:00") < 0


class TestReaderForGame:
    def test_builds_from_a_game_mapping(self):
        r = reader_for_game(
            {"game_id": "g", "kickoff_utc": KICKOFF, "season": 2026})
        assert r.game_id == "g" and r.season == 2026

    @pytest.mark.parametrize("game", [
        {"kickoff_utc": KICKOFF},
        {"game_id": "g"},
        {"game_id": "g", "kickoff_utc": ""},
    ])
    def test_refuses_an_incomplete_game(self, game):
        with pytest.raises(LeakageError, match="missing"):
            reader_for_game(game)
