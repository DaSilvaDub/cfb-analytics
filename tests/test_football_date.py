"""Slate membership is the US Eastern calendar date, not the UTC date.

These are regression tests for a real defect found against the live 2026-09-05
feed: UTC grouping pulled in four Friday-night games and dropped four
Saturday-night West Coast games, misassigning 8 of a 34-game slate.
"""

from __future__ import annotations

import pytest

from cfb_analytics.sources.outlier import parse_event
from cfb_analytics.utils import football_date, weekday_matches_feed


class TestFootballDate:
    @pytest.mark.parametrize(
        "kickoff_utc,expected,why",
        [
            ("2026-09-05T16:00:00+00:00", "2026-09-05", "noon ET Saturday"),
            ("2026-09-05T23:30:00+00:00", "2026-09-05", "7:30pm ET Saturday"),
            # The four games UTC grouping wrongly EXCLUDED from the Saturday slate.
            ("2026-09-06T02:00:00+00:00", "2026-09-05", "10pm ET Sat: CMU@UNM, UNLV@HAW"),
            ("2026-09-06T02:30:00+00:00", "2026-09-05", "10:30pm ET Sat: UCLA@CAL, WKU@NEV"),
            # The games UTC grouping wrongly INCLUDED (they are Friday night).
            ("2026-09-05T00:00:00+00:00", "2026-09-04", "8pm ET Friday: UTEP@OKLA"),
            ("2026-09-05T01:00:00+00:00", "2026-09-04", "9pm ET Friday: FRES@USC"),
            # Genuinely Sunday.
            ("2026-09-06T20:00:00+00:00", "2026-09-06", "4pm ET Sunday: WSU@WASH"),
        ],
    )
    def test_assigns_the_correct_slate(self, kickoff_utc, expected, why):
        assert football_date(kickoff_utc) == expected, why

    def test_differs_from_naive_utc_slicing(self):
        """If these ever agree, the bug has silently returned."""
        kickoff = "2026-09-06T02:30:00+00:00"
        assert kickoff[:10] == "2026-09-06"
        assert football_date(kickoff) == "2026-09-05"

    def test_handles_standard_time_after_dst_ends(self):
        # US DST ends 2026-11-01. A 2026-11-14 23:30Z kickoff is 6:30pm EST.
        assert football_date("2026-11-14T23:30:00+00:00") == "2026-11-14"

    def test_handles_post_midnight_est_kickoff(self):
        assert football_date("2026-11-15T03:30:00+00:00") == "2026-11-14"


class TestWeekdayCrossCheck:
    """The feed's dayOfWeek code matches date.weekday(): Mon=0 ... Sun=6."""

    @pytest.mark.parametrize(
        "kickoff_utc,code",
        [
            ("2026-09-05T16:00:00+00:00", 5),   # Saturday
            ("2026-09-06T02:30:00+00:00", 5),   # still Saturday in Eastern
            ("2026-09-05T00:00:00+00:00", 4),   # Friday
            ("2026-09-06T20:00:00+00:00", 6),   # Sunday
        ],
    )
    def test_agrees_with_feed_code(self, kickoff_utc, code):
        assert weekday_matches_feed(kickoff_utc, code) is True

    def test_detects_disagreement(self):
        assert weekday_matches_feed("2026-09-06T02:30:00+00:00", 6) is False

    def test_missing_code_is_not_a_mismatch(self):
        assert weekday_matches_feed("2026-09-05T16:00:00+00:00", None) is None

    def test_unparseable_code_is_not_a_mismatch(self):
        assert weekday_matches_feed("2026-09-05T16:00:00+00:00", "Saturday") is None


class TestParseEventExposesSlate:
    def test_record_carries_football_date_and_agreement(self, schedule_event):
        schedule_event["scheduledTime"] = "2026-09-06T02:30:00+00:00"
        schedule_event["dayOfWeek"] = 5
        record = parse_event(schedule_event)
        assert record["football_date"] == "2026-09-05"
        assert record["kickoff_utc"][:10] == "2026-09-06"
        assert record["weekday_agrees"] is True
