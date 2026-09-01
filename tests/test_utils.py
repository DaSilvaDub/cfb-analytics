from __future__ import annotations

import pytest

from cfb_analytics.utils import (
    american_to_decimal,
    decimal_to_american,
    implied_probability,
    stable_id,
    to_utc_iso,
)


class TestAmericanToDecimal:
    def test_converts_favorite_price(self):
        assert american_to_decimal(-260) == pytest.approx(1.3846153846, abs=1e-9)

    def test_converts_underdog_price(self):
        assert american_to_decimal(190) == pytest.approx(2.9)

    def test_accepts_string_price_from_feed(self):
        assert american_to_decimal("-265") == pytest.approx(1.377358, abs=1e-5)

    @pytest.mark.parametrize("bad", [None, "", 0, "abc", float("nan")])
    def test_returns_none_rather_than_defaulting(self, bad):
        assert american_to_decimal(bad) is None

    def test_round_trips_through_american(self):
        for price in (-2000, -260, -110, 145, 190, 5000):
            assert decimal_to_american(american_to_decimal(price)) == price


class TestImpliedProbability:
    def test_heavy_favorite_is_vig_inclusive(self):
        # -2000 implies 95.24% BEFORE devig; the vig-free number is lower.
        assert implied_probability(-2000) == pytest.approx(0.952381, abs=1e-6)

    def test_two_sided_book_overrounds(self):
        total = implied_probability(-110) + implied_probability(-110)
        assert total > 1.0


class TestToUtcIso:
    def test_parses_outlier_offset_without_colon(self):
        assert to_utc_iso("2026-09-05T19:30:00-0700") == "2026-09-06T02:30:00+00:00"

    def test_parses_offset_with_colon(self):
        assert to_utc_iso("2026-09-05T19:30:00-07:00") == "2026-09-06T02:30:00+00:00"

    def test_parses_zulu(self):
        assert to_utc_iso("2026-09-05T19:30:00Z") == "2026-09-05T19:30:00+00:00"

    def test_parses_fractional_offset_form(self):
        assert to_utc_iso("2026-08-31T13:41:41.113000-0700") == "2026-08-31T20:41:41+00:00"

    @pytest.mark.parametrize("bad", [None, "", "not-a-time"])
    def test_unparseable_returns_none_so_caller_decides(self, bad):
        assert to_utc_iso(bad) is None


class TestStableId:
    def test_is_deterministic(self):
        assert stable_id("a", 1, None) == stable_id("a", 1, None)

    def test_distinguishes_field_boundaries(self):
        # "ab" + "c" must not collide with "a" + "bc".
        assert stable_id("ab", "c") != stable_id("a", "bc")

    def test_keeps_full_digest(self):
        assert len(stable_id("x")) == 64
