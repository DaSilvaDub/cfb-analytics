from __future__ import annotations

import pytest

from cfb_analytics.features import market


def price(book, side, american, line=0.0):
    return {"game_id": "evt-1", "market": "ML", "book": book, "side": side,
            "price_american": american, "line": line}


AS_OF = "2026-09-05T20:00:00+00:00"


def build(rows, **kwargs):
    return market.build_consensus("evt-1", "ML", rows, as_of_utc=AS_OF, **kwargs)


class TestAnchorSelection:
    def test_uses_sharp_subset_when_enough_sharp_books_are_priced(self):
        rows = [
            price("PS3838", "HOME", -200), price("PS3838", "AWAY", 180),
            price("CIRCA", "HOME", -205), price("CIRCA", "AWAY", 185),
            price("FANDUEL", "HOME", -260), price("FANDUEL", "AWAY", 210),
        ]
        result = build(rows, sharp_books=["PS3838", "CIRCA"])
        assert result.anchor == "sharp"
        assert result.n_books == 2, "soft books excluded from the anchor"
        assert market.FLAG_NO_SHARP_ANCHOR not in result.flags

    def test_falls_back_to_all_books_when_sharp_books_are_absent(self):
        """The live path today: no sharp book appeared in any 2026-09-01 capture."""
        rows = [
            price("FANDUEL", "HOME", -260), price("FANDUEL", "AWAY", 210),
            price("DRAFTKINGS", "HOME", -255), price("DRAFTKINGS", "AWAY", 205),
            price("CAESARS", "HOME", -250), price("CAESARS", "AWAY", 200),
        ]
        result = build(rows, sharp_books=["PS3838", "CIRCA"])
        assert result.anchor == "all_books"
        assert market.FLAG_NO_SHARP_ANCHOR in result.flags
        assert result.n_books == 3

    def test_one_sharp_book_is_not_enough_to_anchor(self):
        rows = [
            price("PS3838", "HOME", -200), price("PS3838", "AWAY", 180),
            price("FANDUEL", "HOME", -260), price("FANDUEL", "AWAY", 210),
        ]
        result = build(rows, sharp_books=["PS3838", "CIRCA"], min_sharp_books=2)
        assert result.anchor == "all_books"


class TestThinMarketGuards:
    def test_flags_a_market_below_the_consensus_floor(self):
        rows = [price("MIDNITE", "HOME", -260), price("MIDNITE", "AWAY", 210)]
        result = build(rows, min_books_for_consensus=3)
        assert market.FLAG_THIN_MARKET in result.flags
        assert market.FLAG_SINGLE_BOOK in result.flags

    def test_one_sided_market_yields_no_probabilities(self):
        """A fair probability needs both sides; one side cannot be devigged."""
        rows = [price("FANDUEL", "HOME", -260), price("DRAFTKINGS", "HOME", -255)]
        result = build(rows)
        assert result.anchor == "none"
        assert result.sides == ()

    def test_no_priced_rows_returns_none(self):
        assert build([{"book": "X", "side": "HOME", "price_american": None}]) is None


class TestConsensusMath:
    @pytest.fixture
    def result(self):
        rows = [
            price("FANDUEL", "HOME", -260), price("FANDUEL", "AWAY", 210),
            price("DRAFTKINGS", "HOME", -250), price("DRAFTKINGS", "AWAY", 205),
            price("CAESARS", "HOME", -270), price("CAESARS", "AWAY", 200),
        ]
        return build(rows)

    def test_probabilities_sum_to_one(self, result):
        total = sum(s.probs["multiplicative"] for s in result.sides)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_favorite_probability_is_below_the_vig_inclusive_quote(self, result):
        home = result.side("HOME")
        assert home.vig_free_prob < 0.72, "devig must shade the quoted 72%"
        assert home.vig_free_prob > 0.66

    def test_hold_is_positive_for_a_normal_book(self, result):
        assert 0.0 < result.hold < 0.10

    def test_best_price_is_the_lowest_implied_probability(self, result):
        home = result.side("HOME")
        # -250 is the cheapest way to back the favourite among -250/-260/-270.
        assert home.best_price == -250
        assert home.best_book == "DRAFTKINGS"

    def test_stores_all_three_devig_methods(self, result):
        assert set(result.side("HOME").probs) == {"multiplicative", "shin", "power"}

    def test_reports_the_spread_between_methods(self, result):
        spread = market.summarise_probability_spread(result, "HOME")
        assert spread is not None and spread >= 0.0


class TestMedianInProbabilitySpace:
    """Median of American odds directly is meaningless: -110 and +110 are
    adjacent in probability but 220 apart on the American scale."""

    def test_median_straddling_the_plus_minus_100_discontinuity(self):
        rows = [
            price("A", "HOME", -110), price("A", "AWAY", -110),
            price("B", "HOME", 110), price("B", "AWAY", -130),
            price("C", "HOME", -105), price("C", "AWAY", -115),
        ]
        result = build(rows)
        home = result.side("HOME")
        # Naive arithmetic median of (-110, 110, -105) would be -105;
        # in probability space the middle value is the one near even money.
        assert -120 < home.consensus_price < -100

    def test_arbitrage_means_one_book_pricing_both_sides_under_100(self):
        """Real negative hold is a single book's two-sided quote summing < 1."""
        rows = [price("A", "HOME", 110), price("A", "AWAY", 110)]
        result = build(rows, min_books_for_consensus=1)
        assert market.FLAG_ARBITRAGE in result.flags
        assert result.hold < 0

    def test_two_one_sided_books_are_not_arbitrage(self):
        """Different books quoting opposite sides is NOT a negative hold.
        Treating it as one produced 275 phantom arbs on the 2026-09-05 slate."""
        rows = [price("A", "HOME", 110), price("B", "AWAY", 110)]
        result = build(rows, min_books_for_consensus=1)
        assert market.FLAG_ARBITRAGE not in result.flags
        assert market.FLAG_ONE_SIDED in result.flags
        assert result.sides == ()


class TestPerBookDevig:
    """Each book is devigged against itself, then fair probabilities aggregate.

    Regression for BALL@OSU (2026-09-05): HOME priced by 2 books, AWAY by 4.
    Median-then-devig mixed the book sets and produced a -18249 consensus on a
    market whose best real price was -10000.
    """

    def test_asymmetric_book_coverage_does_not_inflate_the_favorite(self):
        rows = [
            # Both sides from both books -> a coherent two-sided market.
            price("FANATICS", "HOME", -10000), price("FANATICS", "AWAY", 2800),
            price("MIDNITE", "HOME", -8000), price("MIDNITE", "AWAY", 5000),
            # Extra one-sided dog quotes, as the live feed had.
            price("KALSHI", "AWAY", 9251), price("PROPHETX", "AWAY", 10000),
        ]
        result = build(rows)
        home = result.side("HOME")
        assert home.vig_free_prob < 0.995
        # The one-sided dog books must not drag the favourite upward.
        assert home.consensus_price is not None and home.consensus_price > -20000

    def test_books_quoting_one_side_are_excluded_from_fair_probability(self):
        rows = [
            price("A", "HOME", -200), price("A", "AWAY", 170),
            price("B", "HOME", -210), price("B", "AWAY", 175),
            price("C", "AWAY", 900),  # one-sided outlier
        ]
        result = build(rows)
        assert market.FLAG_PARTIAL_BOOKS in result.flags
        away = result.side("AWAY")
        # C's +900 would badly distort a cross-book median.
        assert 0.30 < away.vig_free_prob < 0.42

    def test_one_sided_book_still_counts_for_best_price(self):
        """A one-sided quote is not usable for devig but is still bettable."""
        rows = [
            price("A", "HOME", -200), price("A", "AWAY", 170),
            price("B", "HOME", -210), price("B", "AWAY", 175),
            price("C", "AWAY", 900),
        ]
        result = build(rows)
        assert result.side("AWAY").best_price == 900
        assert result.side("AWAY").best_book == "C"


class TestPlaceholderPrices:
    """Books post -100000 to mean 'no action', not '99.9% likely'."""

    def test_drops_placeholder_quotes_and_flags_it(self):
        rows = [
            price("MIDNITE", "HOME", -100000), price("MIDNITE", "AWAY", 5000),
            price("FANATICS", "HOME", -10000), price("FANATICS", "AWAY", 2800),
            price("FLIFF", "HOME", -8000), price("FLIFF", "AWAY", 4000),
        ]
        result = build(rows)
        assert market.FLAG_PLACEHOLDER_DROPPED in result.flags
        assert result.side("HOME").vig_free_prob < 0.99

    def test_a_market_of_only_placeholders_yields_no_fair_probability(self):
        """Dropping the -100000 leaves a one-sided book, so no fair probability
        can be formed. The result is a flagged, side-less consensus rather than
        None -- 'quotes existed but were unusable' is not 'no quotes at all'."""
        rows = [price("A", "HOME", -100000), price("A", "AWAY", 60000)]
        result = build(rows, min_books_for_consensus=1)
        assert result.sides == ()
        assert result.anchor == "none"
        assert market.FLAG_PLACEHOLDER_DROPPED in result.flags

    def test_no_quotes_at_all_returns_none(self):
        assert build([{"book": "A", "side": "HOME", "price_american": None}]) is None

    def test_credible_heavy_favorite_survives_the_filter(self):
        rows = [
            price("A", "HOME", -2000), price("A", "AWAY", 1200),
            price("B", "HOME", -2100), price("B", "AWAY", 1250),
        ]
        result = build(rows, min_books_for_consensus=1)
        assert market.FLAG_PLACEHOLDER_DROPPED not in result.flags
        assert 0.92 < result.side("HOME").vig_free_prob < 0.96


class TestGrouping:
    def test_groups_by_game_market_and_line(self):
        assert market.group_key(price("A", "OVER", -110, line=55.5)) == ("evt-1", "ML", 55.5)

    def test_different_lines_are_different_markets(self):
        a = market.group_key(price("A", "OVER", -110, line=55.5))
        b = market.group_key(price("A", "OVER", -110, line=52.5))
        assert a != b, "devigging -3.5 together with -7.5 would be nonsense"


class TestLineMovement:
    def test_reports_magnitude_and_direction(self):
        move = market.line_movement(
            {"line": -3.0, "price_american": -110},
            {"line": -6.5, "price_american": -110},
        )
        assert move["move_magnitude"] == pytest.approx(-3.5)
        assert move["move_direction"] == "toward"

    def test_rlm_is_always_labelled_line_only(self):
        """True RLM needs ticket/money percentages, which no free source has."""
        move = market.line_movement(
            {"line": -3.0, "price_american": -110},
            {"line": -6.5, "price_american": -105},
        )
        assert move["rlm_basis"] == "line_only"

    def test_missing_endpoint_yields_no_movement_rather_than_a_guess(self):
        move = market.line_movement(None, {"line": -6.5, "price_american": -110})
        assert move["move_magnitude"] is None
        assert move["rlm_flag"] is False

    def test_flat_line_is_neither_toward_nor_away(self):
        move = market.line_movement(
            {"line": -3.0, "price_american": -110},
            {"line": -3.0, "price_american": -115},
        )
        assert move["move_direction"] == "flat"
        assert move["rlm_flag"] is False


class TestStaleness:
    def test_flags_prices_older_than_the_threshold(self):
        assert market.staleness_flag(400, 360) == [market.FLAG_STALE_PRICES]

    def test_fresh_prices_are_unflagged(self):
        assert market.staleness_flag(60, 360) == []


class TestSpreadSidesPairUp:
    """HOME -13.5 and AWAY +13.5 are the same market and must group together.

    Keying on the raw line left them unpaired, which produced 694 phantom
    negative-hold rows out of 1594 spreads while ML and TOTAL showed zero.
    """

    def test_opposite_signed_spread_lines_share_a_group(self):
        home = {"game_id": "g", "market": "SPREAD", "line": -13.5}
        away = {"game_id": "g", "market": "SPREAD", "line": 13.5}
        assert market.group_key(home) == market.group_key(away)

    def test_different_spread_rungs_stay_separate(self):
        a = {"game_id": "g", "market": "SPREAD", "line": -13.5}
        b = {"game_id": "g", "market": "SPREAD", "line": -16.5}
        assert market.group_key(a) != market.group_key(b)

    def test_totals_are_unaffected(self):
        over = {"game_id": "g", "market": "TOTAL", "line": 55.5}
        under = {"game_id": "g", "market": "TOTAL", "line": 55.5}
        assert market.group_key(over) == market.group_key(under)

    def test_a_paired_spread_has_normal_positive_hold(self):
        rows = [
            {"game_id": "evt-1", "market": "SPREAD", "book": "A", "side": "HOME",
             "price_american": -110, "line": -13.5},
            {"game_id": "evt-1", "market": "SPREAD", "book": "A", "side": "AWAY",
             "price_american": -110, "line": 13.5},
        ]
        result = market.build_consensus(
            "evt-1", "SPREAD", rows, as_of_utc=AS_OF, min_books_for_consensus=1)
        assert result.hold > 0, "a paired -110/-110 spread must not look arbed"
        assert market.FLAG_ARBITRAGE not in result.flags

    def test_each_side_keeps_its_own_signed_line(self):
        rows = [
            {"game_id": "evt-1", "market": "SPREAD", "book": "A", "side": "HOME",
             "price_american": -110, "line": -13.5},
            {"game_id": "evt-1", "market": "SPREAD", "book": "A", "side": "AWAY",
             "price_american": -110, "line": 13.5},
        ]
        result = market.build_consensus(
            "evt-1", "SPREAD", rows, as_of_utc=AS_OF, min_books_for_consensus=1)
        assert result.side("HOME").line == -13.5
        assert result.side("AWAY").line == 13.5
