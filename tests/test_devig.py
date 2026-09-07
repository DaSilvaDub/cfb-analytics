from __future__ import annotations

import math

import pytest

from cfb_analytics.errors import DevigError
from cfb_analytics.models import devig
from cfb_analytics.utils import implied_probability


def probs(*american: int) -> list[float]:
    return [implied_probability(a) for a in american]


class TestOverroundAndHold:
    def test_standard_minus_110_pair_holds_about_45_basis_points(self):
        assert devig.hold(probs(-110, -110)) == pytest.approx(0.0454, abs=1e-4)

    def test_fair_market_has_zero_hold(self):
        assert devig.hold([0.5, 0.5]) == pytest.approx(0.0)

    def test_arbitrage_market_has_negative_hold(self):
        assert devig.hold([0.48, 0.48]) < 0


class TestMultiplicative:
    def test_normalises_to_one(self):
        assert math.fsum(devig.multiplicative(probs(-110, -110))) == pytest.approx(1.0)

    def test_symmetric_market_gives_even_probabilities(self):
        assert devig.multiplicative(probs(-110, -110)) == pytest.approx([0.5, 0.5])

    def test_heavy_favorite(self):
        # -2000 / +1200 -> favourite fair probability just under the quoted 95.2%.
        fair = devig.multiplicative(probs(-2000, 1200))
        assert fair[0] == pytest.approx(0.9253, abs=1e-3)


class TestAllMethodsAgreeOnStructure:
    @pytest.mark.parametrize("method", devig.METHODS)
    def test_sums_to_one(self, method):
        fair = devig.devig(probs(-2000, 1200), method)
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("method", devig.METHODS)
    def test_preserves_ordering(self, method):
        fair = devig.devig(probs(-2000, 1200), method)
        assert fair[0] > fair[1]

    @pytest.mark.parametrize("method", devig.METHODS)
    def test_fair_market_is_a_fixed_point(self, method):
        assert devig.devig([0.5, 0.5], method) == pytest.approx([0.5, 0.5], abs=1e-6)


class TestMethodsDisagreeWhereItMatters:
    """The whole reason all three are stored: they differ on heavy favourites."""

    def test_favorite_probability_spread_is_material(self):
        """Measured on -2000/+1200: 0.9253 / 0.9377 / 0.9460 -- a 2.1pp spread.
        That is the difference between clearing a 93% CORE gate and missing it."""
        fair = {m: devig.devig(probs(-2000, 1200), m)[0] for m in devig.METHODS}
        spread = max(fair.values()) - min(fair.values())
        assert spread > 0.015, f"expected a material spread, got {fair}"

    def test_shin_and_power_shade_the_favorite_above_multiplicative(self):
        """Books load margin onto longshots, so proportional devig *understates*
        the favourite. Shin and power take more margin off the dog, which raises
        the favourite's fair probability. Ordering: mult < shin < power."""
        fair = {m: devig.devig(probs(-2000, 1200), m)[0] for m in devig.METHODS}
        assert fair["multiplicative"] < fair["shin"] < fair["power"]

    def test_the_dog_moves_in_the_opposite_direction(self):
        dog = {m: devig.devig(probs(-2000, 1200), m)[1] for m in devig.METHODS}
        assert dog["multiplicative"] > dog["shin"] > dog["power"]

    def test_disagreement_grows_with_how_lopsided_the_market_is(self):
        """The reason all three are stored: they agree on coin-flips and
        diverge exactly where the parlay product shops."""
        def spread(*american):
            fair = [devig.devig(probs(*american), m)[0] for m in devig.METHODS]
            return max(fair) - min(fair)

        assert spread(-2000, 1200) > spread(-150, 130) > spread(-110, -110)

    def test_spread_is_negligible_on_a_balanced_market(self):
        fair = {m: devig.devig(probs(-110, -110), m)[0] for m in devig.METHODS}
        assert max(fair.values()) - min(fair.values()) < 1e-6


class TestShin:
    def test_returns_input_when_market_is_already_fair(self):
        assert devig.shin([0.5, 0.5]) == pytest.approx([0.5, 0.5])

    def test_falls_back_when_book_is_arbitraged(self):
        # Negative margin has no informed-money interpretation.
        result = devig.shin([0.48, 0.48])
        assert result == pytest.approx([0.5, 0.5])

    def test_handles_extreme_favorite_without_blowing_up(self):
        fair = devig.shin(probs(-10000, 4000))
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert 0.95 < fair[0] < 1.0

    def test_handles_three_way_market(self):
        fair = devig.shin([0.40, 0.35, 0.30])
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert len(fair) == 3
        assert fair[0] > fair[1] > fair[2]

    def test_handles_many_outcomes_market(self):
        quotes = [0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01]
        fair = devig.shin(quotes)
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert len(fair) == len(quotes)
        assert all(p > 0.0 for p in fair)


class TestPower:
    def test_solves_exponent_so_probabilities_sum_to_one(self):
        assert math.fsum(devig.power(probs(-250, 200))) == pytest.approx(1.0, abs=1e-9)

    def test_handles_three_way_market(self):
        fair = devig.power([0.40, 0.35, 0.30])
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert len(fair) == 3

    def test_fair_market_is_fixed_point(self):
        assert devig.power([0.5, 0.5]) == pytest.approx([0.5, 0.5])

    def test_negative_hold_arbitrage_handled(self):
        fair = devig.power([0.48, 0.48])
        assert fair == pytest.approx([0.5, 0.5])
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)

    def test_handles_many_outcomes_market(self):
        quotes = [0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01]
        fair = devig.power(quotes)
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert len(fair) == len(quotes)
        assert all(p > 0.0 for p in fair)


class TestValidation:
    def test_none_is_refused(self):
        with pytest.raises(DevigError, match="cannot be None"):
            devig.multiplicative(None)  # type: ignore[arg-type]

    def test_single_outcome_is_refused(self):
        with pytest.raises(DevigError, match="at least two"):
            devig.multiplicative([0.9])

    @pytest.mark.parametrize("bad", [[0.0, 0.5], [-0.1, 0.6]])
    def test_non_positive_quote_is_refused(self, bad):
        with pytest.raises(DevigError, match="positive"):
            devig.multiplicative(bad)

    def test_quote_at_certainty_is_refused_rather_than_devigged(self):
        """A quote >= 1.0 means a misparse; emitting a confident number would
        launder a parsing bug into a probability."""
        with pytest.raises(DevigError, match="not a real price"):
            devig.multiplicative([1.0, 0.05])

    @pytest.mark.parametrize("bad", [
        [float("nan"), 0.5],
        [float("inf"), 0.5],
        [float("-inf"), 0.5],
    ])
    def test_non_finite_quotes_are_refused(self, bad):
        with pytest.raises(DevigError, match="finite numbers"):
            devig.multiplicative(bad)

    def test_unconvertible_types_are_refused(self):
        with pytest.raises(DevigError, match="convertible to float"):
            devig.multiplicative(["not_a_number", 0.5])  # type: ignore[list-item]

    def test_unknown_method_is_refused(self):
        with pytest.raises(DevigError, match="Unknown devig method"):
            devig.devig([0.5, 0.5], "vibes")

    def test_non_string_method_is_refused(self):
        with pytest.raises(DevigError, match="must be a string"):
            devig.devig([0.5, 0.5], 123)  # type: ignore[arg-type]


class TestDevigAll:
    def test_returns_every_method(self):
        result = devig.devig_all(probs(-150, 130))
        assert set(result) == set(devig.METHODS)
        for values in result.values():
            assert math.fsum(values) == pytest.approx(1.0, abs=1e-9)

    def test_allow_failures_omits_failing_methods(self):
        # Extreme longshot causes additive to fail. additive is not in the
        # default METHODS (see devig.py module docstring: it duplicates shin
        # for two-outcome markets), so it must be requested explicitly via
        # CANONICAL_METHODS to exercise this failure path.
        quoted = [0.85, 0.20, 0.01]
        with pytest.raises(DevigError):
            devig.devig_all(quoted, methods=devig.CANONICAL_METHODS, allow_failures=False)

        res = devig.devig_all(quoted, methods=devig.CANONICAL_METHODS, allow_failures=True)
        assert "additive" not in res
        assert "multiplicative" in res
        assert "shin" in res
        assert "power" in res
        assert "odds_ratio" in res

    def test_default_methods_excludes_additive(self):
        # additive is available but not scored by default: it is a duplicate
        # of shin for every two-outcome market this pipeline builds.
        assert "additive" not in devig.METHODS
        assert "additive" in devig.CANONICAL_METHODS


class TestAdditive:
    def test_normalises_to_one(self):
        assert math.fsum(devig.additive(probs(-110, -110))) == pytest.approx(1.0)

    def test_symmetric_market_gives_even_probabilities(self):
        assert devig.additive(probs(-110, -110)) == pytest.approx([0.5, 0.5])

    def test_fair_market_is_fixed_point(self):
        assert devig.additive([0.5, 0.5]) == pytest.approx([0.5, 0.5])

    def test_heavy_favorite(self):
        fair = devig.additive(probs(-2000, 1200))
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert fair[0] == pytest.approx(0.9377, abs=1e-3)
        assert fair[1] == pytest.approx(0.0623, abs=1e-3)

    def test_three_way_market(self):
        fair = devig.additive([0.45, 0.35, 0.26])
        assert fair == pytest.approx([0.43, 0.33, 0.24], abs=1e-6)
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)

    def test_negative_hold_arbitrage_handled(self):
        fair = devig.additive([0.48, 0.48])
        assert fair == pytest.approx([0.5, 0.5])

    def test_margin_exceeding_quote_raises_devig_error(self):
        with pytest.raises(DevigError, match="non-positive probability"):
            devig.additive([0.85, 0.20, 0.01])


class TestOddsRatio:
    def test_normalises_to_one(self):
        assert math.fsum(devig.odds_ratio(probs(-110, -110))) == pytest.approx(1.0)

    def test_symmetric_market_gives_even_probabilities(self):
        assert devig.odds_ratio(probs(-110, -110)) == pytest.approx([0.5, 0.5])

    def test_fair_market_is_fixed_point(self):
        assert devig.odds_ratio([0.5, 0.5]) == pytest.approx([0.5, 0.5])

    def test_heavy_favorite(self):
        fair = devig.odds_ratio(probs(-2000, 1200))
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert 0.935 < fair[0] < 0.945
        assert fair[0] > fair[1]

    def test_three_way_market(self):
        fair = devig.odds_ratio([0.40, 0.35, 0.30])
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert len(fair) == 3
        assert fair[0] > fair[1] > fair[2]

    def test_negative_hold_arbitrage_handled(self):
        fair = devig.odds_ratio([0.48, 0.48])
        assert fair == pytest.approx([0.5, 0.5])

    def test_extreme_favorite_converges(self):
        fair = devig.odds_ratio(probs(-10000, 4000))
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert 0.98 < fair[0] < 1.0

    def test_handles_many_outcomes_market(self):
        quotes = [0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01]
        fair = devig.odds_ratio(quotes)
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert len(fair) == len(quotes)
        assert all(p > 0.0 for p in fair)


class TestPerBookDeviggingAndAggregation:
    def test_devig_per_book_computes_each_book_independently(self):
        quotes = {
            "PINNACLE": probs(-200, 170),
            "DRAFTKINGS": probs(-210, 175),
        }
        res = devig.devig_per_book(quotes, method="multiplicative")
        assert set(res) == {"PINNACLE", "DRAFTKINGS"}
        for book_fair in res.values():
            assert math.fsum(book_fair) == pytest.approx(1.0, abs=1e-9)
            assert book_fair[0] > book_fair[1]

    def test_devig_per_book_skips_malformed_books(self):
        quotes = {
            "GOOD": probs(-110, -110),
            "BAD_PROB": [1.5, 0.2],
            "BAD_LEN": [0.5],
        }
        res = devig.devig_per_book(quotes)
        assert set(res) == {"GOOD"}
        assert res["GOOD"] == pytest.approx([0.5, 0.5])

    def test_devig_per_book_empty_input(self):
        assert devig.devig_per_book({}) == {}

    def test_devig_per_book_all_computes_all_methods(self):
        quotes = {"BOOK_A": probs(-150, 130)}
        res = devig.devig_per_book_all(quotes)
        assert "BOOK_A" in res
        assert set(res["BOOK_A"]) == set(devig.METHODS)

    def test_aggregate_probabilities_median(self):
        book_probs = [
            [0.60, 0.40],
            [0.62, 0.38],
            [0.59, 0.41],
        ]
        agg = devig.aggregate_probabilities(book_probs, aggregator="median")
        assert math.fsum(agg) == pytest.approx(1.0, abs=1e-9)
        assert agg[0] == pytest.approx(0.60)
        assert agg[1] == pytest.approx(0.40)

    def test_aggregate_probabilities_mean(self):
        book_probs = [
            [0.60, 0.40],
            [0.70, 0.30],
        ]
        agg = devig.aggregate_probabilities(book_probs, aggregator="mean")
        assert math.fsum(agg) == pytest.approx(1.0, abs=1e-9)
        assert agg[0] == pytest.approx(0.65)
        assert agg[1] == pytest.approx(0.35)

    def test_aggregate_probabilities_empty_raises(self):
        with pytest.raises(DevigError, match="empty set"):
            devig.aggregate_probabilities([])

    def test_aggregate_probabilities_mismatched_dimensions_raises(self):
        with pytest.raises(DevigError, match="same number of outcomes"):
            devig.aggregate_probabilities([[0.5, 0.5], [0.4, 0.3, 0.3]])

    def test_aggregate_probabilities_unknown_aggregator_raises(self):
        with pytest.raises(DevigError, match="Unknown aggregator"):
            devig.aggregate_probabilities([[0.5, 0.5]], aggregator="mode")

    def test_consensus_probabilities_end_to_end_with_mapping(self):
        quotes = {
            "A": probs(-150, 130),
            "B": probs(-145, 125),
            "C": probs(-155, 135),
        }
        fair = devig.consensus_probabilities(quotes, method="shin")
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert 0.57 < fair[0] < 0.61

    def test_consensus_probabilities_end_to_end_with_sequence(self):
        quotes = [probs(-150, 130), probs(-145, 125), probs(-155, 135)]
        fair = devig.consensus_probabilities(quotes, method="power")
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)

    def test_consensus_probabilities_all_methods(self):
        quotes = [probs(-150, 130), probs(-145, 125)]
        consensus = devig.consensus_probabilities_all(quotes)
        assert set(consensus) == set(devig.METHODS)
        for fair in consensus.values():
            assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)

    def test_consensus_per_book_avoids_asymmetric_coverage_bias(self):
        """Per-book devigging prevents asymmetric book coverage distortion.

        If Book A quotes -10000 / +2800 and Book B quotes -8000 / +5000,
        devigging each book independently keeps the favourite fair probability
        anchored in reality (< 0.995), avoiding cross-book median distortion.
        """
        quotes = {
            "A": probs(-10000, 2800),
            "B": probs(-8000, 5000),
        }
        fair = devig.consensus_probabilities(quotes, method="multiplicative")
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert fair[0] < 0.995
        assert fair[1] > 0.005

    def test_method_disagreement_spread(self):
        probs_by_method = {
            "multiplicative": [0.925, 0.075],
            "shin": [0.938, 0.062],
            "power": [0.946, 0.054],
        }
        disagreement = devig.method_disagreement(probs_by_method)
        assert len(disagreement) == 2
        assert disagreement[0] == pytest.approx(0.021, abs=1e-3)
        assert disagreement[1] == pytest.approx(0.021, abs=1e-3)

    def test_method_disagreement_single_method_returns_zeros(self):
        disagreement = devig.method_disagreement({"multiplicative": [0.6, 0.4]})
        assert disagreement == [0.0, 0.0]

    def test_method_disagreement_empty_returns_empty(self):
        assert devig.method_disagreement({}) == []

    def test_method_disagreement_mismatched_lengths_raises(self):
        with pytest.raises(DevigError, match="identical outcome counts"):
            devig.method_disagreement({
                "m1": [0.6, 0.4],
                "m2": [0.5, 0.3, 0.2],
            })

    def test_method_disagreement_non_finite_raises(self):
        with pytest.raises(DevigError, match="finite numbers"):
            devig.method_disagreement({
                "m1": [float("nan"), 0.4],
                "m2": [0.6, 0.4],
            })

    def test_method_aliases(self):
        for alias, canonical in devig.METHOD_ALIASES.items():
            assert devig.devig([0.6, 0.45], alias) == devig.devig([0.6, 0.45], canonical)
            # Test case-insensitivity and whitespace tolerance
            assert devig.devig([0.6, 0.45], alias.upper()) == devig.devig([0.6, 0.45], canonical)

    def test_devig_per_book_none_input_raises(self):
        with pytest.raises(DevigError, match="cannot be None"):
            devig.devig_per_book(None)  # type: ignore[arg-type]

    def test_devig_per_book_all_none_input_raises(self):
        with pytest.raises(DevigError, match="cannot be None"):
            devig.devig_per_book_all(None)  # type: ignore[arg-type]

    def test_consensus_probabilities_none_input_raises(self):
        with pytest.raises(DevigError, match="cannot be None"):
            devig.consensus_probabilities(None)  # type: ignore[arg-type]

    def test_consensus_probabilities_no_valid_books_raises(self):
        with pytest.raises(DevigError, match="No valid book quotes"):
            devig.consensus_probabilities({"B1": [1.5, 0.5], "B2": [-0.1, 0.5]})

    def test_aggregate_probabilities_invalid_values_raise(self):
        with pytest.raises(DevigError, match="finite numbers in"):
            devig.aggregate_probabilities([[-0.1, 1.1], [0.5, 0.5]])

        with pytest.raises(DevigError, match="finite numbers in"):
            devig.aggregate_probabilities([[float("nan"), 0.5], [0.5, 0.5]])

    def test_aggregate_probabilities_case_insensitive_aggregator(self):
        book_probs = [[0.60, 0.40], [0.62, 0.38]]
        agg = devig.aggregate_probabilities(book_probs, aggregator="  MEDIAN  ")
        assert math.fsum(agg) == pytest.approx(1.0, abs=1e-9)
        assert agg[0] == pytest.approx(0.61)
