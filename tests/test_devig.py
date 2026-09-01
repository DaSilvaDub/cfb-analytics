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


class TestPower:
    def test_solves_exponent_so_probabilities_sum_to_one(self):
        assert math.fsum(devig.power(probs(-250, 200))) == pytest.approx(1.0, abs=1e-9)

    def test_handles_three_way_market(self):
        fair = devig.power([0.40, 0.35, 0.30])
        assert math.fsum(fair) == pytest.approx(1.0, abs=1e-9)
        assert len(fair) == 3


class TestValidation:
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

    def test_unknown_method_is_refused(self):
        with pytest.raises(DevigError, match="Unknown devig method"):
            devig.devig([0.5, 0.5], "vibes")


class TestDevigAll:
    def test_returns_every_method(self):
        result = devig.devig_all(probs(-150, 130))
        assert set(result) == set(devig.METHODS)
        for values in result.values():
            assert math.fsum(values) == pytest.approx(1.0, abs=1e-9)
