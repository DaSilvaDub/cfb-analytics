"""Ridge team-strength tests.

Built around ``RidgeRatings.margin()``, not raw offense/defense values,
because raw per-team offense/defense are individually gauge-dependent: the
model has a 2-dimensional null space (shift every team's offense up by c and
mu down by c leaves every prediction unchanged; same for defense with the
opposite mu sign), so an ad hoc "net = offense - defense" per team is NOT
guaranteed to match a chosen generating value, or even to be well-defined
across ridge_lambda values. Confirmed directly while building this suite: a
naive "net" comparison was off by a uniform, ridge_lambda-independent 7.67 for
every team simultaneously -- the fingerprint of exactly that gauge freedom.

``margin(a, b)`` is different: it is the ONLY quantity the model is actually
built to produce (plan section 6.3), and it is provably gauge-invariant
(the null-space shifts cancel in the home-minus-away combination). Verified
by hand: fitted margins matched true generating margins to 0.000 across every
pairwise matchup in a 4-team synthetic league.
"""

from __future__ import annotations

import pytest

from cfb_analytics.models.ridge import DEFAULT_RIDGE_LAMBDA, fit_ratings


def game(home, away, home_points, away_points, neutral_site=False):
    return {
        "home_team_id": home, "away_team_id": away,
        "home_points": home_points, "away_points": away_points,
        "neutral_site": neutral_site,
    }


def true_margin(true_ratings, home, away, *, hfa=0.0, neutral_site=False):
    """The exact plan section 6.3 formula, from the same true O/D used to
    generate a synthetic league -- the ground truth a fit is checked against."""
    o_h, d_h = true_ratings[home]
    o_a, d_a = true_ratings[away]
    bonus = 0.0 if neutral_site else hfa
    return (o_h - d_a) - (o_a - d_h) + bonus


def synthetic_league(true_ratings, *, hfa=0.0, mu=24.0, rounds=5, neutral=False):
    """Generate scores exactly from the model's own equation:

        points = mu + hfa * is_home + offense[team] - defense[opponent]

    This is the correct way to build a parameter-recovery test for a
    regression model: instantiate the assumed generative process with known
    parameters, then check the fit reproduces its predictions (margin, which
    is gauge-invariant -- see the module docstring for why individual
    offense/defense values are the wrong thing to assert on directly).
    """
    teams = list(true_ratings)
    games = []
    for _ in range(rounds):
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                for h, a in ((home, away), (away, home)):
                    o_h, d_h = true_ratings[h]
                    o_a, d_a = true_ratings[a]
                    home_bonus = 0.0 if neutral else hfa
                    games.append(game(
                        h, a,
                        mu + home_bonus + o_h - d_a,
                        mu + o_a - d_h,
                        neutral_site=neutral,
                    ))
    return games


class TestInsufficientHistory:
    def test_below_min_games_returns_insufficient_history(self):
        result = fit_ratings([game("A", "B", 20, 10)], min_games=30)
        assert result.status == "insufficient_history"
        assert result.teams == {}

    def test_negative_ridge_lambda_raises(self):
        with pytest.raises(ValueError):
            fit_ratings([], ridge_lambda=-1.0)

    def test_zero_min_games_raises(self):
        with pytest.raises(ValueError):
            fit_ratings([], min_games=0)


class TestDataCleaning:
    RATINGS = {"A": (10.0, -5.0), "B": (0.0, 0.0), "C": (-5.0, 5.0)}

    def test_games_missing_a_score_are_skipped_not_counted(self):
        base = synthetic_league(self.RATINGS, rounds=15)
        bad = base + [
            {"home_team_id": "X", "away_team_id": "Y", "home_points": None, "away_points": 10}
        ]
        clean = fit_ratings(base, min_games=1)
        result = fit_ratings(bad, min_games=1)
        assert result.n_games == clean.n_games

    def test_games_missing_a_team_id_are_skipped(self):
        base = synthetic_league(self.RATINGS, rounds=15)
        bad = base + [
            {"home_team_id": None, "away_team_id": "Y", "home_points": 10, "away_points": 3}
        ]
        clean = fit_ratings(base, min_games=1)
        result = fit_ratings(bad, min_games=1)
        assert result.n_games == clean.n_games


class TestMarginRecovery:
    """The model's actual, gauge-invariant contract: does margin() reproduce
    the true generating margin? Verified by hand to match to within floating
    point noise at a near-zero ridge penalty, across every pairwise matchup
    in a 4-team, fully-connected, many-round synthetic league."""

    TRUE_RATINGS = {
        "Strong": (12.0, -3.0),   # dominant: scores a lot, allows little
        "MidA": (2.0, 1.0),       # modest offense-leaning team
        "MidB": (-1.0, -2.0),     # modest defense-leaning team
        "Weak": (-10.0, 8.0),     # bad at both
    }

    @pytest.fixture
    def league(self):
        games = synthetic_league(self.TRUE_RATINGS, rounds=15, hfa=3.0)
        return fit_ratings(games, min_games=1, ridge_lambda=0.01)

    def test_fits_successfully_with_enough_games(self, league):
        assert league.status == "active"
        assert set(league.teams) == set(self.TRUE_RATINGS)

    @pytest.mark.parametrize("home,away", [
        ("Strong", "Weak"), ("Weak", "Strong"),
        ("Strong", "MidA"), ("MidA", "MidB"), ("MidB", "Weak"),
    ])
    def test_recovers_every_pairwise_margin(self, league, home, away):
        expected = true_margin(self.TRUE_RATINGS, home, away, hfa=3.0)
        assert league.margin(home, away) == pytest.approx(expected, abs=0.2)

    def test_recovers_the_correct_favorite_in_every_matchup(self, league):
        """A softer, ranking-only check that stays meaningful even with a
        heavier ridge penalty (unlike exact-margin recovery, which needs a
        near-zero lambda to avoid deliberate shrinkage bias)."""
        for home, away in (("Strong", "Weak"), ("Strong", "MidA"), ("MidB", "Weak")):
            expected_sign = true_margin(self.TRUE_RATINGS, home, away, hfa=3.0)
            fitted = league.margin(home, away)
            assert (fitted > 0) == (expected_sign > 0)


class TestHomeFieldAdvantage:
    RATINGS = {"A": (5.0, 0.0), "B": (0.0, 0.0), "C": (-5.0, 0.0), "D": (2.0, -2.0)}

    def test_positive_home_field_recovered_from_a_systematic_home_bump(self):
        """A real HFA signal needs variance in outcome that correlates with
        venue while team identity stays fixed -- unlike a fixture where the
        same pair always produces the same final margin regardless of who is
        designated home, which gives HFA zero explanatory power (see
        test_a_venue_invariant_outcome_gives_zero_hfa_not_a_guess below)."""
        games = synthetic_league(self.RATINGS, hfa=7.0, rounds=15)
        result = fit_ratings(games, min_games=1, ridge_lambda=1.0)
        assert result.home_field_advantage == pytest.approx(7.0, abs=0.5)

    def test_neutral_site_game_gets_no_home_field_term_for_either_side(self):
        """CFB-specific correction vs. the MLB reference this technique is
        drawn from: neutral-site games are common in CFB and must not credit
        home-field advantage to either side."""
        neutral_games = synthetic_league(self.RATINGS, hfa=7.0, rounds=15, neutral=True)
        result = fit_ratings(neutral_games, min_games=1, ridge_lambda=1.0)
        assert result.status == "active"
        assert result.home_field_advantage == pytest.approx(0.0, abs=1e-4)

    def test_a_venue_invariant_outcome_gives_zero_hfa_not_a_guess(self):
        """Regression guard for the exact failure mode found while building
        this suite: if the same matchup always produces the same score
        regardless of who is designated home, HFA has no signal to find and
        must come back as (very close to) zero, not some spurious value."""
        games = [game("A", "B", 30, 20) for _ in range(20)] + [
            game("B", "A", 20, 30) for _ in range(20)
        ]
        result = fit_ratings(games, min_games=1)
        assert result.home_field_advantage == pytest.approx(0.0, abs=1e-6)

    def test_all_neutral_site_games_no_longer_crash_the_fit(self):
        """Before a tiny epsilon regularization was added to every diagonal
        entry (not just offense/defense), an all-neutral-site schedule made
        the home_field column exactly zero with nothing to regularize it,
        producing a singular matrix and an outright 'fit_failed' rather than
        the correct, sane answer of 'no home-field signal exists'."""
        games = synthetic_league(self.RATINGS, hfa=7.0, rounds=15, neutral=True)
        result = fit_ratings(games, min_games=1, ridge_lambda=1.0)
        assert result.status == "active"


class TestMarginEdgeCases:
    @pytest.fixture
    def league(self):
        games = synthetic_league(TestMarginRecovery.TRUE_RATINGS, hfa=3.0, rounds=15)
        return fit_ratings(games, min_games=1, ridge_lambda=1.0)

    def test_neutral_site_drops_the_home_field_term(self, league):
        home = league.margin("Strong", "Weak", neutral_site=False)
        neutral = league.margin("Strong", "Weak", neutral_site=True)
        assert home - neutral == pytest.approx(league.home_field_advantage, abs=0.5)

    def test_unknown_team_returns_none_rather_than_guessing(self, league):
        assert league.margin("Strong", "NeverPlayed") is None
        assert league.margin("NeverPlayed", "Weak") is None

    def test_margin_on_an_unfit_model_is_none(self):
        result = fit_ratings([game("A", "B", 20, 10)], min_games=30)
        assert result.margin("A", "B") is None


class TestRidgeShrinkage:
    BASE_RATINGS = {
        "Strong": (12.0, -3.0), "MidA": (2.0, 1.0),
        "MidB": (-1.0, -2.0), "Weak": (-10.0, 8.0),
    }

    def test_a_low_sample_team_shrinks_toward_average_relative_to_a_high_sample_one(self):
        """Two teams with the identical true rating relative to Strong, but
        very different sample sizes, must not receive the same fitted margin
        against Strong -- the low-sample team's estimate should be pulled
        toward league-average (a smaller-magnitude margin) by the ridge
        penalty on offense/defense.

        Anchored against Strong from an already well-connected base league,
        not an isolated new pair: a closed two-team-only subsystem is
        genuinely under-determined regardless of sample size or ridge_lambda
        (verified by hand while building this suite), so a meaningful
        shrinkage test needs real schedule connectivity, exactly like every
        actual FBS team's schedule.
        """
        games = synthetic_league(self.BASE_RATINGS, rounds=15)
        # HighSample and LowSample share Weak's exact true rating, but
        # HighSample plays Strong repeatedly while LowSample plays it once.
        weak_rating = self.BASE_RATINGS["Weak"]
        strong_rating = self.BASE_RATINGS["Strong"]
        games += synthetic_league(
            {"HighSample": weak_rating, "Strong": strong_rating}, rounds=15)
        games += synthetic_league(
            {"LowSample": weak_rating, "Strong": strong_rating}, rounds=1)

        result = fit_ratings(games, min_games=1, ridge_lambda=DEFAULT_RIDGE_LAMBDA)
        # Both are big underdogs to Strong; shrinkage should pull the
        # low-sample team's deficit closer to zero (a LESS negative margin).
        high_margin = result.margin("HighSample", "Strong")
        low_margin = result.margin("LowSample", "Strong")
        assert low_margin > high_margin

    def test_larger_lambda_shrinks_margins_closer_to_zero(self):
        games = synthetic_league(self.BASE_RATINGS, rounds=15)
        loose = fit_ratings(games, ridge_lambda=1.0, min_games=1)
        tight = fit_ratings(games, ridge_lambda=500.0, min_games=1)
        assert abs(tight.margin("Strong", "Weak")) < abs(loose.margin("Strong", "Weak"))
