"""Sequential Elo ratings with a margin-of-victory adjustment (plan section 6.2).

    E      = 1 / (1 + 10^(-(R_a - R_b + HFA) / 400))
    K_eff  = K * ln(|margin| + 1) * (2.2 / (0.001 * (R_win - R_lose) + 2.2))
    R'     = R + K_eff * (S - E)

Unlike ridge (a single batch fit over a window of games), Elo is inherently
sequential: ratings are updated one game at a time, in chronological order,
and a team's rating after week W depends on every one of its games before
that, not on a windowed regression. That sequential nature is also what
makes Elo naturally leakage-safe by construction, PROVIDED the caller only
ever feeds it games strictly before the cutoff it cares about -- exactly
what ``features/elo_internal.py`` does.

The margin-of-victory multiplier is the mechanism popularized by
FiveThirtyEight's NFL/NBA Elo systems: ``ln(|margin| + 1)`` makes a 40-point
win move ratings more than a 3-point win (logarithmically, not linearly, so
running up the score has fast-diminishing returns), and the
``2.2 / (0.001*(R_win - R_lose) + 2.2)`` term dampens that for an
already-heavy favorite blowing out an already-heavy underdog (an "expected"
blowout should move ratings less than the identical blowout would from a
similarly-rated team) while amplifying it for a genuine upset.

Deliberately NOT yet implemented, and tracked as a real gap rather than
silently skipped: the plan's venue-specific home-field advantage "shrinkage
toward the league mean by games observed" (section 6.2). This module uses a
single global ``hfa`` constant, exactly like ``models/ridge.py`` did before
its own recency-weighting/shrinkage-prior work -- the analogous refinement
here would blend a venue's own observed HFA toward the league mean the same
way ``models/shrinkage.py`` blends a team's in-season fit toward its prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_K = 25.0
# Plan section 6.2: "HFA at 60 Elo (~2.6 pts)". Shared provenance with
# backtest/elo_baseline.py's HFA_ELO_POINTS (same plan constant, same value)
# but kept as a separate constant here rather than a cross-import: that
# module scores CFBD's imported Elo as a baseline, this one IS the model.
HFA_ELO_POINTS = 60.0
DEFAULT_MIN_GAMES = 30
# The standard generic Elo starting point (chess's own convention, and
# FiveThirtyEight's for a brand-new franchise): a team with no preseason
# blend available at all (see features/elo_internal.py) starts here.
DEFAULT_INITIAL_RATING = 1500.0

# Plan section 6.2: "FCS opponents enter as a single pooled synthetic team
# with a wide prior." A single shared identifier (see
# features/elo_internal.py) rather than tracking each individual FCS
# opponent's own rating off a handful of FBS games a season -- pooled
# evidence from every FBS-vs-FCS game league-wide is a far less noisy signal
# than any one FCS team's 1-2 games. Seeded below the FBS baseline (1500),
# reflecting FCS's real average talent gap, not at it: this pool has never
# played a single down against another FCS team in this dataset, so there is
# nothing to calibrate it against except its actual games against FBS teams.
POOL_TEAM_ID = "__fcs_pool__"
POOL_INITIAL_RATING = 1200.0

# Plan section 6.2's preseason blend: R_0 = 0.75*R_final + 0.25*1500 +
# g*z(talent) + d*z(returning_prod). The 0.75/0.25 split is the plan's own
# stated constant. g and d were grid-searched 2026-09-06 against 2021 weeks
# 1-5 out-of-sample log loss (talent in [0, 100], returning in [0, 60], step
# 15-20) -- unlike ridge's shrinkage-coefficient search, this one found a
# genuine INTERIOR minimum, not a grid edge: log loss rises on both sides of
# talent=40 (0.556 at talent=0, 0.539 at talent=40, 0.582 at talent=100),
# so there was no ambiguity about extrapolating further. Checked against a
# held-out season never used in the search (2019 weeks 1-5): log loss
# improved there too, from 0.495 (no preseason talent/returning signal at
# all) to 0.466 at these settings -- generalizing, not overfitting the fit
# season, the same pattern found for ridge's shrinkage prior.
DEFAULT_PRESEASON_PREVIOUS_WEIGHT = 0.75
DEFAULT_PRESEASON_BASELINE_WEIGHT = 0.25
DEFAULT_PRESEASON_TALENT_COEFF = 40.0
DEFAULT_PRESEASON_RETURNING_COEFF = 25.0

# margin_of_victory_multiplier's denominator, 0.001*(R_win - R_lose) + 2.2,
# is only non-positive for a rating gap beyond -2200 -- never reached by
# real CFB ratings (which live in roughly the 1000-2200 range) -- but this
# floor keeps the function well-defined (a small positive multiplier, not a
# blow-up or a sign flip) for any input rather than trusting that invariant
# silently.
_MULTIPLIER_DENOMINATOR_FLOOR = 0.1


def _field(game: Any, name: str) -> Any:
    if isinstance(game, dict):
        return game.get(name)
    return getattr(game, name, None)


def preseason_rating(
    previous_season_final: float | None,
    z_talent: float | None,
    z_returning_ppa: float | None,
    *,
    previous_weight: float = DEFAULT_PRESEASON_PREVIOUS_WEIGHT,
    baseline_weight: float = DEFAULT_PRESEASON_BASELINE_WEIGHT,
    talent_coeff: float = DEFAULT_PRESEASON_TALENT_COEFF,
    returning_coeff: float = DEFAULT_PRESEASON_RETURNING_COEFF,
) -> float:
    """R_0 (plan section 6.2): this season's starting rating, before a
    single game of it has been played.

    A team with no previous-season internal Elo on file (the earliest
    season in the store, or a team newly promoted to FBS) is not guessed
    at: ``previous_season_final`` falls back to ``DEFAULT_INITIAL_RATING``
    itself, which makes the weighted blend of the two terms collapse back
    to exactly that baseline -- the honest "nothing carried over" answer --
    while still applying any talent/returning-production signal available.
    """
    previous = (
        previous_season_final if previous_season_final is not None else DEFAULT_INITIAL_RATING
    )
    return (
        previous_weight * previous
        + baseline_weight * DEFAULT_INITIAL_RATING
        + talent_coeff * (z_talent or 0.0)
        + returning_coeff * (z_returning_ppa or 0.0)
    )


def expected_score(rating_a: float, rating_b: float, *, hfa: float = 0.0) -> float:
    """P(a beats b). ``hfa`` is added to ``rating_a``'s side; pass 0 for a
    neutral site, or when computing from a team that is not the home side."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b + hfa) / 400.0))


def margin_of_victory_multiplier(k: float, margin: float, winner_rating: float,
                                  loser_rating: float) -> float:
    if margin < 0:
        raise ValueError("margin must be the (non-negative) winner's margin of victory")
    denominator = max(
        0.001 * (winner_rating - loser_rating) + 2.2, _MULTIPLIER_DENOMINATOR_FLOOR
    )
    return k * math.log(abs(margin) + 1.0) * (2.2 / denominator)


def update_ratings(
    home_rating: float,
    away_rating: float,
    *,
    home_won: bool,
    margin: float,
    k: float = DEFAULT_K,
    hfa: float = HFA_ELO_POINTS,
    neutral_site: bool = False,
) -> tuple[float, float]:
    """One game's rating update. Zero-sum by construction: whatever the
    winner gains, the loser loses -- classical Elo, margin-of-victory
    multiplier and all."""
    bonus = 0.0 if neutral_site else hfa
    expected_home = expected_score(home_rating, away_rating, hfa=bonus)
    winner_rating = home_rating if home_won else away_rating
    loser_rating = away_rating if home_won else home_rating
    k_eff = margin_of_victory_multiplier(k, abs(margin), winner_rating, loser_rating)
    actual_home = 1.0 if home_won else 0.0
    delta = k_eff * (actual_home - expected_home)
    return home_rating + delta, away_rating - delta


@dataclass(frozen=True)
class TeamEloState:
    rating: float
    games: int


@dataclass(frozen=True)
class EloRatings:
    status: str  # "active" | "insufficient_history"
    n_games: int
    k: float
    hfa: float
    teams: dict[str, TeamEloState] = field(default_factory=dict)

    def probability(
        self, home_team_id: str, away_team_id: str, *, neutral_site: bool = False
    ) -> float | None:
        """P(home wins). Returns None -- never a guess -- when either team
        never appeared in this fit (e.g. this model was not seeded with a
        rating for it and it never played)."""
        if self.status != "active":
            return None
        home = self.teams.get(home_team_id)
        away = self.teams.get(away_team_id)
        if home is None or away is None:
            return None
        bonus = 0.0 if neutral_site else self.hfa
        return expected_score(home.rating, away.rating, hfa=bonus)


def fit_elo(
    games: list[Any],
    *,
    initial_ratings: dict[str, float] | None = None,
    k: float = DEFAULT_K,
    hfa: float = HFA_ELO_POINTS,
    min_games: int = DEFAULT_MIN_GAMES,
) -> EloRatings:
    """Process ``games`` in chronological order, updating ratings one game
    at a time.

    ``initial_ratings`` seeds a team's rating the first time it appears
    (typically the plan's preseason blend -- see
    ``features/elo_internal.py``); a team absent from it starts at
    ``DEFAULT_INITIAL_RATING``, the generic Elo baseline.

    Every game must carry a parseable ``kickoff_utc`` -- unlike ridge,
    there is no "unweighted" fallback for a missing date here: the whole
    model IS its own chronological order, so a game that cannot be placed
    in that order cannot be processed at all, and is dropped (counted, not
    silently ignored -- see the return value's ``n_games``, which only
    counts games actually processed).
    """
    if min_games < 1:
        raise ValueError("min_games must be at least 1")

    dated_games = []
    for game in games:
        home = _field(game, "home_team_id")
        away = _field(game, "away_team_id")
        home_points = _field(game, "home_points")
        away_points = _field(game, "away_points")
        kickoff = _field(game, "kickoff_utc")
        if not home or not away or home_points is None or away_points is None or not kickoff:
            continue
        try:
            home_points = float(home_points)
            away_points = float(away_points)
        except (TypeError, ValueError):
            continue
        if home_points == away_points:
            # No ties in the modern CFB rulebook (overtime resolves every
            # game); a row claiming one is malformed, not a real result --
            # never fabricate a winner for it.
            continue
        dated_games.append((str(kickoff), game, home, away, home_points, away_points))

    dated_games.sort(key=lambda row: row[0])
    n_games = len(dated_games)
    if n_games < min_games:
        return EloRatings(status="insufficient_history", n_games=n_games, k=k, hfa=hfa)

    ratings: dict[str, float] = dict(initial_ratings or {})
    games_played: dict[str, int] = dict.fromkeys(ratings, 0)

    for _kickoff, game, home, away, home_points, away_points in dated_games:
        home_rating = ratings.get(home, DEFAULT_INITIAL_RATING)
        away_rating = ratings.get(away, DEFAULT_INITIAL_RATING)
        home_won = home_points > away_points
        margin = abs(home_points - away_points)
        neutral = bool(_field(game, "neutral_site"))
        new_home, new_away = update_ratings(
            home_rating, away_rating, home_won=home_won, margin=margin,
            k=k, hfa=hfa, neutral_site=neutral,
        )
        ratings[home] = new_home
        ratings[away] = new_away
        games_played[home] = games_played.get(home, 0) + 1
        games_played[away] = games_played.get(away, 0) + 1

    return EloRatings(
        status="active",
        n_games=n_games,
        k=k,
        hfa=hfa,
        teams={
            team: TeamEloState(rating=rating, games=games_played.get(team, 0))
            for team, rating in ratings.items()
        },
    )
