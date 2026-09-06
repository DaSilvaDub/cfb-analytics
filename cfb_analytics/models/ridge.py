"""Opponent-adjusted offense/defense ratings via ridge regression.

Model (one observation per team-side per game):

    points = mu + home_field * is_home + offense[team] - defense[opponent]

Ridge (L2 shrinkage toward zero) applies to the offense/defense terms only,
never to ``mu`` or ``home_field``, for two reasons: the offense/defense split
is otherwise rank-deficient (only their difference is ever observed, so the
system has a null space along "add k to every offense and every defense"),
and a team with few games shrinks toward league-average rather than an
extreme estimate built from a handful of results.

This mirrors the technique already proven out in the sibling `outlier`
project's `team_strength.py` (fit on MLB run totals via the same dense
normal-equations construction and Gaussian elimination) -- not its code, this
repo shares none with that one by design. One deliberate CFB-specific
correction: a **neutral-site game contributes no home-field term for either
side**. The MLB reference has no neutral-site concept; CFB does (bowl games,
international games, many rivalry games), and `games.neutral_site` already
distinguishes it.

Recency weighting (plan section 6.1) is implemented: passing ``as_of``
weights each game by ``exp(-delta_days / tau)``, ``tau = 45`` fixed by the
plan, where ``delta_days`` is the gap between that game's kickoff and
``as_of``. A game exactly at ``as_of`` gets full weight (1.0); one 45 days
older gets ~37%; 90 days older, ~13%. Omitting ``as_of`` (the default)
disables weighting entirely -- every observation gets weight 1.0, which is
also what a game with an unparseable or missing ``kickoff_utc`` gets even
when ``as_of`` IS given, since a fit should degrade to "acts as if this
game were current" rather than crash over one row missing a date.

Deliberately NOT yet implemented, and tracked as explicit follow-on work
rather than silently skipped: the early-season shrinkage prior blending in
returning-production and recruiting talent (the plan's ``O_prior`` blend --
see ``models/shrinkage.py`` and ``features/team_ratings.py``, where it is
actually applied as a post-fit blend, not inside this module). This module
produces the plain, unblended ``O_fit``/``D_fit`` that blend is built on top
of.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cfb_analytics.models.linalg import solve

DEFAULT_RIDGE_LAMBDA = 25.0
DEFAULT_MIN_GAMES = 30

# Plan section 6.1: w_i = exp(-delta_days_i / tau), tau = 45. Not a tuned
# value -- the plan states this constant directly, unlike ridge_lambda
# ("seeded at 25... retuned") or the shrinkage coefficients (explicitly
# "fitted... by minimising out-of-sample margin error").
RECENCY_TAU_DAYS = 45.0


def _field(game: Any, name: str) -> Any:
    if isinstance(game, dict):
        return game.get(name)
    return getattr(game, name, None)


def _recency_weight(game: Any, as_of: datetime | None) -> float:
    if as_of is None:
        return 1.0
    raw = _field(game, "kickoff_utc")
    if raw is None:
        return 1.0
    try:
        kickoff = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))
    except ValueError:
        return 1.0
    delta_days = (as_of - kickoff).total_seconds() / 86400.0
    return math.exp(-delta_days / RECENCY_TAU_DAYS)


@dataclass(frozen=True)
class _Observation:
    scoring_team: str
    opponent_team: str
    is_home: bool
    points: float
    weight: float


def _observations(games: list[Any], *, as_of: datetime | None = None) -> list[_Observation]:
    """Two scoring observations per game, or none for a row missing a score.

    A neutral-site game (``neutral_site`` truthy) sets ``is_home=False`` for
    BOTH sides -- see the module docstring for why this deviates from the
    MLB reference this technique is drawn from.
    """
    observations: list[_Observation] = []
    for game in games:
        home = _field(game, "home_team_id")
        away = _field(game, "away_team_id")
        home_points = _field(game, "home_points")
        away_points = _field(game, "away_points")
        if not home or not away or home_points is None or away_points is None:
            continue
        try:
            home_points = float(home_points)
            away_points = float(away_points)
        except (TypeError, ValueError):
            continue
        neutral = bool(_field(game, "neutral_site"))
        weight = _recency_weight(game, as_of)
        observations.append(_Observation(home, away, not neutral, home_points, weight))
        observations.append(_Observation(away, home, False, away_points, weight))
    return observations


@dataclass(frozen=True)
class TeamRating:
    offense: float
    defense: float
    games: int


@dataclass(frozen=True)
class RidgeRatings:
    status: str  # "active" | "insufficient_history" | "fit_failed"
    n_games: int
    ridge_lambda: float
    league_avg_points: float | None = None
    home_field_advantage: float | None = None
    teams: dict[str, TeamRating] = field(default_factory=dict)

    def margin(
        self, home_team_id: str, away_team_id: str, *, neutral_site: bool = False
    ) -> float | None:
        """Projected home-minus-away point margin (plan section 6.3).

        Returns None -- never a guess -- when either team has no rating in
        this fit, e.g. an FCS opponent the ridge was never trained on.
        """
        if self.status != "active":
            return None
        home = self.teams.get(home_team_id)
        away = self.teams.get(away_team_id)
        if home is None or away is None:
            return None
        hfa = 0.0 if neutral_site else (self.home_field_advantage or 0.0)
        return (home.offense - away.defense) - (away.offense - home.defense) + hfa


def fit_ratings(
    games: list[Any],
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_games: int = DEFAULT_MIN_GAMES,
    as_of: datetime | None = None,
) -> RidgeRatings:
    """Fit ratings from ``games``.

    ``as_of`` enables recency weighting (see the module docstring): pass the
    fit's cutoff instant (typically the same ``as_of_utc`` the caller used to
    select which games are even eligible) and older games count for less.
    Omit it for the old, unweighted behavior -- every existing caller that
    does not pass it sees no change.
    """
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda cannot be negative")
    if min_games < 1:
        raise ValueError("min_games must be at least 1")

    observations = _observations(games, as_of=as_of)
    game_count = len(observations) // 2
    if game_count < min_games:
        return RidgeRatings(
            status="insufficient_history", n_games=game_count, ridge_lambda=ridge_lambda
        )

    teams = sorted(
        {obs.scoring_team for obs in observations} | {obs.opponent_team for obs in observations}
    )
    offense_index = {team: 2 + i for i, team in enumerate(teams)}
    defense_index = {team: 2 + len(teams) + i for i, team in enumerate(teams)}
    n = 2 + 2 * len(teams)
    xtx = [[0.0] * n for _ in range(n)]
    xty = [0.0] * n

    for obs in observations:
        indices = [0, 1, offense_index[obs.scoring_team], defense_index[obs.opponent_team]]
        values = [1.0, 1.0 if obs.is_home else 0.0, 1.0, -1.0]
        for a, va in zip(indices, values, strict=True):
            xty[a] += obs.weight * va * obs.points
            for b, vb in zip(indices, values, strict=True):
                xtx[a][b] += obs.weight * va * vb

    # A tiny, fixed regularization on EVERY diagonal entry (mu, home_field,
    # and every offense/defense term) guards against an exactly-singular
    # system in degenerate schedules -- e.g. every game in the fit being
    # neutral-site, which makes the home_field column all zeros and its own
    # diagonal entry exactly 0 with no ridge_lambda to rescue it (ridge_lambda
    # below applies to offense/defense only). This epsilon is negligible for
    # any real, well-conditioned schedule; it exists purely so a degenerate
    # input degrades to a sane answer (home_field ~= 0) instead of an outright
    # fit failure.
    _EPSILON = 1e-8
    for idx in range(n):
        xtx[idx][idx] += _EPSILON
    for idx in range(2, n):
        xtx[idx][idx] += ridge_lambda

    solution = solve(xtx, xty)
    if solution is None:
        return RidgeRatings(status="fit_failed", n_games=game_count, ridge_lambda=ridge_lambda)

    team_games: dict[str, int] = dict.fromkeys(teams, 0)
    for obs in observations:
        team_games[obs.scoring_team] += 1

    return RidgeRatings(
        status="active",
        n_games=game_count,
        ridge_lambda=ridge_lambda,
        league_avg_points=solution[0],
        home_field_advantage=solution[1],
        teams={
            team: TeamRating(
                offense=solution[offense_index[team]],
                defense=solution[defense_index[team]],
                games=team_games[team],
            )
            for team in teams
        },
    )
