"""L2-penalized logistic regression via IRLS (plan section 6.4's ``P_logit``).

IRLS (iteratively reweighted least squares) for logistic regression is
exactly Newton-Raphson on the penalized log-likelihood. At each step, solve

    (X^T W X + lambda*D) * delta = X^T (y - p) - lambda * w_reg

for ``delta`` and set ``w <- w + delta``, where ``p = sigmoid(X w)``,
``W = diag(p_i * (1 - p_i))``, ``D`` is the identity with a 0 at the
intercept position (the intercept is never penalized, exactly like ridge
never penalizes ``mu``/``home_field``), and ``w_reg`` is ``w`` with that same
0 at the intercept position. This reuses ``models/linalg.solve`` for the
linear system at each iteration -- the same dense Gaussian-elimination
solver ridge's own normal equations use, at a much smaller scale here (a
handful of features, not ~460 team parameters).

The plan calls for the full T1-T5 feature vector (~40 features: EPA/success
rate/havoc diffs, QB experience, rest/travel/rivalry, ...). This module is
feature-agnostic -- it fits whatever design matrix it is given -- but the
actual feature set assembled by ``features/ensemble.py`` is deliberately a
SUBSET of that (talent/returning-production diffs plus a home-field
indicator), not the full T1-T5 vector: see that module's docstring for
exactly what is included now and what is a tracked gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cfb_analytics.models.linalg import solve

DEFAULT_L2_LAMBDA = 1.0
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_TOLERANCE = 1e-6

# Keeps p_i away from exactly 0 or 1: a perfectly separated batch would
# otherwise send W's diagonal to exactly zero, at which point (X^T W X) loses
# rank in whatever directions perfectly separate the data, independent of
# regularization strength on those directions. This is the same fix
# backtest/metrics.py's log_loss uses for the same underlying reason.
_PROB_EPS = 1e-9


def sigmoid(x: float) -> float:
    # The naive exp(-x) overflows for very negative x; this form is the
    # standard numerically-stable split.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"logit is only defined on (0, 1), got {p!r}")
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class LogisticFit:
    status: str  # "active" | "insufficient_data" | "fit_failed"
    n: int
    weights: list[float] = field(default_factory=list)
    feature_names: tuple[str, ...] = ()
    iterations: int = 0

    def probability(self, features: list[float]) -> float | None:
        """Sigmoid(intercept + weights . features). None if this fit never
        converged to anything usable, or the feature vector's length does
        not match what this model was trained on."""
        if self.status != "active":
            return None
        if len(features) != len(self.weights) - 1:
            return None
        z = self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], features, strict=True))
        return sigmoid(z)


def fit_logistic(
    design_rows: list[list[float]],
    outcomes: list[bool],
    *,
    feature_names: tuple[str, ...] = (),
    l2_lambda: float = DEFAULT_L2_LAMBDA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    min_n: int = 30,
) -> LogisticFit:
    """Fit ``P(outcome) = sigmoid(intercept + w . features)`` by IRLS.

    ``design_rows`` is one feature vector per observation (NOT including the
    intercept column -- this function adds it). ``l2_lambda`` penalizes every
    weight except the intercept, exactly like ridge's own regularization
    choice and for the same reason: an unpenalized intercept is just the
    league's baseline win rate in log-odds, not a per-feature effect that
    needs shrinking.
    """
    n = len(design_rows)
    if n != len(outcomes):
        raise ValueError("design_rows and outcomes must be the same length")
    if n < min_n:
        return LogisticFit(status="insufficient_data", n=n, feature_names=feature_names)

    n_features = len(design_rows[0]) + 1  # +1 for the intercept
    if any(len(row) + 1 != n_features for row in design_rows):
        raise ValueError("every row in design_rows must have the same number of features")

    x = [[1.0, *row] for row in design_rows]
    y = [1.0 if outcome else 0.0 for outcome in outcomes]
    weights = [0.0] * n_features

    for iteration in range(1, max_iterations + 1):
        linear = [sum(wj * xij for wj, xij in zip(weights, row, strict=True)) for row in x]
        probabilities = [min(max(sigmoid(z), _PROB_EPS), 1.0 - _PROB_EPS) for z in linear]

        xtwx = [[0.0] * n_features for _ in range(n_features)]
        gradient = [0.0] * n_features
        for row, y_i, p_i in zip(x, y, probabilities, strict=True):
            w_i = p_i * (1.0 - p_i)
            residual = y_i - p_i
            for a in range(n_features):
                gradient[a] += row[a] * residual
                xa = row[a] * w_i
                for b in range(n_features):
                    xtwx[a][b] += xa * row[b]

        for j in range(1, n_features):  # never penalize the intercept (index 0)
            xtwx[j][j] += l2_lambda
            gradient[j] -= l2_lambda * weights[j]

        delta = solve(xtwx, gradient)
        if delta is None:
            return LogisticFit(
                status="fit_failed", n=n, feature_names=feature_names, iterations=iteration
            )

        weights = [w + d for w, d in zip(weights, delta, strict=True)]
        if max(abs(d) for d in delta) < tolerance:
            return LogisticFit(
                status="active", n=n, weights=weights,
                feature_names=feature_names, iterations=iteration,
            )

    return LogisticFit(
        status="active", n=n, weights=weights,
        feature_names=feature_names, iterations=max_iterations,
    )
