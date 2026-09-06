"""Log-odds ensemble pooling (plan section 6.4).

    logit(P_ens) = sum_k v_k * logit(P_k),    v on the simplex

Pooling in log-odds space (rather than averaging probabilities directly) is
the standard choice for combining independently-derived probability
estimates: it is symmetric in a way a plain average is not (blending a 90%
and a 10% call by averaging gives 50%, which is not what either model
"means"; blending in log-odds space with equal weight gives back a 50%
call only because the two calls exactly cancel in log-odds, which is the
right invariant).

This module is pure math -- given member probabilities and weights, blend
them. Fitting the weights ``v`` (plan: "coordinate search minimising
walk-forward out-of-sample log loss") is a calibration concern, not a math
one, and lives in ``backtest/ensemble_fit.py`` instead, next to
``backtest/calibration.py``'s other fitted constants.
"""

from __future__ import annotations

from cfb_analytics.models.logistic import logit, sigmoid


def pool_probabilities(member_probs: dict[str, float], weights: dict[str, float]) -> float:
    """Blend member probabilities in log-odds space.

    ``weights`` need not be every member ``pool_probabilities`` has ever
    heard of -- pass only the members that actually have a prediction for
    THIS game (a member with insufficient history for an early week is
    absent, not fabricated as 0.5) -- but every key in ``weights`` must have
    a matching entry in ``member_probs``, and the weights present are
    renormalized to sum to 1 among themselves, so dropping a member for one
    game does not silently pull the blend toward a coin flip.
    """
    if not weights:
        raise ValueError("weights cannot be empty")
    missing = [name for name in weights if name not in member_probs]
    if missing:
        raise ValueError(f"no probability given for weighted member(s): {missing}")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive number")
    logit_sum = sum(w * logit(member_probs[name]) for name, w in weights.items())
    return sigmoid(logit_sum / total_weight)
