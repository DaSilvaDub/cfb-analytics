"""Walk-forward probability calibration for ensemble P(home) (promotion gate).

Temperature scaling and Platt logistic on logit(p) — few degrees of freedom,
stdlib-only. Fit strictly on *prior* (season, week) folds; never on the
evaluation slice. Isotonic is intentionally omitted: with thin mid-bins it
overfits easily and needs a peek-proof fold design that is harder to audit.

Identity (T=1 / a=1,b=0) is used until ``min_fit_n`` prior labeled games exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from cfb_analytics.models.logistic import fit_logistic, logit, sigmoid

Method = Literal["temperature", "platt", "identity"]
Fold = Literal["week", "season"]

_PROB_EPS = 1e-9
_DEFAULT_MIN_FIT_N = 200
_TEMP_GRID = tuple(round(0.50 + 0.05 * i, 2) for i in range(51))  # 0.50 .. 3.00


@dataclass(frozen=True)
class ProbCalibrator:
    """Parametric map p -> p_cal on (0, 1)."""

    method: Method
    # temperature: temperature; platt: (a, b) for sigmoid(a*logit(p)+b);
    # identity: unused
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    n_fit: int = 0

    def apply(self, p: float) -> float:
        clipped = min(max(p, _PROB_EPS), 1.0 - _PROB_EPS)
        if self.method == "identity":
            return clipped
        z = logit(clipped)
        if self.method == "temperature":
            t = self.temperature
            if t <= 0:
                raise ValueError(f"temperature must be positive, got {t!r}")
            return sigmoid(z / t)
        # platt
        return sigmoid(self.platt_a * z + self.platt_b)

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "n_fit": self.n_fit,
        }


IDENTITY = ProbCalibrator(method="identity", n_fit=0)


def _nll(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    total = 0.0
    n = len(probs)
    for p, won in zip(probs, outcomes, strict=True):
        clipped = min(max(p, _PROB_EPS), 1.0 - _PROB_EPS)
        total += -math.log(clipped) if won else -math.log(1.0 - clipped)
    return total / n


def fit_temperature(
    probs: Sequence[float],
    outcomes: Sequence[bool],
    *,
    grid: Sequence[float] = _TEMP_GRID,
) -> ProbCalibrator:
    """1-DOF temperature: p_cal = sigmoid(logit(p) / T). Grid search on NLL."""
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    if len(probs) < 2:
        raise ValueError("need at least 2 games to fit temperature")
    best_t = 1.0
    best_loss = float("inf")
    for t in grid:
        if t <= 0:
            continue
        calibrated = [sigmoid(logit(min(max(p, _PROB_EPS), 1.0 - _PROB_EPS)) / t) for p in probs]
        loss = _nll(calibrated, outcomes)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return ProbCalibrator(method="temperature", temperature=best_t, n_fit=len(probs))


def fit_platt(
    probs: Sequence[float],
    outcomes: Sequence[bool],
    *,
    l2_lambda: float = 0.0,
    min_n: int = 2,
) -> ProbCalibrator:
    """2-DOF Platt: p_cal = sigmoid(a * logit(p) + b) via IRLS logistic."""
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    design = [[logit(min(max(p, _PROB_EPS), 1.0 - _PROB_EPS))] for p in probs]
    fit = fit_logistic(
        design,
        list(outcomes),
        feature_names=("logit_p",),
        l2_lambda=l2_lambda,
        min_n=min_n,
    )
    if fit.status != "active" or len(fit.weights) != 2:
        return IDENTITY
    # weights[0]=intercept b, weights[1]=a
    return ProbCalibrator(
        method="platt",
        platt_a=float(fit.weights[1]),
        platt_b=float(fit.weights[0]),
        n_fit=len(probs),
    )


def fit_calibrator(
    probs: Sequence[float],
    outcomes: Sequence[bool],
    *,
    method: Method = "temperature",
) -> ProbCalibrator:
    if method == "identity":
        return ProbCalibrator(method="identity", n_fit=len(probs))
    if method == "temperature":
        return fit_temperature(probs, outcomes)
    if method == "platt":
        return fit_platt(probs, outcomes)
    raise ValueError(f"unknown calibration method: {method!r}")


@dataclass(frozen=True)
class TimedProb:
    """One labeled probability with chronological keys for walk-forward."""

    season: int
    week: int
    p: float
    won: bool
    # Opaque handle so callers can zip back to game order (e.g. game_id).
    key: str = ""


def walk_forward_calibrate(
    rows: Sequence[TimedProb],
    *,
    method: Method = "platt",
    min_fit_n: int = _DEFAULT_MIN_FIT_N,
    fold: Fold = "season",
) -> list[tuple[float, float, ProbCalibrator]]:
    """Calibrate each row using only strictly prior fold labels.

    ``fold="season"`` (default): one fit per season on all prior seasons —
    stable with few DOF, no within-season peek. ``fold="week"``: one fit
    per (season, week) on all earlier weeks (and prior seasons).

    Returns list aligned with ``rows``: (raw_p, calibrated_p, calibrator_used).
    When fewer than ``min_fit_n`` prior games exist, identity is applied.
    """
    if not rows:
        return []

    groups: dict[tuple[int, int], list[int]] = {}
    for i, row in enumerate(rows):
        key = (row.season, 0 if fold == "season" else row.week)
        groups.setdefault(key, []).append(i)

    order_sorted = sorted(groups)
    prior_probs: list[float] = []
    prior_outcomes: list[bool] = []
    out: list[tuple[float, float, ProbCalibrator]] = [None] * len(rows)  # type: ignore[list-item]

    for fold_key in order_sorted:
        if len(prior_probs) >= min_fit_n and method != "identity":
            cal = fit_calibrator(prior_probs, prior_outcomes, method=method)
        else:
            cal = IDENTITY
        for i in groups[fold_key]:
            raw = rows[i].p
            out[i] = (raw, cal.apply(raw), cal)
        for i in groups[fold_key]:
            prior_probs.append(rows[i].p)
            prior_outcomes.append(rows[i].won)

    return out
