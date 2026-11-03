"""Linear market/model probability blend (settings.blend.market_weight_floor).

    p_blend = w * p_market + (1 - w) * p_model,   w in [floor, 1]

This is the Outlier-style discipline cap: until a segment shows out-of-sample
Brier improvement from a larger model share, the market side is never weighted
below ``market_weight_floor`` (default 0.75 → model share ≤ 0.25).

Pure math only — walk-forward fitting of ``w`` lives in
``backtest/market_blend_fit.py``. Live callers read the floor from
``config.settings()["blend"]`` and pass it here; backtest/promote must apply
the same formula so evidence matches live scoring.
"""

from __future__ import annotations

_PROB_EPS = 1e-12
DEFAULT_MARKET_WEIGHT_FLOOR = 0.75


def clip_prob(p: float) -> float:
    return min(max(float(p), _PROB_EPS), 1.0 - _PROB_EPS)


def clamp_market_weight(w: float, *, floor: float = DEFAULT_MARKET_WEIGHT_FLOOR) -> float:
    """Clamp ``w`` into ``[floor, 1]``. Raises if floor is outside [0, 1]."""
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"market_weight_floor must be in [0, 1], got {floor!r}")
    return min(1.0, max(float(floor), float(w)))


def blend_with_market(
    p_market: float,
    p_model: float,
    w: float,
    *,
    floor: float = DEFAULT_MARKET_WEIGHT_FLOOR,
) -> float:
    """Return ``w * p_market + (1-w) * p_model`` with ``w`` clamped to ``[floor, 1]``."""
    weight = clamp_market_weight(w, floor=floor)
    return clip_prob(weight * clip_prob(p_market) + (1.0 - weight) * clip_prob(p_model))


def market_weight_floor_from_settings(settings: dict | None) -> float:
    """Read ``blend.market_weight_floor`` from a settings dict (default 0.75)."""
    if not settings:
        return DEFAULT_MARKET_WEIGHT_FLOOR
    blend = settings.get("blend") or {}
    raw = blend.get("market_weight_floor", DEFAULT_MARKET_WEIGHT_FLOOR)
    floor = float(raw)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"settings.blend.market_weight_floor out of range: {floor!r}")
    return floor
