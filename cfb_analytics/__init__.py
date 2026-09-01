"""NCAA FBS football analytics pipeline.

Three separate models by design — safety (who is least likely to lose), value
(is the price worth paying), and totals — plus a parlay optimizer that can
refuse a leg which buys payout at too high a cost in win probability.

Every output is shadow-mode until the backtest clears the promotion gate in
``config/promotion.json``.
"""

__version__ = "0.1.0"
