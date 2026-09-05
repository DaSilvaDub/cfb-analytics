"""Walk-forward backtest harness (plan section 8).

Nothing here is decision-grade. This package exists to answer one question
honestly: does a model's own claimed win probability match what actually
happened, out of sample, when it is only ever shown data from strictly
before each kickoff? See ``moneyline.run_moneyline_backtest`` for the
current entry point.
"""

from __future__ import annotations
