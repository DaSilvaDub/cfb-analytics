"""Feature layer.

Every feature read goes through ``asof.AsOfReader``, which refuses data
timestamped at or after kickoff. A backtest that leaks looks excellent and is
worthless, so the guard raises rather than warns.
"""
