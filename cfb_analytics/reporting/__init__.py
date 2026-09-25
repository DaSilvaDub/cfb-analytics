"""Reporting and Presentation Layer for cfb-analytics.

Provides:
- Authoritative terminal card renderers (docs/grok_rules.md §5).
- Structured JSON export utilities with mandatory shadow mode watermark.
- Pure Python standard library implementation (no rich, tabulate, or external AI).
"""

from __future__ import annotations

from cfb_analytics.reporting.export import (
    export_board_json,
    export_mispriced_json,
    export_parlay_json,
    export_reasoning_json,
)
from cfb_analytics.reporting.terminal import (
    render_board_terminal,
    render_mispriced_card,
    render_mispriced_terminal,
    render_parlay_card,
    render_reasoning_card,
)

__all__ = [
    "render_reasoning_card",
    "render_mispriced_card",
    "render_parlay_card",
    "render_board_terminal",
    "render_mispriced_terminal",
    "export_reasoning_json",
    "export_mispriced_json",
    "export_parlay_json",
    "export_board_json",
]
