"""Data models for Comprehensive Mispriced Line Scanner (Milestone 2).

Provides frozen dataclasses, enums, and constants for identifying, devigging,
evaluating, and ranking mispriced opportunities across Game Spreads, Game Totals,
Team Props, and Moneylines.

All models strictly enforce:
- Pure Python standard library primitives (no pandas, numpy, or scipy).
- Absolute immutability via `@dataclass(frozen=True)` and tuple sequences.
- Mandatory Shadow Mode disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Mandatory shadow mode disclaimer for unpromoted research output
SHADOW_MODE_DISCLAIMER: str = "UNPROMOTED - shadow output, not decision-grade"


class PlayTier(str, Enum):
    """Candidate wagering recommendation tier based on 0-100 play score."""

    ELITE = "ELITE"          # Score >= 90.0, exceptional edge and confirmation
    STRONG = "STRONG"        # Score >= 80.0, strong fundamental and pricing edge
    QUALIFIED = "QUALIFIED"  # Score >= 70.0, baseline threshold for actionable recommendation
    LEAN = "LEAN"            # Score >= 60.0, marginal positive edge or thin confirmation
    PASS = "PASS"            # Score < 60.0 or non-positive edge / failed qualification gate
    AVOID = "AVOID"          # Disqualified or high-risk candidate


class QualificationStatus(str, Enum):
    """Status emitted by minimum qualification gating criteria."""

    QUALIFIED = "QUALIFIED"                  # Cleared all gates (EV > 0, edge > 0, n_books >= 3, score >= 70)
    PASS = "PASS"                            # Non-positive edge or non-positive EV
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Thin books, missing price/fair prob, low quality
    STALE = "STALE"                          # Odds snapshot expired or stale
    REVIEW = "REVIEW"                        # Unresolved injury, news flag, or counter-thesis


@dataclass(frozen=True)
class PlayScore:
    """Component breakdown and total score for candidate play scoring (0-100 scale).

    Components:
    1. Historical Hit Rate (max 25 pts)
    2. Estimated Quantitative Edge (max 20 pts)
    3. Line Movement Confirmation (max 15 pts)
    4. Price / Odds Value (max 15 pts)
    5. Sample Size (max 10 pts)
    6. Confirming Insights (max 15 pts)
    """

    hit_rate_score: float
    edge_score: float
    movement_score: float
    price_value_score: float
    sample_size_score: float
    confirming_score: float
    total_score: float
    tier: PlayTier | str
    qualification_status: QualificationStatus | str
    rejection_reasons: tuple[str, ...] = ()
    shadow_mode_disclaimer: str = SHADOW_MODE_DISCLAIMER

    @property
    def is_actionable(self) -> bool:
        """True if play score is >= 70.0 and qualification status is QUALIFIED."""
        status_str = (
            self.qualification_status.value
            if isinstance(self.qualification_status, QualificationStatus)
            else str(self.qualification_status)
        )
        return status_str == QualificationStatus.QUALIFIED.value and self.total_score >= 70.0

    @property
    def is_qualified(self) -> bool:
        """True if cleared all qualification gates."""
        status_str = (
            self.qualification_status.value
            if isinstance(self.qualification_status, QualificationStatus)
            else str(self.qualification_status)
        )
        return status_str == QualificationStatus.QUALIFIED.value


@dataclass(frozen=True)
class MispricedOpportunity:
    """Immutable data record representing an identified mispriced betting line."""

    game_id: str
    market_type: str                   # 'SPREAD', 'TOTAL', 'TEAM_PROP', 'ML'
    market: str                        # e.g., 'SPREAD', 'TOTAL', 'POINTS', 'RUSHING_YARDS', 'RECEIVING_YARDS'
    side: str                          # 'HOME', 'AWAY', 'OVER', 'UNDER'
    line: float                        # Posted market line
    posted_price_american: int         # Actionable American price (e.g. -110, +105)
    posted_price_decimal: float        # Converted decimal price (e.g. 1.909, 2.05)
    consensus_fair_prob: float         # Vig-free consensus probability (Shin or Multiplicative)
    consensus_fair_price_american: int # Consensus fair American price
    model_projected_line: float        # Fundamental model projection (margin, total, yards, points)
    model_prob: float                  # Model-estimated win / cover probability
    edge_pct: float                    # model_prob - consensus_fair_prob
    ev: float                          # Expected Value: model_prob * posted_price_decimal - 1.0
    method_spread: float               # Disagreement between devig methods: |prob_shin - prob_mult|
    n_books: int                       # Number of books contributing to consensus
    play_score: float                  # Candidate play score (0.0 to 100.0)
    play_tier: str                     # ELITE, STRONG, QUALIFIED, LEAN, PASS
    qual_status: str                   # QUALIFIED, PASS, INSUFFICIENT_DATA, STALE, REVIEW
    best_book: str | None = None
    game_label: str = ""               # e.g. "Away at Home"
    kickoff_et: str = ""
    flags: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    shadow_mode_disclaimer: str = SHADOW_MODE_DISCLAIMER

    @property
    def is_actionable(self) -> bool:
        """True if the opportunity meets all criteria for actionability."""
        return (
            self.qual_status == QualificationStatus.QUALIFIED.value
            and self.play_tier in (PlayTier.ELITE.value, PlayTier.STRONG.value, PlayTier.QUALIFIED.value)
            and self.edge_pct > 0.0
            and self.ev > 0.0
            and self.n_books >= 3
        )
