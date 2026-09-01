"""Market consensus: raw book prices to vig-free probabilities.

This is the **M model** from the plan -- what the market thinks, computed from
prices alone with no fundamentals. It never sees a rating, and the safety model
never sees its output. Keeping those two apart is what stops "edge" from being
an artifact of the market predicting itself.

Anchoring, in order of preference (the fallback chain from plan §3.3):

1. **sharp subset** when at least ``min_sharp_books`` sharp books are priced;
2. otherwise **all books**, flagged ``no_sharp_anchor`` so downstream can
   reduce confidence;
3. never a single book silently standing in for "the market" -- below
   ``min_books_for_consensus`` the market is reported as unpriced.

Measured 2026-09-01: no sharp book appeared in any of 8 captures across 7
slates, so branch 2 is the live path today. Branch 1 exists because coverage is
time-varying and the survey is ongoing, not because it is currently reachable.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cfb_analytics.errors import DevigError
from cfb_analytics.models import devig
from cfb_analytics.utils import implied_probability

FLAG_NO_SHARP_ANCHOR = "no_sharp_anchor"
FLAG_THIN_MARKET = "thin_market"
FLAG_SINGLE_BOOK = "single_book"
FLAG_STALE_PRICES = "stale_prices"
FLAG_ARBITRAGE = "negative_hold"
FLAG_ONE_SIDED = "no_two_sided_book"
FLAG_PARTIAL_BOOKS = "partial_book_coverage"
FLAG_PLACEHOLDER_DROPPED = "placeholder_price_dropped"

# Books post prices like -100000 to mean "no action", not "99.9% likely".
# Seen live from MIDNITE and DRAFTKINGS on 2026-09-05. Treating them as real
# quotes drags a consensus to absurdity, so they are dropped and flagged.
MAX_CREDIBLE_PROB = 0.995
MIN_CREDIBLE_PROB = 0.0005


@dataclass(frozen=True)
class SideQuote:
    """The market's view of one side of one market."""

    side: str
    line: float | None
    consensus_price: int | None
    best_price: int | None
    best_book: str | None
    n_books: int
    probs: dict[str, float] = field(default_factory=dict)  # method -> vig-free prob

    @property
    def vig_free_prob(self) -> float | None:
        """Default published probability. Shin unless it was unavailable."""
        return self.probs.get("shin") or self.probs.get("multiplicative")


@dataclass(frozen=True)
class MarketConsensus:
    game_id: str
    market: str
    line: float | None
    as_of_utc: str
    n_books: int
    hold: float | None
    sides: tuple[SideQuote, ...]
    anchor: str  # "sharp" | "all_books" | "none"
    flags: tuple[str, ...]

    def side(self, name: str) -> SideQuote | None:
        for quote in self.sides:
            if quote.side == name:
                return quote
        return None


def _probability(american: Any) -> float | None:
    """Implied probability, or None if the price could not be parsed.

    ``implied_probability`` returns None rather than raising, so every call site
    has to decide what an unparseable price means. Here it means "drop that
    book", never "treat it as zero" -- a silently-zeroed price would drag a
    consensus toward the longshot without leaving a trace.
    """
    return implied_probability(american)


def _median_american(prices: Sequence[int]) -> int | None:
    """Median in *probability* space, then converted back.

    Taking a median of American odds directly is wrong: -110 and +110 are
    adjacent in probability but 220 apart on the American scale, and the
    discontinuity at +/-100 makes the arithmetic meaningless.

    Returns None when no price in the group could be parsed.
    """
    probs = sorted(p for p in (_probability(x) for x in prices) if p is not None)
    if not probs:
        return None
    return _prob_to_american(statistics.median(probs))


def _prob_to_american(prob: float) -> int:
    if not 0.0 < prob < 1.0:
        raise ValueError(f"Probability out of range: {prob}")
    if prob >= 0.5:
        return -int(round(prob / (1.0 - prob) * 100))
    return int(round((1.0 - prob) / prob * 100))


def _best(prices: Iterable[tuple[int, str]]) -> tuple[int | None, str | None]:
    """Best available price for a bettor: the lowest implied probability.

    Unparseable prices are dropped rather than sorted as if they were free
    money -- a None sorting first would name a broken quote as "best price".
    """
    scored = [
        (prob, price, book)
        for price, book in prices
        if (prob := _probability(price)) is not None
    ]
    if not scored:
        return (None, None)
    _, price, book = min(scored, key=lambda triple: triple[0])
    return (price, book)


def build_consensus(
    game_id: str,
    market: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: str,
    sharp_books: Sequence[str] = (),
    min_books_for_consensus: int = 3,
    min_sharp_books: int = 2,
) -> MarketConsensus | None:
    """Collapse many books' prices on one market into a fair-probability view.

    ``rows`` must already be restricted to a single (game, market, line) group
    and to pre-kickoff timestamps -- this function does no as-of filtering, so
    callers go through ``AsOfReader`` first.
    """
    quoted_rows = [r for r in rows if r.get("price_american") is not None and r.get("side")]
    if not quoted_rows:
        return None

    flags: list[str] = []
    priced = []
    for row in quoted_rows:
        prob = _probability(row["price_american"])
        if prob is None:
            continue
        if not MIN_CREDIBLE_PROB <= prob <= MAX_CREDIBLE_PROB:
            # A "no action" placeholder, not a market. Dropped, never averaged.
            continue
        priced.append(row)
    if len(priced) < len(quoted_rows):
        flags.append(FLAG_PLACEHOLDER_DROPPED)
    if not priced:
        return None

    sharp_set = {b.upper() for b in sharp_books}

    sharp_rows = [r for r in priced if str(r["book"]).upper() in sharp_set]
    sharp_book_count = len({str(r["book"]).upper() for r in sharp_rows})
    if sharp_book_count >= min_sharp_books:
        selected, anchor = sharp_rows, "sharp"
    else:
        selected, anchor = priced, "all_books"
        flags.append(FLAG_NO_SHARP_ANCHOR)

    by_side: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_side.setdefault(str(row["side"]).upper(), []).append(row)

    n_books = len({str(r["book"]).upper() for r in selected})
    if n_books < min_books_for_consensus:
        flags.append(FLAG_THIN_MARKET)
    if n_books == 1:
        flags.append(FLAG_SINGLE_BOOK)

    # A fair probability needs both sides; one-sided data cannot be devigged.
    if len(by_side) < 2:
        return MarketConsensus(
            game_id=game_id, market=market,
            line=_first_line(selected), as_of_utc=as_of_utc, n_books=n_books,
            hold=None, sides=(), anchor="none",
            flags=tuple([*flags, FLAG_THIN_MARKET] if FLAG_THIN_MARKET not in flags else flags),
        )

    side_names = sorted(by_side)

    # Devig each book against ITSELF, then aggregate the fair probabilities.
    #
    # The tempting shortcut -- median each side across books, then devig the two
    # medians -- is wrong, because the book sets behind each side differ. Seen
    # live on BALL@OSU: HOME priced by 2 books, AWAY by 4. Devigging those two
    # medians together mixes different books' margins, manufactures phantom
    # arbitrage, and produced a -18249 "consensus" on a market whose best real
    # price was -10000.
    per_book: dict[str, dict[str, float]] = {}
    for row in selected:
        book = str(row["book"]).upper()
        prob = _probability(row["price_american"])
        if prob is not None:
            per_book.setdefault(book, {})[str(row["side"]).upper()] = prob

    two_sided = {
        book: quotes for book, quotes in per_book.items()
        if len(quotes) == len(side_names) and len(quotes) >= 2
    }
    if not two_sided:
        return MarketConsensus(
            game_id=game_id, market=market, line=_first_line(selected),
            as_of_utc=as_of_utc, n_books=n_books, hold=None, sides=(),
            anchor="none",
            flags=tuple(dict.fromkeys([*flags, FLAG_ONE_SIDED, FLAG_THIN_MARKET])),
        )
    if len(two_sided) < len(per_book):
        flags.append(FLAG_PARTIAL_BOOKS)

    fair_by_method: dict[str, dict[str, list[float]]] = {m: {} for m in devig.METHODS}
    holds: list[float] = []
    for quotes in two_sided.values():
        ordered = [quotes[name] for name in side_names]
        try:
            book_probs = devig.devig_all(ordered)
        except DevigError:
            continue
        holds.append(devig.hold(ordered))
        for method, values in book_probs.items():
            for index, name in enumerate(side_names):
                fair_by_method[method].setdefault(name, []).append(values[index])

    if not holds:
        return MarketConsensus(
            game_id=game_id, market=market, line=_first_line(selected),
            as_of_utc=as_of_utc, n_books=n_books, hold=None, sides=(),
            anchor="none", flags=tuple(dict.fromkeys([*flags, FLAG_THIN_MARKET])),
        )

    book_hold = statistics.median(holds)
    if book_hold < 0:
        flags.append(FLAG_ARBITRAGE)

    # Median across books is taken per side, so the set can drift off 1.0;
    # renormalise so the published probabilities remain a distribution.
    all_probs: dict[str, list[float]] = {}
    for method, by_name in fair_by_method.items():
        if len(by_name) != len(side_names):
            continue
        medians = [statistics.median(by_name[name]) for name in side_names]
        total = math.fsum(medians)
        if total > 0:
            all_probs[method] = [value / total for value in medians]

    priced_sides = side_names
    consensus_prices: dict[str, int] = {}
    for name in side_names:
        # Consensus price is the median across the SAME two-sided book set, so
        # the displayed price and the fair probability describe one market.
        prices = [
            int(r["price_american"]) for r in by_side[name]
            if str(r["book"]).upper() in two_sided
        ]
        median_price = _median_american(prices)
        if median_price is not None:
            consensus_prices[name] = median_price

    sides = []
    for index, name in enumerate(priced_sides):
        best_price, best_book = _best(
            [(int(r["price_american"]), str(r["book"])) for r in by_side[name]]
        )
        sides.append(
            SideQuote(
                side=name,
                line=signed_display_line(by_side[name], name),
                consensus_price=consensus_prices[name],
                best_price=best_price,
                best_book=best_book,
                n_books=len({str(r["book"]).upper() for r in by_side[name]}),
                probs={method: values[index] for method, values in all_probs.items()},
            )
        )

    return MarketConsensus(
        game_id=game_id,
        market=market,
        line=_first_line(selected),
        as_of_utc=as_of_utc,
        n_books=n_books,
        hold=book_hold,
        sides=tuple(sides),
        anchor=anchor,
        flags=tuple(dict.fromkeys(flags)),
    )


def _first_line(rows: Sequence[Mapping[str, Any]]) -> float | None:
    for row in rows:
        if row.get("line") is not None:
            return float(row["line"])
    return None


def group_key(row: Mapping[str, Any]) -> tuple[str, str, float | None]:
    """The two sides of one market, keyed so they actually pair up.

    Grouping by line matters: -3.5 and -7.5 are different markets and devigging
    them together is nonsense. But the sides of a **spread** are stored with
    opposite signs -- HOME -13.5 against AWAY +13.5 -- so keying on the raw line
    puts them in different groups and they never pair.

    Measured before this fix: SPREAD showed 694 of 1594 consensus rows with
    negative hold (phantom arbitrage from mispaired ladder rungs) while ML
    showed 0 of 76 and TOTAL 0 of 3684, because those two already share a line
    value across sides. Spreads are therefore keyed on the absolute line.
    """
    line = row.get("line")
    value = float(line) if line is not None else None
    market_code = str(row["market"])
    if market_code == "SPREAD" and value is not None:
        value = abs(value)
    return (str(row["game_id"]), market_code, value)


def signed_display_line(rows: Sequence[Mapping[str, Any]], side: str) -> float | None:
    """The line as that side actually sees it, for display and storage.

    Grouping uses the absolute spread, but a stored row should read -13.5 for
    the favourite and +13.5 for the dog, not 13.5 for both.
    """
    for row in rows:
        if str(row.get("side", "")).upper() == side.upper() and row.get("line") is not None:
            return float(row["line"])
    return None


def line_movement(
    opening: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Open-to-current movement.

    ``rlm_basis`` is always ``line_only``: true reverse line movement needs
    ticket/money percentages, which are not available from any free source, so
    the inference is labelled rather than dressed up as the real thing.
    """
    if not opening or not current:
        return {"move_magnitude": None, "move_direction": None,
                "rlm_flag": False, "rlm_basis": "line_only"}

    open_line = opening.get("line")
    current_line = current.get("line")
    open_price = opening.get("price_american")
    current_price = current.get("price_american")

    magnitude = None
    direction = None
    if open_line is not None and current_line is not None:
        magnitude = float(current_line) - float(open_line)
        direction = "toward" if magnitude < 0 else ("away" if magnitude > 0 else "flat")

    # Line and price disagreeing is the only RLM signal available without
    # ticket counts, and it is weak. Labelled, never presented as confirmed.
    rlm = False
    if (
        magnitude is not None
        and open_price is not None
        and current_price is not None
        and magnitude != 0
    ):
        now_prob = _probability(int(current_price))
        then_prob = _probability(int(open_price))
        if now_prob is not None and then_prob is not None:
            price_shift = now_prob - then_prob
            rlm = (magnitude < 0 and price_shift < 0) or (magnitude > 0 and price_shift > 0)

    return {
        "open_line": open_line,
        "open_price": open_price,
        "current_line": current_line,
        "current_price": current_price,
        "move_magnitude": magnitude,
        "move_direction": direction,
        "rlm_flag": rlm,
        "rlm_basis": "line_only",
    }


def staleness_flag(minutes_before_kickoff: float, stale_after_minutes: float) -> list[str]:
    return [FLAG_STALE_PRICES] if minutes_before_kickoff > stale_after_minutes else []


def summarise_probability_spread(consensus: MarketConsensus, side: str) -> float | None:
    """How far apart the three devig methods land on this side.

    Reported because the methods disagree most on heavy favourites -- exactly
    the population the parlay product draws from. A wide spread means the
    "fair" probability is method-dependent and confidence should drop.
    """
    quote = consensus.side(side)
    if quote is None or len(quote.probs) < 2:
        return None
    values = list(quote.probs.values())
    return max(values) - min(values)


def is_finite_probability(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value) and 0.0 < value < 1.0
