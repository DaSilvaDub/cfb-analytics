"""Remove the bookmaker's margin from a set of quoted prices.

Multi-method devigging, all pure stdlib. They disagree most where this pipeline cares
most -- on heavy favourites, where the choice moves the fair probability by
1-3 percentage points, which is the difference between a CORE leg and an AVOID.
So multiple methods are computed and stored, and the backtest picks the winner on
out-of-sample log loss rather than the author picking one on taste.

* **multiplicative** (proportional): p_i = q_i / sum(q). Simple, and assumes the
  margin is spread proportionally. Known to under-price favourites, because in
  practice books load more margin onto longshots.
* **additive** (equal margin): p_i = q_i - (sum(q) - 1) / n. Assumes an equal
  margin deduction across all outcomes. Fails gracefully if the deducted margin
  exceeds a longshot's quoted probability. Verified 2026-09-06: for a
  two-outcome market this is algebraically identical to `shin` to within
  float noise (<= 1.8e-13 over 20k random books) -- the equal-margin and
  informed-money models only diverge once there are 3+ outcomes. Since every
  market this pipeline builds is two-sided, `additive` is excluded from the
  default `METHODS` set below to avoid scoring the same estimator twice; it
  stays available via `devig(quoted, "additive")` or `CANONICAL_METHODS` for
  a future 3-way market (e.g. soccer draw lines).
* **shin**: models the margin as compensation for informed bettors (Shin 1993).
  Usually lands between multiplicative and power, and is the most defensible
  default for two-outcome markets.
* **power**: p_i = q_i^k with k solved so the probabilities sum to 1. Applies the
  most correction to longshots.
* **odds_ratio** (Keith Cheung 2015 / log-odds): p_i = q_i / (q_i + c * (1 - q_i))
  with c solved so the probabilities sum to 1. Monotonic in c; models bookmaker
  margin as a distortion on the odds scale. Computed in `METHODS` for
  comparison, but not yet in `STORED_METHODS`: `market_consensus` has no
  `prob_odds_ratio` column, so a value computed here does not currently
  survive to the database. Add the column (plus a schema migration in db.py
  and the INSERT in features/build_market.py) before moving it into
  `STORED_METHODS`.

Per-Book Devigging Architecture:
Per-book devigging removes each sportsbook's margin against itself before any
cross-book aggregation is performed (e.g. via median or mean). Aggregating
raw odds across books before devigging mixes differing book sets and produces
severely distorted consensus probabilities (e.g. BALL@OSU regression).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from cfb_analytics.errors import DevigError

CANONICAL_METHODS = ("multiplicative", "additive", "shin", "power", "odds_ratio")
# additive is deliberately excluded here: it is a duplicate of shin for every
# two-outcome market this pipeline builds (see module docstring). Pass
# CANONICAL_METHODS explicitly to reach it.
METHODS = ("multiplicative", "shin", "power", "odds_ratio")
# odds_ratio is deliberately excluded here: market_consensus has no
# prob_odds_ratio column yet, so it would be computed and discarded rather
# than backtested. See module docstring.
STORED_METHODS = ("multiplicative", "shin", "power")

_TOLERANCE = 1e-12
_MAX_ITERATIONS = 200

METHOD_ALIASES: dict[str, str] = {
    "proportional": "multiplicative",
    "mult": "multiplicative",
    "equal_margin": "additive",
    "linear": "additive",
    "add": "additive",
    "log_odds": "odds_ratio",
    "log-odds": "odds_ratio",
    "logistic": "odds_ratio",
    "cheung": "odds_ratio",
    "odds-ratio": "odds_ratio",
    "or": "odds_ratio",
    "geometric": "power",
    "exp": "power",
}


def overround(quoted: Sequence[float]) -> float:
    """Sum of quoted (vig-inclusive) probabilities. 1.0 means no margin."""
    return math.fsum(quoted)


def hold(quoted: Sequence[float]) -> float:
    """The book's margin as a fraction of the total market. Negative = arb."""
    total = overround(quoted)
    if total <= 0:
        raise DevigError("Quoted probabilities must sum to a positive number")
    return (total - 1.0) / total


def _validate(quoted: Sequence[float]) -> list[float]:
    if quoted is None:
        raise DevigError("Quoted probabilities cannot be None")
    try:
        values = [float(q) for q in quoted]
    except (TypeError, ValueError) as exc:
        raise DevigError("Quoted probabilities must be convertible to float") from exc
    if len(values) < 2:
        raise DevigError("Devigging needs at least two mutually exclusive outcomes")
    if any(not math.isfinite(q) for q in values):
        raise DevigError("Quoted probabilities must be finite numbers")
    if any(q <= 0.0 for q in values):
        raise DevigError("Quoted probabilities must all be positive")
    if any(q >= 1.0 for q in values):
        # A single quote at or above 1.0 implies a certainty; the market is
        # malformed or the price was misparsed. Refuse rather than emit a
        # confident-looking number.
        raise DevigError("A quoted probability was >= 1.0, which is not a real price")
    return values


def multiplicative(quoted: Sequence[float]) -> list[float]:
    values = _validate(quoted)
    total = math.fsum(values)
    return [q / total for q in values]


def additive(quoted: Sequence[float]) -> list[float]:
    """Equal margin: subtract an equal share of the overround from each outcome.

    p_i = q_i - (sum(q) - 1) / n

    Assumes the bookmaker adds equal margin to each outcome regardless of odds.
    If the margin per outcome exceeds any quoted probability (e.g. on longshots
    in heavy-vig markets), additive devig produces a non-positive probability
    and raises DevigError.
    """
    values = _validate(quoted)
    total = math.fsum(values)
    if abs(total - 1.0) < _TOLERANCE:
        return list(values)
    n = len(values)
    margin_per_outcome = (total - 1.0) / n
    result: list[float] = []
    for q in values:
        p = q - margin_per_outcome
        if p <= 0.0:
            raise DevigError(
                f"Additive devig produced non-positive probability ({p:.6f}) "
                f"for quote {q:.6f}; market margin exceeds longshot quote"
            )
        result.append(p)
    return _renormalise(result)


def power(quoted: Sequence[float]) -> list[float]:
    """Solve sum(q_i^k) = 1 for k > 0.

    sum(q_i^k) is strictly decreasing in k for 0 < q_i < 1, so the root is
    unique and bisection is safe.
    """
    values = _validate(quoted)
    total = math.fsum(values)
    if abs(total - 1.0) < _TOLERANCE:
        return list(values)

    def total_at(k: float) -> float:
        return math.fsum(q**k for q in values)

    if total > 1.0:
        low, high = 1.0, 2.0
        while total_at(high) > 1.0:
            high *= 2.0
            if high > 1e12:
                raise DevigError("Power devig failed to bracket a solution")
    else:
        low, high = 0.5, 1.0
        while total_at(low) < 1.0:
            low /= 2.0
            if low < 1e-12:
                raise DevigError("Power devig failed to bracket a solution")

    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        if total_at(mid) > 1.0:
            low = mid
        else:
            high = mid
        if high - low < _TOLERANCE:
            break
    k = (low + high) / 2.0
    result = [q**k for q in values]
    return _renormalise(result)


def shin(quoted: Sequence[float]) -> list[float]:
    """Shin (1993): margin as compensation for a proportion z of informed money.

    p_i = [sqrt(z^2 + 4(1-z) q_i^2 / S) - z] / (2(1-z)),  S = sum(q)

    Numerically rationalized as:
    p_i = [2 q_i^2 / S] / [sqrt(z^2 + 4(1-z) q_i^2 / S) + z]

    This algebraic reformulation avoids catastrophic cancellation (subtracting
    near-equal floats) and avoids division by zero as z -> 1.
    sum(p_i) is strictly decreasing in z over [0, 1], with sum(p_i(0)) = sqrt(S) > 1
    and sum(p_i(1)) = sum(q_i^2) / S < 1, guaranteeing a unique root found via
    bisection over [0, 1].
    """
    values = _validate(quoted)
    total = math.fsum(values)
    if abs(total - 1.0) < _TOLERANCE:
        return list(values)
    if total < 1.0:
        # Negative margin: no informed-money interpretation exists. Fall back
        # rather than invent a z, and let the caller see a normalised book.
        return multiplicative(values)

    def implied_prob(q: float, z: float) -> float:
        radicand = z * z + 4.0 * (1.0 - z) * (q * q / total)
        return (2.0 * q * q / total) / (math.sqrt(radicand) + z)

    def implied_sum(z: float) -> float:
        return math.fsum(implied_prob(q, z) for q in values)

    low, high = 0.0, 1.0
    if implied_sum(low) <= 1.0:
        return multiplicative(values)
    if implied_sum(high) >= 1.0:
        return multiplicative(values)

    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        if implied_sum(mid) > 1.0:
            low = mid
        else:
            high = mid
        if high - low < _TOLERANCE:
            break
    z = (low + high) / 2.0
    result = [implied_prob(q, z) for q in values]
    return _renormalise(result)


def odds_ratio(quoted: Sequence[float]) -> list[float]:
    """Odds ratio (Keith Cheung 2015 / log-odds / proportional odds).

    Solves for constant odds ratio multiplier c > 0 such that:
        p_i = q_i / (q_i + c * (1 - q_i))
    and sum(p_i) = 1.

    Each term is strictly monotonically decreasing in c for 0 < q_i < 1, so the
    sum is strictly decreasing from n (as c -> 0) to 0 (as c -> inf), guaranteeing
    a unique root found via bisection.
    """
    values = _validate(quoted)
    total = math.fsum(values)
    if abs(total - 1.0) < _TOLERANCE:
        return list(values)

    def total_at(c: float) -> float:
        return math.fsum(q / (q + c * (1.0 - q)) for q in values)

    if total > 1.0:
        low, high = 1.0, 2.0
        while total_at(high) > 1.0:
            high *= 2.0
            if high > 1e15:
                raise DevigError("Odds ratio devig failed to bracket a solution")
    else:
        low, high = 0.5, 1.0
        while total_at(low) < 1.0:
            low /= 2.0
            if low < 1e-15:
                raise DevigError("Odds ratio devig failed to bracket a solution")

    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        if total_at(mid) > 1.0:
            low = mid
        else:
            high = mid
        if high - low < _TOLERANCE:
            break

    c = (low + high) / 2.0
    result = [q / (q + c * (1.0 - q)) for q in values]
    return _renormalise(result)


_DISPATCH = {
    "multiplicative": multiplicative,
    "additive": additive,
    "shin": shin,
    "power": power,
    "odds_ratio": odds_ratio,
}


def _renormalise(values: list[float]) -> list[float]:
    """Absorb bisection residue so the result sums to exactly 1."""
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise DevigError("Devig produced non-positive or non-finite probabilities")
    total = math.fsum(values)
    if total <= 0.0 or not math.isfinite(total):
        raise DevigError("Devig produced a non-positive or non-finite total")
    return [v / total for v in values]


def devig(quoted: Sequence[float], method: str = "multiplicative") -> list[float]:
    if not isinstance(method, str):
        raise DevigError(f"Devig method must be a string, got {type(method).__name__}")
    m_clean = method.lower().strip()
    canonical = METHOD_ALIASES.get(m_clean, m_clean)
    if canonical not in _DISPATCH:
        raise DevigError(f"Unknown devig method {method!r}; expected one of {CANONICAL_METHODS}")
    return _DISPATCH[canonical](quoted)


def devig_all(
    quoted: Sequence[float],
    methods: Sequence[str] | None = None,
    *,
    allow_failures: bool = False,
) -> dict[str, list[float]]:
    """Every requested method (defaults to METHODS), computed for backtesting and comparison.

    If allow_failures is True, methods that encounter DevigError (such as additive
    on an extreme longshot in a 3+ outcome market -- pass methods=CANONICAL_METHODS
    to include it) are omitted rather than raising DevigError.
    """
    target_methods = methods if methods is not None else METHODS
    results: dict[str, list[float]] = {}
    for method in target_methods:
        try:
            results[method] = devig(quoted, method)
        except DevigError:
            if not allow_failures:
                raise
    return results


def devig_per_book(
    book_quotes: Mapping[str, Sequence[float]],
    method: str = "multiplicative",
) -> dict[str, list[float]]:
    """Devig each book against itself before any cross-book aggregation.

    Core pipeline invariant: books must never be combined before devigging.
    Each book's overround is removed using that book's own prices. Books that
    fail validation or devigging are excluded.
    """
    if book_quotes is None:
        raise DevigError("Book quotes cannot be None")
    if not book_quotes:
        return {}
    fair_by_book: dict[str, list[float]] = {}
    for book, quotes in book_quotes.items():
        try:
            fair_by_book[book] = devig(quotes, method)
        except DevigError:
            continue
    return fair_by_book


def devig_per_book_all(
    book_quotes: Mapping[str, Sequence[float]],
    methods: Sequence[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Devig each book against itself across all requested methods.

    Returns {book: {method: [fair_probs]}}.
    """
    if book_quotes is None:
        raise DevigError("Book quotes cannot be None")
    if not book_quotes:
        return {}
    target_methods = methods if methods is not None else METHODS
    results: dict[str, dict[str, list[float]]] = {}
    for book, quotes in book_quotes.items():
        book_res: dict[str, list[float]] = {}
        for m in target_methods:
            try:
                book_res[m] = devig(quotes, m)
            except DevigError:
                continue
        if book_res:
            results[book] = book_res
    return results


def aggregate_probabilities(
    probabilities_by_book: Sequence[Sequence[float]],
    aggregator: str = "median",
) -> list[float]:
    """Aggregate per-book fair probabilities across outcomes and renormalise.

    Args:
        probabilities_by_book: Sequence of fair probability vectors (one per book),
            each of the same length N >= 2.
        aggregator: 'median' (standard, robust to rogue books) or 'mean'.

    Returns:
        Consensus probability distribution summing to 1.0.
    """
    if not probabilities_by_book:
        raise DevigError("Cannot aggregate empty set of book probabilities")

    agg_str = aggregator.lower().strip() if isinstance(aggregator, str) else ""
    if agg_str == "median":
        agg_fn = statistics.median
    elif agg_str == "mean":
        agg_fn = statistics.mean
    else:
        raise DevigError(f"Unknown aggregator {aggregator!r}; expected 'median' or 'mean'")

    try:
        first_row = [float(p) for p in probabilities_by_book[0]]
    except (TypeError, ValueError) as exc:
        raise DevigError("Probabilities must be convertible to float") from exc

    n_outcomes = len(first_row)
    if n_outcomes < 2:
        raise DevigError("Probabilities must contain at least two outcomes")

    clean_rows: list[list[float]] = []
    for idx, row in enumerate(probabilities_by_book):
        try:
            float_row = [float(p) for p in row]
        except (TypeError, ValueError) as exc:
            raise DevigError(f"Book {idx} probabilities must be convertible to float") from exc
        if len(float_row) != n_outcomes:
            raise DevigError("All books must quote the same number of outcomes")
        if any(not math.isfinite(p) or p <= 0.0 or p >= 1.0 for p in float_row):
            raise DevigError("Book probabilities must be finite numbers in (0, 1)")
        clean_rows.append(float_row)

    aggregated = [
        agg_fn([row[j] for row in clean_rows])
        for j in range(n_outcomes)
    ]
    return _renormalise(aggregated)


def consensus_probabilities(
    book_quotes: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    method: str = "multiplicative",
    aggregator: str = "median",
) -> list[float]:
    """Devig each book against itself, then aggregate across books to consensus.

    Enforces the architecture design principle: never average raw odds across
    books before devigging; devig each book's market first, then aggregate.
    """
    if book_quotes is None:
        raise DevigError("Book quotes cannot be None")
    if isinstance(book_quotes, Mapping):
        quotes_list = list(book_quotes.values())
    else:
        quotes_list = list(book_quotes)

    fair_books: list[list[float]] = []
    for quotes in quotes_list:
        try:
            fair_books.append(devig(quotes, method))
        except DevigError:
            continue

    if not fair_books:
        raise DevigError("No valid book quotes could be devigged")

    return aggregate_probabilities(fair_books, aggregator=aggregator)


def consensus_probabilities_all(
    book_quotes: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    methods: Sequence[str] | None = None,
    aggregator: str = "median",
) -> dict[str, list[float]]:
    """Compute consensus fair probabilities for each devigging method."""
    target_methods = methods if methods is not None else METHODS
    results: dict[str, list[float]] = {}
    for m in target_methods:
        try:
            results[m] = consensus_probabilities(book_quotes, method=m, aggregator=aggregator)
        except DevigError:
            continue
    if not results:
        raise DevigError("No devig method produced consensus probabilities")
    return results


def method_disagreement(
    probabilities_by_method: Mapping[str, Sequence[float]],
) -> list[float]:
    """Calculate the max spread across methods for each outcome.

    High spread indicates method-dependent uncertainty (e.g. heavy favourites).
    """
    if not probabilities_by_method:
        return []
    sample = next(iter(probabilities_by_method.values()))
    try:
        sample_floats = [float(p) for p in sample]
    except (TypeError, ValueError) as exc:
        raise DevigError("Probabilities must be convertible to float") from exc
    n_outcomes = len(sample_floats)
    if n_outcomes < 2:
        raise DevigError("Probabilities must contain at least two outcomes")

    clean_by_method: dict[str, list[float]] = {}
    for method, probs in probabilities_by_method.items():
        try:
            float_probs = [float(p) for p in probs]
        except (TypeError, ValueError) as exc:
            raise DevigError(f"Method {method!r} probabilities must be float") from exc
        if len(float_probs) != n_outcomes:
            raise DevigError("All methods must produce identical outcome counts")
        if any(not math.isfinite(p) for p in float_probs):
            raise DevigError("Probabilities must be finite numbers")
        clean_by_method[method] = float_probs

    if len(clean_by_method) < 2:
        return [0.0] * n_outcomes

    return [
        max(clean_by_method[m][j] for m in clean_by_method)
        - min(clean_by_method[m][j] for m in clean_by_method)
        for j in range(n_outcomes)
    ]
