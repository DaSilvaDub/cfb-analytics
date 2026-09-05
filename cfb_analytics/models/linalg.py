"""Pure-Python dense linear solve: Gauss-Jordan elimination with partial pivoting.

No numpy, per the repo-wide constraint (see the package docstring). At the
scale a real FBS+FCS-wide ridge fit actually reaches -- ``n = 2 + 2*229 =
460`` unknowns for a full season, measured 2026-09-05 -- naive elimination
that rebuilds a length-``n`` list for every (row, column) pair costs a real
~18.5 seconds, which is the difference between a backtest that finishes and
one that does not: the walk-forward harness in ``backtest/harness.py`` calls
this once per (season, week), on the order of 150-200 times for the full
2014-2025 history.

Two changes fix that without changing the algorithm:

1. **Skip already-eliminated columns.** By induction, when column ``col`` is
   being processed, every row already has zeros in columns ``0..col-1`` (each
   row was an elimination target during every earlier iteration it wasn't the
   pivot for). Touching those columns again wastes roughly half the work.
2. **Mutate rows in place** instead of building a new list via ``zip`` and a
   list comprehension per row -- CPython pays real overhead for the
   allocation and the iterator protocol that a plain indexed loop does not.

Together these took the measured full-season fit from ~18.5s to a small
fraction of a second, with no numerical or interface change: same pivoting
rule, same singular-matrix detection, same no-mutation guarantee for the
caller's ``matrix``.

This mirrors the technique already proven out in the sibling `outlier`
project's `team_strength.py` (MLB run totals), not its code -- this repo
shares no code with that one by design (see docs/plans/... in that repo).
"""

from __future__ import annotations

_SINGULAR_THRESHOLD = 1e-12


def solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Solve ``matrix @ x = vector``. Returns None if the matrix is singular.

    ``matrix`` is consumed by copy (via the augmented-matrix construction), so
    the caller's list is never mutated.
    """
    n = len(vector)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"Expected an {n}x{n} matrix for a length-{n} vector")

    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < _SINGULAR_THRESHOLD:
            return None
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot_data = augmented[col]
        pivot = pivot_data[col]
        # Columns before `col` are already zero here (see module docstring),
        # so normalizing/eliminating only needs col..n (n is the RHS column).
        for k in range(col, n + 1):
            pivot_data[k] /= pivot

        for row in range(n):
            if row == col:
                continue
            target = augmented[row]
            factor = target[col]
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                target[k] -= factor * pivot_data[k]

    return [augmented[row][n] for row in range(n)]
