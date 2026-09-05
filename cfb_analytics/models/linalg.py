"""Pure-Python dense linear solve: Gaussian elimination with partial pivoting.

No numpy, per the repo-wide constraint (see the package docstring). At the
scale this pipeline needs -- roughly 2 + 2*134 ~= 270 unknowns for an
FBS-wide ridge fit -- a dense O(n^3) elimination is a few hundred milliseconds
in pure Python, so there is no need for an iterative method.

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

        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[col], strict=True)
            ]

    return [augmented[row][n] for row in range(n)]
