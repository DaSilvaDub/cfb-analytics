from __future__ import annotations

import pytest

from cfb_analytics.models.linalg import solve


class TestSolve:
    def test_identity_matrix_returns_the_vector_unchanged(self):
        identity = [[1.0, 0.0], [0.0, 1.0]]
        assert solve(identity, [3.0, 5.0]) == pytest.approx([3.0, 5.0])

    def test_solves_a_known_2x2_system(self):
        # 2x + y = 5; x + 3y = 10  =>  x=1, y=3
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        result = solve(matrix, [5.0, 10.0])
        assert result == pytest.approx([1.0, 3.0])

    def test_solves_a_known_3x3_system(self):
        # x + y + z = 6; 2y + 5z = -4; 2x + 5y - z = 27  => x=5, y=3, z=-2
        matrix = [[1.0, 1.0, 1.0], [0.0, 2.0, 5.0], [2.0, 5.0, -1.0]]
        result = solve(matrix, [6.0, -4.0, 27.0])
        assert result == pytest.approx([5.0, 3.0, -2.0])

    def test_singular_matrix_returns_none(self):
        # Second row is a multiple of the first -> rank-deficient.
        matrix = [[1.0, 2.0], [2.0, 4.0]]
        assert solve(matrix, [1.0, 2.0]) is None

    def test_does_not_mutate_the_input_matrix(self):
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        original = [row[:] for row in matrix]
        solve(matrix, [5.0, 10.0])
        assert matrix == original

    def test_requires_partial_pivoting_to_avoid_a_zero_pivot(self):
        # A naive no-pivot elimination would divide by zero on matrix[0][0].
        matrix = [[0.0, 1.0], [1.0, 1.0]]
        result = solve(matrix, [2.0, 3.0])
        assert result == pytest.approx([1.0, 2.0])

    def test_mismatched_dimensions_raise(self):
        with pytest.raises(ValueError):
            solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0, 3.0])
