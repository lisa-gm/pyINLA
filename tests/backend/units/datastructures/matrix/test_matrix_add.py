# tests/backend/units/datastructures/matrix/test_matrix_add.py

import numpy as np
import pytest

from dalia.backend.datastructures import DenseMatrix, SparseMatrix
from tests.backend import ATOLS, RTOLS

from .conftest import EXTERNAL_DENSE_TYPES, EXTERNAL_SPARSE_TYPES

# Test-specific: Expected results for add
ADD_EXPECTED = {
    ("sparse", "sparse"): SparseMatrix,
    ("sparse", "dense"): DenseMatrix,
    ("dense", "sparse"): DenseMatrix,
    ("dense", "dense"): DenseMatrix,
}


class TestAddReturnTypes:
    """Verify dispatch returns correct Matrix subclass"""

    # Tests __add__ for all combinations of internal and external types
    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_add_sparse(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_sparse_add_dense(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("sparse", "dense")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_add_dense(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_dense_add_sparse(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("dense", "sparse")])

    # Tests __radd__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_radd_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_radd_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("dense", "sparse")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_radd_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_radd_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left + right
        assert isinstance(result, ADD_EXPECTED[("sparse", "dense")])


class TestAddCorrectness:
    """Verify add produces correct numerical results"""

    # Tests __add__ for all combinations of internal and external types
    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_add_sparse(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        reference = left.toarray() + right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_sparse_add_dense(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        reference = (
            left.toarray() + right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() + right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_add_dense(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        reference = (
            left.toarray() + right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() + right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_dense_add_sparse(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left + right
        reference = left.toarray() + right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    # Tests __radd__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_radd_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left + right
        reference = left.toarray() + right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_radd_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left + right
        reference = (
            left.toarray() + right.toarray()
            if hasattr(left, "toarray")
            else left + right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_radd_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left + right
        reference = (
            left.toarray() + right.toarray()
            if hasattr(left, "toarray")
            else left + right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_radd_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left + right
        reference = left.toarray() + right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )
