# tests/backend/units/datastructures/matrix/test_matrix_matmul.py
import numpy as np
import pytest

from dalia.backend.datastructures import DenseMatrix, SparseMatrix
from tests.backend import ATOLS, RTOLS

from .conftest import EXTERNAL_DENSE_TYPES, EXTERNAL_SPARSE_TYPES

# Test-specific: Expected results for matmul
MATMUL_EXPECTED = {
    ("sparse", "sparse"): SparseMatrix,
    ("sparse", "dense"): DenseMatrix,
    ("dense", "sparse"): DenseMatrix,
    ("dense", "dense"): DenseMatrix,
}


class TestMatmulReturnTypes:
    """Verify dispatch returns correct Matrix subclass"""

    # Tests __matmul__ for all combinations of internal and external types
    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_matmul_sparse(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_sparse_matmul_dense(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "dense")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_matmul_dense(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_dense_matmul_sparse(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "sparse")])

    # Tests __rmatmul__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_rmatmul_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_rmatmul_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "dense")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_rmatmul_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_rmatmul_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "sparse")])


class TestMatmulResults:
    """Verify matmul produces correct numerical results"""

    # Tests __matmul__ for all combinations of internal and external types
    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_matmul_sparse(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_sparse_matmul_dense(self, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() @ right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_matmul_dense(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() @ right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_dense_matmul_sparse(self, right_type, matrix_factory):
        left = matrix_factory("DenseMatrix")
        right = matrix_factory(right_type)
        result = left @ right
        reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    # Tests __rmatmul__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_rmatmul_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left @ right
        reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    def test_sparse_rmatmul_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left @ right
        reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_rmatmul_dense(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix")
        result = left @ right
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(left, "toarray")
            else left @ right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    def test_dense_rmatmul_sparse(self, left_type, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix")
        result = left @ right
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(left, "toarray")
            else left @ right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )
