# tests/backend/units/datastructures/matrix/test_matrix_sub.py
import numpy as np
import pytest

from dalia.backend.config import cupy_version
from dalia.backend.datastructures import DenseMatrix, SparseMatrix

if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp

from tests.backend import ATOLS, RTOLS

from .conftest import EXTERNAL_DENSE_TYPES, EXTERNAL_SPARSE_TYPES, INTERNAL_DEVICE_TYPES

# Test-specific: Expected results for sub
SUB_EXPECTED = {
    ("sparse", "sparse"): SparseMatrix,
    ("sparse", "dense"): DenseMatrix,
    ("dense", "sparse"): DenseMatrix,
    ("dense", "dense"): DenseMatrix,
}


class TestSubReturnTypes:
    """Verify dispatch returns correct Matrix subclass"""

    # Tests __sub__ for all combinations of internal and external types
    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_sub_sparse(self, device, right_type, matrix_factory):
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_sub_dense(self, right_type, device, matrix_factory):
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("sparse", "dense")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_sub_dense(self, right_type, device, matrix_factory):
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_sub_sparse(self, right_type, device, matrix_factory):
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("dense", "sparse")])

    # Tests __rsub__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_rsub_sparse(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_rsub_sparse(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("dense", "sparse")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_rsub_dense(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_rsub_dense(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left - right
        assert isinstance(result, SUB_EXPECTED[("sparse", "dense")])


class TestSubCorrectness:
    """Verify correctness of __sub__ and __rsub__ implementations"""

    # Tests __sub__ for all combinations of internal and external types
    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_sub_sparse(self, right_type, device, matrix_factory):
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        if right_type == "cupy_csr" or right_type == "cupy_csc" or right_type == "cupy_coo":
            reference = left.toarray() - cp.asnumpy(right.toarray())
        else:
            reference = left.toarray() - right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_sub_dense(self, right_type, device, matrix_factory):
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        if right_type == "cupy":
            right = cp.asnumpy(right)
        reference = (
            left.toarray() - right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() - right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_sub_dense(self, right_type, device, matrix_factory):
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        if right_type == "cupy":
            right = cp.asnumpy(right)
        reference = (
            left.toarray() - right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() - right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_sub_sparse(self, right_type, device, matrix_factory):
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left - right
        if right_type == "cupy_csr" or right_type == "cupy_csc" or right_type == "cupy_coo":
            reference = left.toarray() - cp.asnumpy(right.toarray())
        else:
            reference = left.toarray() - right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    # Tests __rsub__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_rsub_sparse(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left - right
        if left_type == "cupy_csr" or left_type == "cupy_csc" or left_type == "cupy_coo":
            reference = cp.asnumpy(left.toarray()) - right.toarray()
        else:
            reference = left.toarray() - right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_rsub_sparse(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left - right
        if left_type == "cupy":
            left = cp.asnumpy(left)
        reference = (
            left.toarray() - right.toarray()
            if hasattr(left, "toarray")
            else left - right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_dense_rsub_dense(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left - right
        if left_type == "cupy":
            left = cp.asnumpy(left)
        reference = (
            left.toarray() - right.toarray()
            if hasattr(left, "toarray")
            else left - right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    def test_sparse_rsub_dense(self, left_type, device, matrix_factory):
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left - right
        if left_type == "cupy_csr" or left_type == "cupy_csc" or left_type == "cupy_coo":
            reference = cp.asnumpy(left.toarray()) - right.toarray()
        else:
            reference = left.toarray() - right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )
