# tests/backend/units/datastructures/matrix/test_matrix_matmul.py
import numpy as np
import pytest

from dalia.backend.config import cupy_version, set_memory_regime
from dalia.backend.datastructures import DenseMatrix, SparseMatrix

if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp
from tests.backend import ATOLS, RTOLS

from .conftest import EXTERNAL_DENSE_TYPES, EXTERNAL_SPARSE_TYPES, INTERNAL_DEVICE_TYPES, MEMORY_REGIMES

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
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_matmul_sparse(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_matmul_dense(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "dense")])

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_matmul_dense(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_matmul_sparse(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "sparse")])

    # Tests __rmatmul__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_rmatmul_sparse(self, device, left_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "sparse")])

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_rmatmul_dense(self, device, left_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("sparse", "dense")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_rmatmul_dense(self, device, left_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "dense")])

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_rmatmul_sparse(self, device, left_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left @ right
        assert isinstance(result, MATMUL_EXPECTED[("dense", "sparse")])


class TestMatmulResults:
    """Verify matmul produces correct numerical results"""

    # Tests __matmul__ for all combinations of internal and external types
    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_matmul_sparse(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        if right_type == "cupy_csr" or right_type == "cupy_csc" or right_type == "cupy_coo":
            reference = left.toarray() @ cp.asnumpy(right.toarray())
        else:
            reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_matmul_dense(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("SparseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        if right_type == "cupy":
            right = cp.asnumpy(right)
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() @ right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_matmul_dense(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        if right_type == "cupy":
            right = cp.asnumpy(right)
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(right, "toarray")
            else left.toarray() @ right
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_matmul_sparse(self, device, right_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory("DenseMatrix", device=device)
        right = matrix_factory(right_type)
        result = left @ right
        if right_type == "cupy_csr" or right_type == "cupy_csc" or right_type == "cupy_coo":
            reference = left.toarray() @ cp.asnumpy(right.toarray())
        else:
            reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    # Tests __rmatmul__ for all combinations of internal and external types
    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_rmatmul_sparse(self, left_type, device, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left @ right
        if left_type == "cupy_csr" or left_type == "cupy_csc" or left_type == "cupy_coo":
            reference = cp.asnumpy(left.toarray()) @ right.toarray()
        else:
            reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_sparse_rmatmul_dense(self, left_type, device, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left @ right
        if left_type == "cupy_csr" or left_type == "cupy_csc" or left_type == "cupy_coo":
            reference = cp.asnumpy(left.toarray()) @ right.toarray()
        else:
            reference = left.toarray() @ right.toarray()
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_rmatmul_dense(self, device, left_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("DenseMatrix", device=device)
        result = left @ right
        if left_type == "cupy":
            left = cp.asnumpy(left)
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(left, "toarray")
            else left @ right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )

    @pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
    @pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("memory_regime", MEMORY_REGIMES)
    def test_dense_rmatmul_sparse(self, device, left_type, memory_regime, matrix_factory):
        set_memory_regime(memory_regime)
        left = matrix_factory(left_type)
        right = matrix_factory("SparseMatrix", device=device)
        result = left @ right
        if left_type == "cupy":
            left = cp.asnumpy(left)
        reference = (
            left.toarray() @ right.toarray()
            if hasattr(left, "toarray")
            else left @ right.toarray()
        )
        assert np.allclose(
            result.toarray(), reference, rtol=RTOLS["strict"], atol=ATOLS["strict"]
        )
