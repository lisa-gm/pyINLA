# tests/backend/units/datastructures/matrix/test_matrix_matmul.py
import pytest

from conftest import EXTERNAL_SPARSE_TYPES, EXTERNAL_DENSE_TYPES


from backend.datastructures import SparseMatrix, DenseMatrix


# Test-specific: Expected results for matmul
MATMUL_EXPECTED = {
    ("sparse", "sparse"): SparseMatrix,
    ("sparse", "dense"): DenseMatrix,
    ("dense", "sparse"): DenseMatrix,
    ("dense", "dense"): DenseMatrix,
}


# Tests __matmul__ for all combinations of internal and external types
@pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_matmul_sparse(right_type, matrix_factory):
    left = matrix_factory("SparseMatrix")
    right = matrix_factory(right_type)
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("sparse", "sparse")])


@pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_sparse_matmul_dense(right_type, matrix_factory):
    left = matrix_factory("SparseMatrix")
    right = matrix_factory(right_type)
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("sparse", "dense")])


@pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_matmul_dense(right_type, matrix_factory):
    left = matrix_factory("DenseMatrix")
    right = matrix_factory(right_type)
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("dense", "dense")])


@pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_dense_matmul_sparse(right_type, matrix_factory):
    left = matrix_factory("DenseMatrix")
    right = matrix_factory(right_type)
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("dense", "sparse")])


# Tests __rmatmul__ for all combinations of internal and external types
@pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_rmatmul_sparse(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("SparseMatrix")
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("sparse", "sparse")])


@pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_rmatmul_dense(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("DenseMatrix")
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("sparse", "dense")])


@pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_rmatmul_dense(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("DenseMatrix")
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("dense", "dense")])


@pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_rmatmul_sparse(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("SparseMatrix")
    result = left @ right
    assert isinstance(result, MATMUL_EXPECTED[("dense", "sparse")])
