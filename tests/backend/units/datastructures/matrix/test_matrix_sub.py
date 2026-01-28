# tests/backend/units/datastructures/matrix/test_matrix_sub.py
import pytest

from conftest import EXTERNAL_SPARSE_TYPES, EXTERNAL_DENSE_TYPES


from backend.datastructures import SparseMatrix, DenseMatrix


# Test-specific: Expected results for sub
SUB_EXPECTED = {
    ("sparse", "sparse"): SparseMatrix,
    ("sparse", "dense"): DenseMatrix,
    ("dense", "sparse"): DenseMatrix,
    ("dense", "dense"): DenseMatrix,
}


# Tests __sub__ for all combinations of internal and external types
@pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_sub_sparse(right_type, matrix_factory):
    left = matrix_factory("SparseMatrix")
    right = matrix_factory(right_type)
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("sparse", "sparse")])


@pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_sparse_sub_dense(right_type, matrix_factory):
    left = matrix_factory("SparseMatrix")
    right = matrix_factory(right_type)
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("sparse", "dense")])


@pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_sub_dense(right_type, matrix_factory):
    left = matrix_factory("DenseMatrix")
    right = matrix_factory(right_type)
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("dense", "dense")])


@pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_dense_sub_sparse(right_type, matrix_factory):
    left = matrix_factory("DenseMatrix")
    right = matrix_factory(right_type)
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("dense", "sparse")])


# Tests __rsub__ for all combinations of internal and external types
@pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_rsub_sparse(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("SparseMatrix")
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("sparse", "sparse")])


@pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_rsub_sparse(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("SparseMatrix")
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("dense", "sparse")])


@pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_rsub_dense(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("DenseMatrix")
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("dense", "dense")])


@pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_rsub_dense(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("DenseMatrix")
    result = left - right
    assert isinstance(result, SUB_EXPECTED[("sparse", "dense")])
