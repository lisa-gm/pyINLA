# tests/backend/units/datastructures/matrix/test_matrix_add.py
import pytest

from .conftest import EXTERNAL_SPARSE_TYPES, EXTERNAL_DENSE_TYPES

from dalia.backend.datastructures import SparseMatrix, DenseMatrix


# Test-specific: Expected results for add
ADD_EXPECTED = {
    ("sparse", "sparse"): SparseMatrix,
    ("sparse", "dense"): DenseMatrix,
    ("dense", "sparse"): DenseMatrix,
    ("dense", "dense"): DenseMatrix,
}


# Tests __add__ for all combinations of internal and external types
@pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_add_sparse(right_type, matrix_factory):
    left = matrix_factory("SparseMatrix")
    right = matrix_factory(right_type)
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("sparse", "sparse")])


@pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_sparse_add_dense(right_type, matrix_factory):
    left = matrix_factory("SparseMatrix")
    right = matrix_factory(right_type)
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("sparse", "dense")])


@pytest.mark.parametrize("right_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_add_dense(right_type, matrix_factory):
    left = matrix_factory("DenseMatrix")
    right = matrix_factory(right_type)
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("dense", "dense")])


@pytest.mark.parametrize("right_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_dense_add_sparse(right_type, matrix_factory):
    left = matrix_factory("DenseMatrix")
    right = matrix_factory(right_type)
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("dense", "sparse")])


# Tests __radd__ for all combinations of internal and external types
@pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_radd_sparse(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("SparseMatrix")
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("sparse", "sparse")])


@pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_radd_sparse(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("SparseMatrix")
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("dense", "sparse")])


@pytest.mark.parametrize("left_type", ["DenseMatrix"] + EXTERNAL_DENSE_TYPES)
def test_dense_radd_dense(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("DenseMatrix")
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("dense", "dense")])


@pytest.mark.parametrize("left_type", ["SparseMatrix"] + EXTERNAL_SPARSE_TYPES)
def test_sparse_radd_dense(left_type, matrix_factory):
    left = matrix_factory(left_type)
    right = matrix_factory("DenseMatrix")
    result = left + right
    assert isinstance(result, ADD_EXPECTED[("sparse", "dense")])
