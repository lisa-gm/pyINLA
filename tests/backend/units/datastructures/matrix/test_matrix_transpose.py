# tests/backend/units/datastructures/matrix/test_matrix_transpose.py
import numpy as np
import pytest

from dalia.backend.datastructures import DenseMatrix, SparseMatrix
from .conftest import INTERNAL_DEVICE_TYPES

@pytest.mark.parametrize("matrix_type", ["SparseMatrix", "DenseMatrix"])
@pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
def test_transpose_shape(matrix_type, device, matrix_factory):
    """Test that transpose has correct shape"""
    matrix = matrix_factory(matrix_type, device=device, shape=(3, 4))
    transposed = matrix.T
    assert transposed.shape == (4, 3)


@pytest.mark.parametrize("matrix_type", ["SparseMatrix", "DenseMatrix"])
@pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
def test_transpose_correctness(matrix_type, device, matrix_factory):
    """Test that transpose has correct values"""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    matrix = matrix_factory(matrix_type, device=device, shape=data.shape, data=data)
    transposed = matrix.T

    # Convert to dense for comparison
    if isinstance(transposed, SparseMatrix):
        result = transposed.toarray()
    else:
        result = transposed.toarray()

    expected = data.T
    assert np.allclose(result, expected)


@pytest.mark.parametrize("matrix_type", ["SparseMatrix", "DenseMatrix"])
@pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
def test_transpose_double(matrix_type, device, matrix_factory):
    """Test that (A.T).T == A"""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    matrix = matrix_factory(matrix_type, device=device, shape=data.shape, data=data)
    double_transposed = matrix.T.T

    # Convert both to dense for comparison
    if isinstance(matrix, SparseMatrix):
        original = matrix.toarray()
        result = double_transposed.toarray()
    else:
        original = matrix.toarray()
        result = double_transposed.toarray()

    assert np.allclose(result, original)


@pytest.mark.parametrize("matrix_type", ["SparseMatrix", "DenseMatrix"])
@pytest.mark.parametrize("device", INTERNAL_DEVICE_TYPES)
def test_transpose_type_preserved(matrix_type, device, matrix_factory):
    """Test that transpose returns correct Matrix subclass"""
    matrix = matrix_factory(matrix_type, device=device)
    transposed = matrix.T

    if matrix_type == "SparseMatrix":
        assert isinstance(transposed, SparseMatrix)
    elif matrix_type == "DenseMatrix":
        assert isinstance(transposed, DenseMatrix)


def test_dense_transpose_is_view():
    """Test that dense transpose is a view (shares memory)"""
    original_data = np.array([[1, 2], [3, 4]])
    matrix = DenseMatrix(original_data.copy())  # Copy to avoid side effects
    transposed = matrix.T

    # Modify transpose
    transposed[1, 0] = 999

    # Original should be affected (at position [0, 1] in original = [1, 0] in transpose)
    assert matrix[0, 1] == 999
