# Copyright 2024-2025 DALIA authors. All rights reserved.

import warnings

import pytest


@pytest.mark.mpi_skip()
def test_factorize_correctness(
    generate_spd_spmatrix,
    create_solver,
    solver_type,
    matrix_size,
    density,
):
    A = generate_spd_spmatrix(matrix_size, density)

    solver = create_solver(solver_type)

    # Test that factorization is callable without errors
    solver.factorize(A)

    # Warn that this test only checks callability, not numerical correctness
    warnings.warn(
        f"Test passed but only verified sparse_solver.{solver_type}.factorize() is callable. "
        "Cannot verify against reference Cholesky as current sparse solver is using LU decomposition. "
        "Correctness is still tested through `solve()` and `logdet()` tests.",
        UserWarning,
        stacklevel=2,
    )
