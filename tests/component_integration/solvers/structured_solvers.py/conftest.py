# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest
from dalia import backend_flags, xp, ArrayLike

SEED = 63

np.random.seed(SEED)

if backend_flags["cupy_avail"]:
    import cupy as cp

    cp.random.seed(cp.uint64(SEED))


N_DIAG_BLOCKS = [
    pytest.param(1, id="n_diag_blocks=1"),
    pytest.param(2, id="n_diag_blocks=2"),
    pytest.param(3, id="n_diag_blocks=3"),
    pytest.param(4, id="n_diag_blocks=4"),
]
@pytest.fixture(params=N_DIAG_BLOCKS, autouse=True)
def n_diag_blocks(request: pytest.FixtureRequest) -> int:
    return request.param

DIAGONAL_BLOCKSIZE = [
    pytest.param(2, id="diagonal_blocksize=2"),
    pytest.param(3, id="diagonal_blocksize=3"),
]
@pytest.fixture(params=DIAGONAL_BLOCKSIZE, autouse=True)
def diagonal_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param

ARROWHEAD_BLOCKSIZE = [
    pytest.param(0, id="arrowhead_blocksize=0"),
    pytest.param(2, id="arrowhead_blocksize=2"),
    pytest.param(3, id="arrowhead_blocksize=3"),
]
@pytest.fixture(params=ARROWHEAD_BLOCKSIZE, autouse=True)
def arrowhead_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param

SOLVERS_TYPES = [
    pytest.param("serinv", id="solvers_types=serinv"),
]
@pytest.fixture(params=SOLVERS_TYPES)
def solver_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def create_solver():
    from .utils import _create_solver
    return _create_solver

@pytest.fixture
def create_pobta():
    from .utils import _create_pobta
    return _create_pobta

@pytest.fixture
def create_pobt():
    from .utils import _create_pobt
    return _create_pobt

@pytest.fixture
def allclose_dense_structured():
    from .utils import _allclose_dense_structured
    return _allclose_dense_structured