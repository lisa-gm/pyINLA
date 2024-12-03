# Copyright 2024 pyINLA authors. All rights reserved.

import os
from warnings import warn
from typing import Any, TypeAlias, TypeVar
from numpy.typing import ArrayLike

from pyinla.__about__ import __version__


# Allows user to specify the array module via an environment variable.
ARRAY_MODULE = os.environ.get("ARRAY_MODULE")
if ARRAY_MODULE is not None:
    if ARRAY_MODULE == "numpy":
        import numpy as xp
        import scipy as sp

    elif ARRAY_MODULE == "cupy":
        try:
            import cupy as xp
            import cupyx.scipy as sp

            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)

        except ImportError as e:
            warn(f"'CuPy' is unavailable, defaulting to 'NumPy'. ({e})")
            import numpy as xp
            import scipy as sp
    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{ARRAY_MODULE}'")
else:
    # If the user does not specify the array module, prioritize numpy.
    warn("No `ARRAY_MODULE` specified, pyINLA.core defaulting to 'NumPy'.")
    import numpy as xp
    import scipy as sp

# In any case, check if CuPy is available.
CUPY_AVAILABLE = False
try:
    import cupy as xp
    import cupyx.scipy as sp

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    xp.abs(1)

    CUPY_AVAILABLE = True
except ImportError as e:
    warn(f"No 'CuPy' backend detected. ({e})")


MPI_AVAILABLE = False
try:
    from mpi4py import MPI

    comm_rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()

    MPI_AVAILABLE = True
except ImportError as e:
    warn(f"No 'MPI' backend detected. ({e})")

    comm_rank = 0
    comm_size = 1


# Some type aliases for the array module.
_ScalarType = TypeVar("ScalarType", bound=xp.generic, covariant=True)
_DType = xp.dtype[_ScalarType]
NDArray: TypeAlias = xp.ndarray[Any, _DType]

__all__ = [
    "__version__",
    "xp",
    "sp",
    "ArrayLike",
    "NDArray",
    "comm_rank",
    "comm_size",
    "CUPY_AVAILABLE",
    "MPI_AVAILABLE",
]
