# Copyright 2024 pyINLA authors. All rights reserved.

import os
from warnings import warn
from typing import Any, TypeAlias, TypeVar
from numpy.typing import ArrayLike

from pyinla.__about__ import __version__

backend_flags = {
    "array_module": None,
    "cupy_avail": False,
    "mpi_avail": False,
}

# Allows user to specify the array module via an environment variable.
backend_flags["array_module"] = os.environ.get("ARRAY_MODULE")

if backend_flags["array_module"] is not None:
    if backend_flags["array_module"] == "numpy":
        import numpy as xp
        import scipy as sp

    elif backend_flags["array_module"] == "cupy":
        try:
            import cupy as xp
            import cupyx.scipy as sp

            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)

        except (ImportError, ImportWarning, ModuleNotFoundError) as e:
            warn(f"'CuPy' is unavailable, defaulting to 'NumPy'. ({e})")
            import numpy as xp
            import scipy as sp
    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{backend_flags['array_module']}'")
else:
    # If the user does not specify the array module, prioritize numpy.
    warn("No `ARRAY_MODULE` specified, pyINLA.core defaulting to 'NumPy'.")
    import numpy as xp
    import scipy as sp

# In any case, check if CuPy is available.
try:
    import cupy

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    cupy.abs(1)

    backend_flags["cupy_avail"] = True
except (ImportError, ImportWarning, ModuleNotFoundError) as e:
    warn(f"No 'CuPy' backend detected. ({e})")


try:
    from mpi4py import MPI

    comm_rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()

    backend_flags["mpi_avail"] = True
except (ImportError, ImportWarning, ModuleNotFoundError) as e:
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
    "backend_flags",
]
