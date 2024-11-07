# Copyright 2024 pyINLA authors. All rights reserved.

import os
from warnings import warn

from numpy.typing import ArrayLike

from pyinla.__about__ import __version__

# Allows user to specify the array module via an environment variable.
ARRAY_MODULE = os.environ.get("ARRAY_MODULE")
if ARRAY_MODULE is not None:
    if ARRAY_MODULE == "numpy":
        import numpy as xp
        from scipy import sparse

    elif ARRAY_MODULE == "cupy":
        try:
            import cupy as xp
            from cupyx.scipy import sparse

            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)

        except ImportError as e:
            warn(f"'CuPy' is unavailable, defaulting to 'NumPy'. ({e})")
            import numpy as xp
            from scipy import sparse
    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{ARRAY_MODULE}'")
else:
    # If the user does not specify the array module, prioritize numpy.
    warn("No `ARRAY_MODULE` specified, pyINLA.core defaulting to 'NumPy'.")
    import numpy as xp
    from scipy import sparse

# In any case, check if CuPy is available.
CUPY_AVAILABLE = False
try:
    import cupy as xp
    from cupyx.scipy import sparse

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    xp.abs(1)

    CUPY_AVAILABLE = True
except ImportError as e:
    warn(f"No 'CuPy' backend detected. ({e})")

__all__ = ["__version__", "xp", "sparse", "ArrayLike", "CUPY_AVAILABLE"]
