# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from pyinla import backend_flags

if backend_flags["cupy_avail"]:
    from pyinla.kernels.blockmapping import compute_block_slice, compute_block_sort_index

__all__ = [
    "compute_block_slice",
    "compute_block_sort_index",
]
