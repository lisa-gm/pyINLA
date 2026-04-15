# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia.utils.gpu_utils import (
    format_size,
    free_unused_gpu_memory,
    get_array_module_name,
    get_available_devices,
    get_device,
    get_host,
    memory_report,
    set_device,
)
from dalia.utils.host import get_host_configuration
from dalia.utils.link_functions import cloglog, scaled_logit, sigmoid
from dalia.utils.multiprocessing import (
    DummyCommunicator,
    allgather,
    allreduce,
    bcast,
    get_active_comm,
    print_msg,
    smartsplit,
    synchronize,
    synchronize_gpu,
)
from dalia.utils.print_utils import (
    add_str_header,
    align_tables_side_by_side,
    ascii_logo,
    boxify,
)
from dalia.utils.spmatrix_utils import bdiag_tiling, extract_diagonal, memory_footprint
from .scalar_ndarray import ensure_scalar

__all__ = [
    "get_available_devices",
    "set_device",
    "get_array_module_name",
    "get_host",
    "get_device",
    "get_host_configuration",
    "sigmoid",
    "cloglog",
    "scaled_logit",
    "print_msg",
    "synchronize",
    "synchronize_gpu",
    "get_active_comm",
    "smartsplit",
    "allreduce",
    "allgather",
    "bcast",
    "bdiag_tiling",
    "extract_diagonal",
    "memory_footprint",
    "free_unused_gpu_memory",
    "add_str_header",
    "align_tables_side_by_side",
    "boxify",
    "ascii_logo",
    "memory_report",
    "format_size",
    "DummyCommunicator",
]
