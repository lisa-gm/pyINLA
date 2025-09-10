# Copyright 2024-2025 DALIA authors. All rights reserved.

import inspect
import os

import psutil

from dalia import NDArray, backend_flags, xp

if backend_flags["cupy_avail"]:
    import cupy as cp


def get_available_devices() -> list:
    """
    Return a list of available GPU devices.

    Returns
    -------
    device_properties : list
        A list of available GPU devices. Returns None if CuPy is not available.
    """
    device_properties = None

    if backend_flags["cupy_avail"]:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        device_properties = [
            cp.cuda.runtime.getDeviceProperties(i)["name"] for i in range(n_gpus)
        ]

    return device_properties


def set_device(comm_rank: int, comm_size: int) -> None:
    """
    Set the device to use.

    Parameters
    ----------
    device_id : int
        The device id to use.
    """
    if backend_flags["cupy_avail"]:
        available_devices = get_available_devices()
        device_id = comm_rank % len(available_devices)
        cp.cuda.Device(device_id).use()
        # TOLOG: print(f"Rank {comm_rank} is using device {device_id}.")


def get_array_module_name(arr: NDArray) -> str:
    """Given an array, returns the array's module name.
    This works for `numpy` even when `cupy` is not available.

    Parameters
    ----------
    arr : NDArray
        The array to check.

    Returns
    -------
    submodule_name : str
        The array module name used by the array.

    """
    submodule = inspect.getmodule(type(arr))
    return submodule.__name__.split(".")[0]


def get_host(arr: NDArray) -> NDArray:
    """Returns the host array of the given array.

    Parameters
    ----------
    arr : NDArray
        The array to convert.

    Returns
    -------
    host_arr : np.ndarray
        The equivalent numpy array.
    """
    if get_array_module_name(arr) == "numpy":
        return arr
    return arr.get()


def get_device(arr: NDArray) -> NDArray:
    """Returns the device array of the given array.

    Parameters
    ----------
    arr : NDArray
        The array to convert.

    Returns
    -------
    device_arr : NDArray
        The equivalent cupy array.
    """
    if get_array_module_name(arr) == "cupy":
        return arr
    return xp.asarray(arr)


def format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


# query memory usage GPU and free unused memory
def free_unused_gpu_memory() -> int:
    """Free unused memory on the GPU."""

    if backend_flags["cupy_avail"]:
        mempool = cp.get_default_memory_pool()

        mempool.free_all_blocks()

        return mempool.total_bytes()

    else:
        # return dummy value for numpy
        return 1


def memory_report() -> int:
    """Report the current memory usage.

    Returns
    -------
    used_memory : int
        The current memory usage in bytes.
    total_memory : int
        The total memory available in bytes.
    """
    used_memory = 0
    total_memory = 0

    if backend_flags["cupy_avail"] and backend_flags["array_module"] == "cupy":
        # Get (GPU) memory usage
        mempool = cp.get_default_memory_pool()
        used_memory = mempool.used_bytes()
        total_memory = mempool.total_bytes()
    else:
        # Get (CPU) memory usage
        pid = os.getpid()
        used_memory = psutil.Process(pid).memory_info().rss
        total_memory = psutil.virtual_memory().total

    return used_memory, total_memory


def debug_gpu_memory_usage(msg: str = "", sync: bool = True) -> None:
    """Prints the GPU memory usage."""
    if xp.__name__ == "cupy":

        import numpy as np
        from mpi4py import MPI
        from mpi4py.MPI import COMM_WORLD as global_comm

        free_memory, total_memory = xp.cuda.Device().mem_info
        usage = np.array((total_memory - free_memory) / total_memory)

        if sync:
            average_usage = np.empty(1)
            max_usage = np.empty(1)
            global_comm.Allreduce(usage, average_usage, op=MPI.SUM)
            global_comm.Allreduce(usage, max_usage, op=MPI.MAX)
            average_usage /= global_comm.size

            if global_comm.rank == 0:
                print(
                    f"{msg} | Rank-average device memory usage: {average_usage[0] * 100:.4f}%",
                    flush=True,
                )
                print(
                    f"{msg} | Max device memory usage: {max_usage[0] * 100:.4f}%",
                    flush=True,
                )
        else:
            print(f"{msg} | Rank {global_comm.rank} device memory usage: {usage * 100:.4f}%", flush=True)


