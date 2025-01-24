# Copyright 2024 pyINLA authors. All rights reserved.

import inspect

from pyinla import backend_flags, NDArray, xp

if backend_flags["cupy_avail"]:
    import cupy as cp


def get_available_devices() -> list:
    """
    Return a list of available GPU devices.

    Returns
    -------
    device_properties: list. A list of available GPU devices. Returns None if CuPy is not available.
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
    device_id: int. The device id to use.
    """
    if backend_flags["cupy_avail"]:
        available_devices = get_available_devices()
        device_id = comm_rank % len(available_devices)
        cp.cuda.Device(device_id).use()
        # TOLOG: COMPUTE INFOS
        print(f"Rank {comm_rank} is using device {device_id}.")


def get_array_module_name(arr: NDArray) -> str:
    """Given an array, returns the array's module name.

    This works for `numpy` even when `cupy` is not available.

    Parameters
    ----------
    arr : NDArray
        The array to check.

    Returns
    -------
    str
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
    np.ndarray
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
    NDArray
        The equivalent cupy array.

    """
    if get_array_module_name(arr) == "cupy":
        return arr
    return xp.asarray(arr)

