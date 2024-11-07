# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def get_available_devices() -> list:
    """
    Return a list of available GPU devices.

    Returns
    -------
    device_properties: list. A list of available GPU devices. Returns None if CuPy is not available.
    """
    device_properties = None

    if CUPY_AVAILABLE:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        device_properties = [
            cp.cuda.runtime.getDeviceProperties(i)["name"] for i in range(n_gpus)
        ]

    return device_properties


def set_device(device_id: int):
    """
    Set the device to use.

    Parameters
    ----------
    device_id: int. The device id to use.
    """

    if CUPY_AVAILABLE:
        cp.cuda.Device(device_id).use()
