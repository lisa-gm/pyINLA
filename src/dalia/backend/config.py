
default_hw_target = "host"
memory_regime = "manual"
memory_threshold = 0.95
cupy_version = None
target_list = ["host"]
regime_list = ["auto", "manual"]

def check_cupy_availability():
    """Check if CuPy is available.
    
    Returns:
        str or None: The version of CuPy if available, otherwise None.
    """
    global cupy_version
    try:
        import cupy
         
        cupy_version= cupy.__version__

        global target_list
        if "accelerator" not in target_list:
            target_list.append("accelerator")
    except ImportError:
        pass
    return cupy_version

def set_default_hw_target(hw_target):
    """Set the default hardware target.

    'host' will use the CPU and 'accelerator' will use an accelerator like a GPU if available. 
    If 'accelerator' is selected but not available, it will raise an error.
    
    Args:
        hw_target (str): 'host' or if supported by the system: 'accelerator'.

    Returns:
        str: The set hardware target.

    Raises:
        ValueError: If an invalid hardware target is provided.
    """
    global default_hw_target
    global target_list
    
    if hw_target not in target_list:
        raise ValueError(f"Invalid hardware target type '{hw_target}'. Supported target types are {target_list}.")
    default_hw_target = hw_target
    return default_hw_target

def set_memory_regime(regime):
    """Set the memory regime for the backend.

    Automatic memory management ('auto') will try to calculate everything on the accelerator,
    as long as there is enough memory available. If ther isn't enough memory, it will fall back to the host.

    Manual memorry management ('manual') will determine the calcualtion location based on the left operand.
    
    Args:
        regime (str): 'auto' for automatic memory management, 'manual' for user-controlled memory management.

    Returns:
        str: The set memory regime.

    Raises:
        ValueError: If an invalid memory regime is provided.
    """
    global regime_list
    if regime not in regime_list: 
        raise ValueError(f"Invalid memory regime. Supported regimes are {regime_list}.")
    global memory_regime 
    memory_regime = regime
    return memory_regime

def set_memory_threshold(threshold):
    """Set memory threshold for automatic memory management.

    This threshold determines the maximum percentage of accelerator memory that can be used for calculations.

    Args:
        threshold (float): A value between 0 and 1 representing the percentage of accelerator memory.
    Returns:
        float: The set memory threshold.
    Raises:
        ValueError: If the threshold is not between 0 and 1.
    """
    global memory_threshold
    if not (threshold > 0 and threshold <= 1):
        raise ValueError("Memory threshold must be a value between 0 and 1.")
    memory_threshold = threshold
    return memory_threshold

__all__ = [
    "default_hw_target",
    "memory_regime",
    "memory_threshold",
    "cupy_version",
    "target_list",
]