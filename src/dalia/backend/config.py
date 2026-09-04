
default_hw_target = "host"
memory_regime = "manual"
memory_threshold = 0.95
cupy_version = None
nvmath_version = None
gputil_version = None
target_list = ["host"]
regime_list = ["manual"]
default_override = False

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

def check_nvmath_availability():
    """Check if NVIDIA Math Libraries are available.
    
    Returns:
        str or None: The version of NVIDIA Math Libraries if available, otherwise None.
    """
    global nvmath_version
    try:
        import nvmath
        nvmath_version = nvmath.__version__
    except ImportError:
        pass
    return nvmath_version

def check_gputil_availability():
    """Check if NVIDIA Math Libraries are available.
    
    Returns:
        str or None: The version of NVIDIA Math Libraries if available, otherwise None.
    """
    global gputil_version
    try:
        import gputil
        gputil_version = gputil.__version__

        global regime_list
        if "auto" not in regime_list:
            regime_list.append("auto")
    except ImportError:
        pass
    return gputil_version

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

def set_override(override):
    """Set the override flag for memory management.

    This flag determines whether to override the default memory management behavior.

    Args:
        value (bool): True to enable override, False to disable.
    Returns:
        bool: The set override value.
    """
    global default_override
    default_override = override
    return default_override

__all__ = [
    "default_hw_target",
    "memory_regime",
    "memory_threshold",
    "cupy_version",
    "nvmath_version",
    "target_list",
    "default_override",
]