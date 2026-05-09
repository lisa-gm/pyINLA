
default_hw_target = None

def set_default_hw_target(hw_target):
    """Detect the default hardware target based on the availability of cupy."""
    default_hw_target = hw_target
    return default_hw_target

__all__ = [
    "default_hw_target",
]