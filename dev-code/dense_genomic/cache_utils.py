"""..."""


def restore_from_cache(elements: list | dict):
    """
    Restore the given elements from cache.

    Parameters
    ----------
    elements : list | dict
        The elements to restore from cache.

    Notes
    -----
    - Each element must implement the `restore_from_cache` method.
    """
    if isinstance(elements, dict):
        elements = list(elements.values())

    for element in elements:
        element.restore_from_cache()


def store_in_cache(elements: list | dict):
    """
    Store the given elements in cache.

    Parameters
    ----------
    elements : list | dict
        The elements to store in cache.

    Notes
    -----
    - Each element must implement the `store_in_cache` method.
    """
    if isinstance(elements, dict):
        elements = list(elements.values())

    for element in elements:
        element.store_in_cache()
