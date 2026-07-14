"""Hyperparameter dataclasses"""

from dataclasses import dataclass


def assemble_hyperparameter_dict(
    hyperparameters: list["Hyperparameter"],
) -> dict[str, "Hyperparameter"]:
    """Assemble a dict of hyperparameters from a list of Hyperparameter objects.

    Parameters
    ----------
    hyperparameters : list[Hyperparameter]
        List of Hyperparameter objects.

    Returns
    -------
    dict[str, Hyperparameter]
        Dict with keys matching Hyperparameter.name and values being the Hyperparameter objects.

    Notes
    -----
    - Using this helper function ensure that one of the key design principles of
    the HyperparameterManager is respected: the name field in Hyperparameter must
    match the dict key in Model.
    """
    return {hp.name: hp for hp in hyperparameters}


@dataclass
class Hyperparameter:
    """A single hyperparameter with its metadata."""

    # Unique identifier, should be unique across all hyperparameters in the model.
    name: str
    # Value of the hyperparameter.
    value: float

    # Wether or not this hyperparameter is fixed or can be optimized.
    # if True, not optimized
    is_fixed: bool = False

    # Hyperparameter bounds for optimization
    bounds: tuple[float, float] | None = (
        None  # Default no bounds for optimization (-inf, +inf)
    )
