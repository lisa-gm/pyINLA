from .latent_structure import LatentStructure


def kronecker(LS_1: LatentStructure, LS_2: LatentStructure) -> LatentStructure:
    """Return a LatentStructure that is the Kronecker product of the given latent structures."""
    # ...


def linear_combination(
    latent_structures: list[LatentStructure], weights: list[float]
) -> LatentStructure:
    """Return a LatentStructure that is the linear combination of the given latent structures."""
    # ...
