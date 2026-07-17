"""
...
"""

from abc import ABC, abstractmethod


class StatisticalModel(ABC):
    """Class for statistical models."""

    latent_structures: list[LatentStructure] = []
    latent_assembly_operator = None

    hyperparameters: dict[Hyperparameter] = {}
    are_hyperparameters_true: bool = False

    priors: smthg[Prior] = ...

    likelihood: smthg[Likelihood] = ...

    # 1. Class attributes (if any)
    # 2. Initialization
    def __init__(self):
        pass

    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods
    @abstractmethod
    def assemble_prior_latent(self, theta):
        """Assemble the log prior of the latent variables given theta."""

    @abstractmethod
    def assemble_conditional_latent(self, theta):
        """Assemble the log conditional of the latent parameters."""

    # 11. Private/protected methods (start with _)
    @staticmethod
    @abstractmethod
    def _scale_hyperparameters_from_external_to_internal_scale(theta):
        """Scale hyperparameters to internal scale."""

    @staticmethod
    @abstractmethod
    def _scale_hyperparameters_from_internal_to_external_scale(theta):
        """Scale hyperparameters to external scale."""
