from abc import ABC, abstractmethod

from pathlib import Path

from .hyperparameter import Hyperparameter
from dalia.backend.datastructure import Matrix

class StatisticalModelConfig:
    dataset_path: Path


class StatisticalModel(ABC):
    # I think that the conditional precision matrix should not be part of the model anymore.

    def __init__(self, config: StatisticalModelConfig):
        self.config = config

        # Load the model hyperparameters and their initial values
        self.hyperparameters : list[Hyperparameter] = ...

        # Load and construct the initial prior precision matrix
        # -> This should be called someting like "model components"
        # -> prior_precision_matrix_components
        self.q_prior : Matrix = ...

        # Load and construct the design matrix
        # -> Same here, design_matrix_components
        self.design_matrix : Matrix = ...

    def assemble_prior_precision_matrix(self, hyperparameters: list[Hyperparameter]) -> Matrix:
        if self.q_prior.is_cached():
            # load it back from cache
            self.q_prior.restore()

        # Apply some transofmration to self.q_prior
        q_prior_at_hp : Matrix = f(hp, self.q_prior)

        # Cache the bare precision matrix
        self.q_prior.cache()

        return q_prior_at_hp
    
    def assemble_design_matrix(self) -> Matrix:
        # ...
        restore_from_cache(design_matrix_components)

        # Assemble the design matrix from its components
        design_matrix = self._assemble_design_matrix(design_matrix_components)

        cache(design_matrix_components)

        return design_matrix
    
    @abstractmethod
    def _assemble_design_matrix(self) -> Matrix:
        ...

class GenomicModel(StatisticalModel):
    def __init__(self):
        super().__init__()
