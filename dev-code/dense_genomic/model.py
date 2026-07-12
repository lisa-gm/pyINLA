"""Statistical Model

Design decisions:
- Conditional precision matrix: I think that the conditional
precision matrix should not be part of the model anymore. 4
Its construction (qp - adat) is completly independant of the
model (depends on the type of prior but that can be probed from
the Model), hence should not be part of it.
- Hyperparameters, their initial values, and how we get the
precision matrix at a given hyperparameter value: A model
has to know about the hyperparameters keys (to be able to use
them/map them when producing a prior precision matrix at a
given HP-point) however a Model do not store the updated
hyperparameters values every time the prior precision matrix
construction is requested. Constructing a prior precision
matrix at a given HP-point require the user to provide the
HP-value through the public API (no self get updated!).

"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from dalia.backend.datastructure import DenseMatrix, Matrix

from .cache_utils import restore_from_cache, store_in_cache
from .hyperparameter import Hyperparameter


class StatisticalModelConfig:
    dataset_path: Path

    n_observations: int


class StatisticalModel(ABC):
    """Abstract base class for statistical models, providing a
    common interface for assembling prior precision matrices and
    design matrices from their components.
    """

    def __init__(self, config: StatisticalModelConfig):
        self.config = config

        # Load the model hyperparameters and their initial values
        self.hyperparameters: dict[str, Hyperparameter] = ...

        # Specific statistical model overload these methods depending on the
        # components of the model (e.g. iid, regression, spatial, temporal, etc.)
        self.prior_components: dict = self._load_prior_components()
        self.design_components: dict = self._load_design_components()

    # --- Public API ---

    def assemble_prior_precision_matrix(
        self, hyperparameters: dict[str, Hyperparameter]
    ) -> Matrix:
        """Assemble the prior precision matrix from its components 
        given (at) the current hyperparameter values.

        Parameters
        ----------
        hyperparameters : dict[str, Hyperparameter]
            The current hyperparameter values.

        Returns
        -------
        Matrix
            The assembled prior precision matrix.

        Notes
        -----
        - For now the caching idea is defered to later optimization of the
        implementation. With signatures:
        - restore_from_cache(elements=self.prior_components)
        - store_in_cache(elements=self.prior_components)
        """
        # Assemble the prior precision matrix from its components
        # given the current hyperparameter values
        q_prior = self._assemble_prior_precision_matrix(
            prior_components=self.prior_components,
            hyperparameters=hyperparameters,
        )

        return q_prior

    def assemble_design_matrix(self) -> Matrix:
        """Assemble the design matrix from its components.
        
        Returns
        -------
        Matrix
            The assembled design matrix.
        """
        restore_from_cache(elements=self.design_components)

        # Assemble the design matrix from its components
        design_matrix = self._assemble_design_matrix(
            design_components=self.design_components
        )

        store_in_cache(elements=self.design_components)

        return design_matrix

    # --- Abstract methods ---

    @abstractmethod
    def _load_prior_components(self) -> dict:
        """Abstract method whom specification will load the specific
        components of the model needed to assemble the prior precision matrix.

        Returns
        -------
        dict
            A dictionary containing the components needed to assemble the prior precision matrix of the specific model.
        """
        ...

    @abstractmethod
    def _load_design_components(self) -> dict:
        """Abstract method whom specification will load the specific
        components of the model needed to assemble the design matrix.
        """
        ...

    @abstractmethod
    def _assemble_prior_precision_matrix(
        self, prior_components: dict, hyperparameters: dict[str, Hyperparameter]
    ) -> Matrix:
        """Abstract method whom specification will assemble the prior precision matrix
        from its components given the current hyperparameter values.

        Parameters
        ----------
        prior_components : dict
            The components of the prior precision matrix.
        hyperparameters : dict[str, Hyperparameter]
            The current hyperparameter values.

        Returns
        -------
        Matrix
            The assembled prior precision matrix.
        """
        ...

    @abstractmethod
    def _assemble_design_matrix(self, design_components: dict) -> Matrix:
        """Abstract method whom specification will assemble the design matrix
        from its components.
        """
        ...


class GenomicModelConfig(StatisticalModelConfig):
    # Component: iid
    iid_prior_n: int
    iid_design_name: (
        str  # Name of the file containing the design matrix for the iid component
    )

    # Component: queen
    queen_prior_name: str  # Name of the file containing the Queen contiguity matrix
    queen_design_name: (
        str  # Name of the file containing the design matrix for the queen component
    )

    # Component: regression
    regression_prior_n: int
    regression_design_name: str  # Name of the file containing the design matrix for the regression component


class GenomicModel(StatisticalModel):
    def __init__(self, config: GenomicModelConfig):
        super().__init__(config)

    def _load_prior_components(self) -> dict:
        """Load the different components of the GenomicModel.

        Returns
        -------
        dict
            A dictionary containing the components needed to assemble the prior precision matrix of the GenomicModel.

        Components
        ----------
        - iid: Independent and identically distributed prior component
        - queen: Spatial prior component based on the Queen contiguity matrix
        - regression: Regression prior component
        """
        # Load or assemble each components of the statistical model
        # . This will be modified using the appropriate Matrix specifications,
        # in particular DiagonalMatrix for the iid and regression components.
        iid_prior_matrix = DenseMatrix(data=np.eye(N=self.config.iid_prior_n, dtype=np.float64))
        queen_prior_matrix = DenseMatrix(
            data=np.load(self.config.dataset_path / self.config.queen_prior_name)
        )
        regression_prior_matrix = DenseMatrix(
            data=np.eye(N=self.config.regression_prior_n, dtype=np.float64)
        )

        return {
            "iid": iid_prior_matrix,
            "queen": queen_prior_matrix,
            "regression": regression_prior_matrix,
        }

    def _load_design_components(self) -> dict:
        """Load the different design components of the GenomicModel.

        Returns
        -------
        dict
            A dictionary containing the components needed to assemble the design matrix of the GenomicModel.
        """
        iid_design_matrix = DenseMatrix(
            data=np.load(self.config.dataset_path / self.config.iid_design_name)
        )
        queen_design_matrix = DenseMatrix(
            data=np.load(self.config.dataset_path / self.config.queen_design_name)
        )
        regression_design_matrix = DenseMatrix(
            data=np.load(self.config.dataset_path / self.config.regression_design_name)
        )

        return {
            "iid": iid_design_matrix,
            "queen": queen_design_matrix,
            "regression": regression_design_matrix,
        }

    def _assemble_prior_precision_matrix(
        self, prior_components: dict, hyperparameters: dict[str, Hyperparameter]
    ) -> Matrix:
        """Assemble the prior precision matrix of the Genomic model from its components
        given the current hyperparameter values.

        Parameters
        ----------
        prior_components : dict
            The components of the prior precision matrix.
        hyperparameters : dict[str, Hyperparameter]
            The current hyperparameter values.

        Returns
        -------
        Matrix
            The assembled prior precision matrix.

        Components
        ----------
        - iid: Independent and identically distributed prior component
        - queen: Spatial prior component based on the Queen contiguity matrix
        - regression: Regression prior component
        """
        # Extract the required components
        iid_component = prior_components["iid"]
        queen_component = prior_components["queen"]
        regression_component = prior_components["regression"]

        # Initialize the prior precision matrix with the appropriate shape
        prior_shape = (
            iid_component.shape + queen_component.shape + regression_component.shape
        )
        q_prior = DenseMatrix(data=np.zeros(prior_shape, dtype=np.float64))

        # Assemble the prior precision matrix from its components at the current hyperparameter values
        # . This block is gonna become way easier with the block-matrix. Then no need to
        # maintain knowledge about the different shapes of each component.
        # . This also currently assumes that each component of this specific model
        # is a square matrix.
        iid_prior_n: int = iid_component.shape[0]
        n_queen: int = queen_component.shape[0]

        block_offsets: list = [0, iid_prior_n, iid_prior_n + n_queen]
        q_prior[: block_offsets[1], :iid_prior_n] = (
            iid_component * hyperparameters["tau_iid"].value
        )
        q_prior[
            block_offsets[1] : block_offsets[2], block_offsets[1] : block_offsets[2]
        ] = (queen_component * hyperparameters["tau_queen"].value)
        q_prior[block_offsets[2] :, block_offsets[2] :] = (
            regression_component * hyperparameters["prec_regression"].value
        )

        return q_prior

    def _assemble_design_matrix(self, design_components: dict) -> Matrix:
        """Assemble the design matrix of the Genomic model from its components.

        Parameters
        ----------
        design_components : dict
            The components of the design matrix.

        Returns
        -------
        Matrix
            The assembled design matrix.

        Components
        ----------
        - iid: Independent and identically distributed design component
        - queen: Spatial design component based on the Queen contiguity matrix
        - regression: Regression design component
        """
        # Extract the required components
        iid_design_matrix = design_components["iid"]
        queen_design_matrix = design_components["queen"]
        regression_design_matrix = design_components["regression"]

        # Initialize the design matrix with the appropriate shape
        n_observations: int = self.config.n_observations
        n_iid: int = iid_design_matrix.shape[1]
        n_queen: int = queen_design_matrix.shape[1]
        n_regression: int = regression_design_matrix.shape[1]

        design_shape: tuple = (n_observations, n_iid + n_queen + n_regression)
        design_matrix: DenseMatrix = DenseMatrix(
            data=np.zeros(design_shape, dtype=np.float64)
        )

        # Assemble the design matrix from its components
        block_offsets: list = [0, n_iid, n_iid + n_queen]
        design_matrix[:, block_offsets[0] : block_offsets[1]] = iid_design_matrix
        design_matrix[:, block_offsets[1] : block_offsets[2]] = queen_design_matrix
        design_matrix[:, block_offsets[2] :] = regression_design_matrix

        return design_matrix
