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

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# from cache_utils import restore_from_cache, store_in_cache
from hp_dataclass import Hyperparameter

from dalia.backend.datastructures import DenseMatrix, Matrix


@dataclass
class StatisticalModelConfig:
    # Dataset and Model Paths
    path_to_model_components: Path
    path_to_observations: Path

    # Model Hyperparameters
    hyperparameters: dict[str, Hyperparameter]

    def __post_init__(self):
        # Validate: all names match dict keys
        for key, hp in self.hyperparameters.items():
            assert (
                hp.name == key
            ), f"Hyperparameter.name '{hp.name}' must match dict key '{key}'"

        # Validate paths
        if not self.path_to_model_components.exists():
            raise FileNotFoundError(
                f"Path to model components '{self.path_to_model_components}' does not exist."
            )
        if not self.path_to_observations.exists():
            raise FileNotFoundError(
                f"Path to observations '{self.path_to_observations}' does not exist."
            )


class StatisticalModel(ABC):
    """Abstract base class for statistical models, providing a
    common interface for assembling prior precision matrices and
    design matrices from their components.
    """

    def __init__(self, config: StatisticalModelConfig):
        self.config = config

        # Load the model hyperparameters and their initial values
        self._hyperparameters: dict[str, Hyperparameter] = self.config.hyperparameters

        # Load the observations
        # . for now hard-coded the name to be "observations.npy"
        # . this could be part of the config or we might need to accomodate for Pandas DataFrame
        self.observations: np.ndarray = np.load(
            self.config.path_to_observations / "observations.npy"
        )

        # Specific statistical model overload these methods depending on the
        # components of the model (e.g. iid, regression, spatial, temporal, etc.)
        self.prior_components: dict = self._load_prior_components()
        self.design_components: dict = self._load_design_components()

    # --- Public API ---

    def assemble_prior_precision_matrix(
        self, hyperparameters_values: dict[str, float]
    ) -> Matrix:
        """Assemble the prior precision matrix from its components
        given (at) the current hyperparameter values.

        Parameters
        ----------
        hyperparameters_values : dict[str, float]
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
            hyperparameters_values=hyperparameters_values,
        )

        return q_prior

    def design_matrix(self) -> Matrix:
        """Return the design matrix associated with the Model.

        The design matrix is considered to be a static model geometry, it is latily
        assembled at first call but then simply looked up as a property of the model.
        
        Returns
        -------
        Matrix
            The design matrix (per reference, not a copy, hence any modification 
            to the returned matrix will affect the model's design matrix).

        Notes
        -----
        - For now the caching idea is defered to later optimization of the
        implementation. With signatures:
        - restore_from_cache(elements=self.design_components)
        - store_in_cache(elements=self.design_components)
        These caching ideas above are not realy relevant to the components 
        anymore as once the design matrix has been assembled they can be 
        destroyed. However the caching is still very relevant for the design 
        matrix itself.
        """
        # If the design matrix is not already assembled, assemble it from its
        # components and store it as a property of the model for future calls.
        if not hasattr(self, "_design_matrix"):
            self._design_matrix = self._assemble_design_matrix(
                design_components=self.design_components
            )

        return self._design_matrix

    def get_hyperparameters(self) -> dict[str, Hyperparameter]:
        """Get the model's hyperparameters.

        Returns
        -------
        dict[str, Hyperparameter]
            The model's hyperparameters.

        Notes
        -----
        - The hyperparameters are returned by reference, hence any
        modification to the returned dictionary will affect the
        model's hyperparameters.
        """
        return self._hyperparameters

    def set_hyperparameter_value(self, key: str, value: float) -> None:
        """Set a hyperparameter value.

        Parameters
        ----------
        key : str
            The key/name (unique identifier) of the hyperparameter to set.
        value : float
            The new value for the hyperparameter.
        """
        if key not in self._hyperparameters:
            raise KeyError(
                f"Hyperparameter '{key}' does not exist in the model. Adding new hyperparameter after instantiation is not allowed."
            )

        self._hyperparameters[key].value = value

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
        self, prior_components: dict, hyperparameters_values: dict[str, float]
    ) -> Matrix:
        """Abstract method whom specification will assemble the prior precision matrix
        from its components given the current hyperparameter values.

        Parameters
        ----------
        prior_components : dict
            The components of the prior precision matrix.
        hyperparameters_values : dict[str, float]
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


@dataclass
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
        # . This will be modified (later) using the appropriate Matrix specifications,
        # in particular DiagonalMatrix for the iid and regression components.
        iid_prior_matrix = DenseMatrix(
            data=np.eye(N=self.config.iid_prior_n, dtype=np.float64)
        )
        queen_prior_matrix = DenseMatrix(
            data=np.load(
                self.config.path_to_model_components / self.config.queen_prior_name
            )
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
            data=np.load(
                self.config.path_to_model_components / self.config.iid_design_name
            )
        )
        queen_design_matrix = DenseMatrix(
            data=np.load(
                self.config.path_to_model_components / self.config.queen_design_name
            )
        )
        regression_design_matrix = DenseMatrix(
            data=np.load(
                self.config.path_to_model_components
                / self.config.regression_design_name
            )
        )

        return {
            "iid": iid_design_matrix,
            "queen": queen_design_matrix,
            "regression": regression_design_matrix,
        }

    def _assemble_prior_precision_matrix(
        self, prior_components: dict, hyperparameters_values: dict[str, float]
    ) -> Matrix:
        """Assemble the prior precision matrix of the Genomic model from its components
        given the current hyperparameter values.

        Parameters
        ----------
        prior_components : dict
            The components of the prior precision matrix.
        hyperparameters_values : dict[str, float]
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
        iid_component: Matrix = prior_components["iid"]
        queen_component: Matrix = prior_components["queen"]
        regression_component: Matrix = prior_components["regression"]
        # . and their dimmensions
        n_iid: int = iid_component.shape[0]
        n_queen: int = queen_component.shape[0]
        n_regression: int = regression_component.shape[0]

        # Initialize the prior precision matrix with the appropriate shape
        prior_shape: tuple = (
            n_iid + n_queen + n_regression,
            n_iid + n_queen + n_regression,
        )
        q_prior: Matrix = DenseMatrix(data=np.zeros(prior_shape, dtype=np.float64))

        # Assemble the prior precision matrix from its components at the current hyperparameter values
        # . This block is gonna become way easier with the block-matrix. Then no need to
        # maintain knowledge about the different shapes of each component.
        # . This also currently assumes that each component of this specific model
        # is a square matrix.
        block_offsets: list = [0, n_iid, n_iid + n_queen]
        # . assign iid contribution
        q_prior[: block_offsets[1], : block_offsets[1]] = (
            hyperparameters_values["tau_iid"] * iid_component
        )
        # . assign queen contribution
        q_prior[
            block_offsets[1] : block_offsets[2], block_offsets[1] : block_offsets[2]
        ] = (hyperparameters_values["tau_queen"] * queen_component)
        # . assign regression contribution
        q_prior[block_offsets[2] :, block_offsets[2] :] = (
            hyperparameters_values["prec_regression"] * regression_component
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
        # . assumes 1D observations vector (n_observations, 1) and that
        # the design matrix is (n_observations, n_features)
        n_observations: int = self.observations.shape[0]

        # Extract the required components
        iid_design_matrix = design_components["iid"]
        queen_design_matrix = design_components["queen"]
        regression_design_matrix = design_components["regression"]
        # . and their dimmensions
        n_iid: int = iid_design_matrix.shape[1]
        n_queen: int = queen_design_matrix.shape[1]
        n_regression: int = regression_design_matrix.shape[1]

        # Initialize the design matrix with the appropriate shape
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
