"""Hyperparameter-related classes for managing model hyperparameters
during their optimization.

Design decisions
----------------

# 1. Hyperparameter Ownership: Manager Owns HP State
- The `HyperparameterManager` owns all optimization state. The
`StatisticalModel` does not track whether its hyperparameters have
been optimized.

This allows:
- Separation of concerns: optimization is not a statistical property
of a model.
- `StatisticalModel` remains a pure abstraction -> it knows nothing
about INLA, optimizers, or solvers.
- Modifying the inference method should be transparent for the
`StatisticalModel`.
- The optimizer orchestrator (here understand the optimization section of
DALIA) reads from the model, updates it, and owns the lifecycle.

Here are the 3 spaces in which hyperparameters exist and that needs to be co-ordinated:
1. MODEL SPACE (dict-based, named)
   - Keys: "sigma_st", "tau_iid", "tau_queen", etc.
   - Values: Hyperparameter objects with .name, .value, .bounds
   - Used by: Model.assemble_prior_precision_matrix()
   - Order: Irrelevant (dict lookup by key)

2. MANAGER SPACE (ordered array + mapping)
   - Keys: Ordered list ["tau_iid", "tau_queen", "sigma_st"]
   - Values: np.ndarray [τ₁, τ₂, τ₃]
   - Used by: HyperparameterManager internally
   - Order: CRITICAL (index → key mapping)

3. OPTIMIZER SPACE (scipy array interface)
   - Keys: Implicit indices [0, 1, 2, ...]
   - Values: np.ndarray [τ₁, τ₂, τ₃]
   - Used by: scipy.optimize.minimize
   - Order: CRITICAL (same as manager)

# 2. Behavior towards accepted / rejected iterations
When the optimizer is in the phase of "trying out" a new point, we would
still want to relie on the HyperparameterManager to provide the mapping
between the optimizer's array and the model's dict (optimizer space <-> manager space <-> model space).

To handle that, when the optimizer is trying out a new point, this
one if first put in a buffer in the HyperparameterManager. If this point
gets accepted by the optimizer, then the HyperparameterManager
will flush the buffer into the list of accepted hyperparameter values.
If the point is rejected, then the buffer will simply be replaced by the new
tentative point from the optimizer.

# 3. HPM Checkpointing and History tracking
The HPM checkpointing and the history checkpointing are two separated features.

The HPM checkpointing allows to save the entire state of the
HyperparameterManager, including the hyperparameters, their order, the current accepted values,
the iteration number, and the history. This allows to resume optimization
from the last accepted iteration.

The History tracking and checkpointing is mostly a debugging feature,
allowing to track the history of accepted hyperparameter values and
the corresponding objective function values. If set to True,
it will be checkpointed every `checkpoint_history_every` accepted iterations
as well as throught the HPM checkpointing if enabled.

One can decide to checkpoint the HPM without tracking the history,
or to track the history without checkpointing the HPM. Or to do
both, or to do neither.
"""

import copy
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Literal

import numpy as np
from hp_dataclass import Hyperparameter
from model import StatisticalModel


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
class HyperparameterManagerConfig:
    """Configuration for HyperparameterManager (HPM for short).

    Attributes
    ----------
    checkpoint_hpm : bool
        Whether to checkpoint the HPM state (hyperparameter, order, and array)
        to disk. Allows to resume optimization from last accepted iteration.
    checkpoint_hpm_every : int
        How often to checkpoint the HPM state (in accepted iterations).
    checkpoint_hpm_path : Path
        Path to save the HPM state checkpoint file.
    track_history : bool
        Whether to track the history of accepted hyperparameter values. This
        will store an array of shape (num_accepted_iterations, num_hyperparameters)
        and the corresponding objective function values.
    checkpoint_history_every : int
        How often to checkpoint the history of accepted hyperparameter values.
        This takes an effect only if the track_history is set to True.
    checkpoint_history_path : Path
        Path to save the history checkpoint file.
    """

    # HPM State Checkpointing
    checkpoint_hpm: bool = True
    checkpoint_hpm_every: int = 10
    checkpoint_hpm_path: Path = Path("hpm_checkpoint.npy")

    # Hyperparameter Optimization History Checkpointing
    track_history: bool = True
    checkpoint_history_every: int = 10
    checkpoint_history_path: Path = Path("hpm_history_checkpoint.npy")

    def __post_init__(self):
        if self.checkpoint_hpm_every <= 0:
            raise ValueError("checkpoint_hpm_every must be a positive integer")
        if self.checkpoint_history_every <= 0:
            raise ValueError("checkpoint_history_every must be a positive integer")


class HyperparameterManager:
    """
    Bridge between Model (dict-based) and Optimizer (array-based).

    Key design principles:
    1. Manager owns the ordered array representation
    2. Only ACCEPTED iterations update manager._optimized_array and _history
    3. Buffer holds tentative/perturbed values (not yet accepted)
    4. name field in Hyperparameter must match dict key in Model
    """

    def __init__(
        self,
        model: StatisticalModel,
        order: list[str] | None = None,
        config: HyperparameterManagerConfig | None = None,
    ):
        """
        Initialize manager from Model's hyperparameters.

        Parameters
        ----------
        model : StatisticalModel
            The statistical model containing the hyperparameters.
        order : list[str], optional
            Explicit order of hyperparameters in array. If None,
            uses dict insertion order (Python 3.7+). This is usefull if
            you want to enforce a specific order for example for debugging
            purposes.
        config : HyperparameterManagerConfig, optional
            Configuration for the manager. If None, uses default config.
        """
        # Validate: all names match dict keys
        for key, hp in model.get_hyperparameters().items():
            assert (
                hp.name == key
            ), f"Hyperparameter.name '{hp.name}' must match dict key '{key}'"

        # Store metadata
        # . The Model hyperparameters are used to create the mapping
        # at the initialization of the HPM, but not stored. Beyond this point
        # the HPM manages its own state and the Model is only updated at the
        # end of the optimization.
        hyperparameters = copy.deepcopy(model.get_hyperparameters())

        # . Store the keys to verify in case of a checkpoint load that the
        # model has the same hyperparameters
        self._model_keys = set(hyperparameters.keys())

        self._order = order or list(hyperparameters.keys())
        self._bounds = {key: hyperparameters[key].bounds for key in self._order}

        # Create mapping (CRITICAL: index <-> key)
        # . separate concerns from fixed vs optimized hyperparameters
        self._fixed_keys: list[str] = [
            key for key in self._order if hyperparameters[key].is_fixed
        ]
        self._optimized_keys: list[str] = [
            key for key in self._order if not hyperparameters[key].is_fixed
        ]
        # . Create mapping for optimized hyperparameters only (optimizer interface)
        self._optimized_key_to_index: dict[str, int] = {
            key: i for i, key in enumerate(self._optimized_keys)
        }
        self._optimized_index_to_key: dict[int, str] = {
            i: key for key, i in self._optimized_key_to_index.items()
        }
        # . Create full mapping (model interface)
        self._full_key_to_index: dict[str, int] = {
            key: i for i, key in enumerate(self._order)
        }
        self._full_index_to_key: dict[int, str] = {
            i: key for key, i in self._full_key_to_index.items()
        }

        # Initialize array from initial values
        # self._optimized_array is the current (latest) accepted hyperparameter values in array form
        # . This array contains ONLY optimized hyperparameters, not fixed ones
        # . Initialized from the model's hyperparameter values, in the order
        # specified by self._optimized_keys (that itself follows the order specified by self._order)
        self._optimized_array: np.ndarray = np.array(
            [hyperparameters[key].value for key in self._optimized_keys]
        )
        # . Store fixed hyperparameter values separately
        self._fixed_values: dict[str, float] = {
            key: hyperparameters[key].value for key in self._fixed_keys
        }

        # State management
        self._buffer: np.ndarray | None = None  # Tentative values (not accepted)
        self._iteration: int = 0
        self._history: list[tuple[int, np.ndarray, float]] = []  # (iter, array, f)

        # HPM Configuration
        self._config: HyperparameterManagerConfig = (
            config or HyperparameterManagerConfig()
        )

    # API for : Array <-> Dict conversion
    # . Public
    def convert_array_to_dict(
        self,
        array: np.ndarray,
        include_fixed: bool = True,
    ) -> dict[str, float]:
        """
        Convert an array of hyperparameter values to a dict matching Model.hyperparameters keys.
        The given array should match in length and order the optimized hyperparameters
        (not fixed ones). If include_fixed is True, the returned dict will also include
        fixed hyperparameters with their current values.

        This interface is used at the Optimizer -> HPM -> Model boundary, and by default
        includes fixed hyperparameters.

        Parameters
        ----------
        array : np.ndarray
            Array of hyperparameter values in the order specified by self._order.

        Returns
        -------
        dict[str, float]
            Dict with keys matching Model.hyperparameters.

        Raises
        ------
        ValueError
            If the length of the array does not match the number of optimized hyperparameters.

        Example
        -------
        >>> # Assuming the manager has 2 optimized hyperparameters "tau_iid" and "tau_queen"
        >>> manager.convert_array_to_dict(np.array([1.0, 2.0]), include_fixed=True)
        {"tau_iid": 1.0, "tau_queen": 2.0, "prec_regression": 0.1}
        >>> manager.convert_array_to_dict(np.array([1.0, 2.0]), include_fixed=False)
        {"tau_iid": 1.0, "tau_queen": 2.0}
        """
        if len(array) != len(self._optimized_keys):
            raise ValueError(
                "Provided array length does not match the number of optimized hyperparameters"
            )

        # . map the optimized hyperparameters to their respective keys
        return_dict = {
            self._optimized_index_to_key[i]: array[i] for i in range(len(array))
        }

        if include_fixed:
            return_dict.update(self._fixed_values)

        return return_dict

    def get_latest_hyperparameters(
        self,
        format: Literal["dict", "array"] = "dict",
        include_fixed: Literal["default", True, False] = "default",
    ) -> dict[str, float] | np.ndarray:
        """
        Get the latest accepted hyperparameter values in the specified format.

        Parameters
        ----------
        format : Literal["dict", "array"], optional
            The format of the returned hyperparameters.
            "dict" returns a dictionary with keys matching Model.hyperparameters.
            "array" returns a numpy array of hyperparameter values in the order
            specified by self._order.
        include_fixed : Literal["default", True, False], optional
            Whether to include fixed hyperparameters in the returned dict or array.
            If "default", includes fixed hyperparameters if format is "dict",
            and excludes them if format is "array".

        Returns
        -------
        dict[str, float] | np.ndarray
            Latest accepted hyperparameter values in the specified format.

        Raises
        ------
        ValueError
            If an invalid format is specified.

        Example
        -------
        >>> manager.get_latest_hyperparameters(format="dict")
        {"tau_iid": 1.0, "tau_queen": 2.0, "prec_regression": 0.1}
        >>> manager.get_latest_hyperparameters(format="array")
        np.array([1.0, 2.0])
        >>> manager.get_latest_hyperparameters(format="dict", include_fixed=False)
        {"tau_iid": 1.0, "tau_queen": 2.0}
        >>> manager.get_latest_hyperparameters(format="array", include_fixed=True)
        np.array([1.0, 2.0, 0.1])
        """
        if format == "dict":
            if include_fixed == "default":
                include_fixed = True
            return self._get_latest_hyperparameters_dict(include_fixed=include_fixed)
        elif format == "array":
            if include_fixed == "default":
                include_fixed = False
            return self._get_latest_hyperparameters_array(include_fixed=include_fixed)
        else:
            raise ValueError(f"Invalid format '{format}'. Must be 'dict' or 'array'.")

    # . Private
    def _get_latest_hyperparameters_dict(
        self, include_fixed: bool = True
    ) -> dict[str, float]:
        """
        Return a dict of the latest accepted hyperparameter values,
        optionally including fixed hyperparameters. The dictionary is matching
        the Model.hyperparameters keys.

        This interface is used at the HPM -> Model boundary, and by default
        includes fixed hyperparameters.

        Parameters
        ----------
        include_fixed : bool, optional
            Whether to include fixed hyperparameters in the returned dict.

        Returns
        -------
        dict[str, float]
            Dict with keys matching Model.hyperparameters.
        """
        result_dict = {
            self._optimized_index_to_key[i]: self._optimized_array[i]
            for i in range(len(self._optimized_array))
        }

        if include_fixed:
            result_dict.update(self._fixed_values)

        return result_dict

    def _get_latest_hyperparameters_array(
        self, include_fixed: bool = False
    ) -> np.ndarray:
        """Get current accepted values as array (for scipy x0).

        This interface is used at the Optimizer <- HPM, and by default
        do not includes fixed hyperparameters.

        Parameters
        ----------
        include_fixed : bool, optional
            Whether to include fixed hyperparameters in the returned array.

        Returns
        -------
        np.ndarray
            Array of hyperparameter values in the order specified by self._order.
            If include_fixed is True, the array will include fixed hyperparameters
            in their respective positions according to self._order.

        Notes
        -----
        The returned array is a deep copy of the internal state, so modifying
        it will not affect the manager's state.
        """
        if include_fixed:
            # Create a new array including fixed values
            # . If including fixed values, use the global key-index mapping
            array_with_fixed = np.zeros(len(self._order))
            # . map the optimited hyperparameters
            for i, key in enumerate(self._optimized_keys):
                index = self._full_key_to_index[key]
                array_with_fixed[index] = self._optimized_array[i]
            # . map the fixed hyperparameters
            for i, key in enumerate(self._fixed_keys):
                index = self._full_key_to_index[key]
                array_with_fixed[index] = self._fixed_values[key]
            return array_with_fixed

        # .copy() of np.ndarray is a deep copy
        return self._optimized_array.copy()

    # API for : Optimizer Specific Methods
    # . Public
    def get_bounds(self) -> list[tuple[float, float]]:
        """
        Get bounds as list of (lb, ub) tuples matching array order.
        For un-bounded hyperparameters, returns None.

        Returns only bounds for optimized hyperparameters (not fixed ones).

        Returns
        -------
        list[tuple[float, float] | None]
            Bounds in same order as get_array().

        Example
        -------
        >>> manager.get_bounds()
        [(1e-6, 1e6), (1e-6, 1e6), None]
        """
        return [self._bounds[key] for key in self._optimized_keys]

    # API for : State Management
    # . Public
    def buffer_update(self, array: np.ndarray) -> None:
        """
        Buffer a perturbed array (tentative, not yet accepted).

        Called by optimizer BEFORE each objective evaluation.
        The buffer is only committed on accepted iterations.

        Parameters
        ----------
        array : np.ndarray
            Tentative hyperparameter values to buffer.

        Raises
        ------
        ValueError
            If array length does not match the number of optimized hyperparameters.

        Notes
        -----
        The buffer will hold a deep copy of the provided array, so modifying the
        original array after calling this method will not affect the buffer.
        """
        if len(array) != len(self._optimized_keys):
            raise ValueError(
                "Provided buffered array length does not match the number of optimized hyperparameters"
            )

        # .copy() of np.ndarray is a deep copy
        self._buffer = array.copy()

    def commit_buffer(self, f: float | None = None) -> None:
        """
        Commit buffered values as accepted iteration.

        Called by callback AFTER scipy accepts a point.
        Updates _optimized_array, increments iteration, optionally records history.
        """
        if self._buffer is None:
            raise ValueError("No buffered values to commit")

        # Update accepted values
        self._optimized_array = self._buffer.copy()

        # History Tracking and Checkpointing
        if self._config.track_history:
            self._history.append((self._iteration, self._optimized_array.copy(), f))

            # Checkpoint history if needed
            if (self._iteration + 1) % self._config.checkpoint_history_every == 0:
                np.save(
                    self._config.checkpoint_history_path,
                    self._history,
                    allow_pickle=True,
                )

        # HPM State Checkpointing
        if (
            self._config.checkpoint_hpm
            and (self._iteration + 1) % self._config.checkpoint_hpm_every == 0
        ):
            self._checkpoint()

        # Cleanup
        self._buffer = None
        self._iteration += 1

    # API for : History Access
    # . Public
    def get_history(self) -> list[tuple[int, np.ndarray, float]]:
        """Get full history of accepted iterations.

        Returns
        -------
        list[tuple[int, np.ndarray, float]]
            List of tuples containing (iteration_number, hyperparameter_array, objective_value).
        """
        return self._history.copy()

    def get_last_iteration(self) -> int:
        """Get last accepted iteration number."""
        return self._iteration - 1 if self._history else 0

    # API for : Model Integration
    # . Public
    def update_model(self, model: StatisticalModel) -> None:
        """
        Update model's hyperparameters with manager's hyperparameters.

        Called after optimization completes.

        Parameters
        ----------
        model : StatisticalModel
            The statistical model to update with the latest accepted hyperparameter values.

        Notes
        -----
        - Only updates optimized hyperparameters, not fixed ones.
        - The model's hyperparameters are updated in-place.
        """
        for key in self._optimized_keys:
            model.set_hyperparameter_value(
                key, self._optimized_array[self._optimized_key_to_index[key]]
            )

    # API for : Checkpointing
    # . Public
    @classmethod
    def load_checkpoint(
        cls, model: StatisticalModel, path: Path
    ) -> "HyperparameterManager":
        """Restore manager from checkpoint.

        Parameters
        ----------
        model : StatisticalModel
            The statistical model to associate with the restored manager.
        path : Path
            Path to the checkpoint file.

        Returns
        -------
        HyperparameterManager
            Restored manager with state from the checkpoint.
        """
        state = np.load(path, allow_pickle=True).item()

        # Verify that the checkpointed HPM matches the model's hyperparameters
        if set(state["model_keys"]) != set(model.get_hyperparameters().keys()):
            raise ValueError(
                "Checkpointed hyperparameter keys do not match the model's hyperparameter keys. Cannot restore checkpoint."
            )

        # Instanciate the HPM
        manager = cls(model=model, order=state["order"], config=state["config"])

        # Restore the internal state
        manager._optimized_array = state["array"]
        manager._iteration = state["iteration"]
        manager._history = state["history"]

        return manager

    # . Private
    def _checkpoint(self) -> None:
        """Save entire manager state to disk.

        Allow for a restart of the optimization from the last accepted iteration.

        """
        state = {
            "config": self._config,
            "model_keys": self._model_keys,
            "array": self._optimized_array,
            "iteration": self._iteration,
            "history": self._history,
        }
        np.save(self._config.checkpoint_hpm_path, state, allow_pickle=True)
