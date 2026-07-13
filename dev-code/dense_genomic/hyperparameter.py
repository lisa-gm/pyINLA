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

from dataclasses import dataclass
from math import inf
from pathlib import Path

from dalia.statistical_modeling_toolbox.statistical_model import StatisticalModel
import numpy as np
import copy


@dataclass
class HyperparameterManagerConfig:
    """Configuration for HyperparameterManager (HPM for short).
    
    Attributes
    ----------
    checkpoint_hpm : bool
        Whether to checkpoint the HPM state (hyperparameter, order, and array) to disk. Allows to resume optimization from last accepted iteration.
    checkpoint_hpm_every : int
        How often to checkpoint the HPM state (in accepted iterations).
    checkpoint_hpm_path : Path
        Path to save the HPM state checkpoint file.
    track_history : bool
        Whether to track the history of accepted hyperparameter values. This will store an array of shape (num_accepted_iterations, num_hyperparameters) and the corresponding objective function values.
    checkpoint_history_every : int
        How often to checkpoint the history of accepted hyperparameter values. This takes an effect only if the track_history is set to True.
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
    bounds: tuple[float, float] | None = None  # Default no bounds for optimization (-inf, +inf)

class HyperparameterManager:
    """
    Bridge between Model (dict-based) and Optimizer (array-based).
    
    Key design principles:
    1. Manager owns the ordered array representation
    2. Only ACCEPTED iterations update manager._array and _history
    3. Buffer holds tentative/perturbed values (not yet accepted)
    4. name field in Hyperparameter must match dict key in Model
    """

    def __init__(
        self, 
        model: StatisticalModel,
        order: list[str] | None = None,
        config: HyperparameterManagerConfig | None = None
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
            assert hp.name == key, f"Hyperparameter.name '{hp.name}' must match dict key '{key}'"

        # Store metadata
        # . Make a deep copy of the Model hyperparameters to ensure that the 
        # manager's state is independent of the model's state.
        self._hyperparameters = copy.deepcopy(model.get_hyperparameters())
        self._order = order or list(self._hyperparameters.keys())

        # Create mapping (CRITICAL: index <-> key)
        # . separate concerns from fixed vs optimized hyperparameters
        self._fixed_keys: list[str] = [key for key in self._order if self._hyperparameters[key].is_fixed]
        self._optimized_keys: list[str] = [key for key in self._order if not self._hyperparameters[key].is_fixed]
        # . Create mapping for optimized hyperparameters only (optimizer interface)
        self._optimized_key_to_index: dict[str, int] = {key: i for i, key in enumerate(self._optimized_keys)}
        self._optimized_index_to_key: dict[int, str] = {i: key for key, i in self._optimized_key_to_index.items()}
        # . Create full mapping (model interface)
        self._key_to_index: dict[str, int] = {key: i for i, key in enumerate(self._order)}
        self._index_to_key: dict[int, str] = {i: key for key, i in self._key_to_index.items()}
        
        # Initialize array from initial values
        # self._array is the current (latest) accepted hyperparameter values in array form
        # . This array contains ONLY optimized hyperparameters, not fixed ones
        self._array: np.ndarray = np.array([self._hyperparameters[key].value for key in self._optimized_keys])
        # . Store fixed hyperparameter values separately
        self._fixed_values: dict[str, float] = {key: self._hyperparameters[key].value for key in self._fixed_keys}

        # State management
        self._buffer: np.ndarray | None = None  # Tentative values (not accepted)
        self._iteration: int = 0
        self._history: list[tuple[int, np.ndarray, float]] = []  # (iter, array, f)

        # HPM Configuration
        self._config: HyperparameterManagerConfig = config or HyperparameterManagerConfig()

    # Public API for : Array <-> Dict conversion
    def get_dict(self, 
            include_fixed: bool = True
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
        
        Example
        -------
        >>> manager.get_dict(include_fixed=False)
        {"tau_iid": 1.0, "tau_queen": 2.0}
        >>> manager.get_dict(include_fixed=True)
        {"tau_iid": 1.0, "tau_queen": 2.0, "prec_regression": 0.1}
        """
        result_dict = {key: self._hyperparameters[key].value for key in self._optimized_keys}

        if include_fixed:
            result_dict.update(self._fixed_values)

        return result_dict

    def get_array(self, include_fixed: bool = False) -> np.ndarray:
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
            for i, key in enumerate(self._order):
                array_with_fixed[i] = self._array[self._key_to_index[key]]
            return array_with_fixed
        
        # .copy() of np.ndarray is a deep copy
        return self._array.copy()
   
    # Public API for : Optimizer Specific Methods 
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
        return [self._hyperparameters[key].bounds for key in self._optimized_keys]

    # Public API for : State Management
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
            raise ValueError("Provided buffered array length does not match the number of optimized hyperparameters")
        
        # .copy() of np.ndarray is a deep copy
        self._buffer = array.copy()
    
    def commit_buffer(self, f: float | None = None) -> None:
        """
        Commit buffered values as accepted iteration.
        
        Called by callback AFTER scipy accepts a point.
        Updates _array, increments iteration, optionally records history.
        """
        if self._buffer is None:
            raise ValueError("No buffered values to commit")
        
        # Update accepted values
        self._array = self._buffer.copy()
        
        # History Tracking and Checkpointing
        if self._config.track_history:
            self._history.append((self._iteration, self._array.copy(), f))
            
            # Checkpoint history if needed
            if (self._iteration + 1) % self._config.checkpoint_history_every == 0:
                np.save(self._config.checkpoint_history_path, self._history, allow_pickle=True)

        # HPM State Checkpointing
        if self._config.checkpoint_hpm and (self._iteration + 1) % self._config.checkpoint_hpm_every == 0:
            self._checkpoint()

        # Update HPM hyperparameters with accepted values
        for i, key in enumerate(self._optimized_keys):
            self._hyperparameters[key].value = self._array[i]

        # Cleanup
        self._buffer = None
        self._iteration += 1
    
    # I think that this method might not be needed as when scipy will
    # reject a point, it will simply call the objective function again 
    # with a new point through the `buffer_update` interface. The buffer 
    # will naturally be replaced by the new tentative point, and the 
    # previous one will be discarded.
    # def reject_buffer(self) -> None:
    #     """
    #     Discard buffered values (scipy rejected the perturbation).
    #
    #     No state change — buffer remains for next perturbation.
    #     """
    #     self._buffer = None  # Clear buffer, wait for next perturbation

    # Public API for : History Access
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

    # Public API for : Model Integration
    def update_model(self, model: StatisticalModel) -> None:
        """
        Update model's hyperparameters with manager's hyperparameters.
        
        Called after optimization completes.
        """
        for key in self._optimized_keys:
            model.set_hyperparameter(key, self._hyperparameters[key].copy())

    # Public API for : Checkpointing
    @classmethod
    def load_checkpoint(cls, path: Path) -> "HyperparameterManager":
        """Restore manager from checkpoint."""
        state = np.load(path, allow_pickle=True).item()
        
        manager = cls(
            hyperparameters=state["hyperparameters"], 
            order=state["order"],
            config=state["config"]
        )
        manager._array = state["array"]
        manager._iteration = state["iteration"]
        manager._history = state["history"]
        return manager

    # Private API for : Checkpointing
    def _checkpoint(self) -> None:
        """Save entire manager state to disk.

        Allow for a restart of the optimization from the last accepted iteration.
        
        """
        state = {
            "hyperparameters": self._hyperparameters,
            "order": self._order,
            "config": self._config,
            "array": self._array,
            "iteration": self._iteration,
            "history": self._history
        }
        np.save(self._config.checkpoint_hpm_path, state, allow_pickle=True)

    