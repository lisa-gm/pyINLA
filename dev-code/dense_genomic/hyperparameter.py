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
from scipy.optimize import Bounds



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
    bounds: Bounds = Bounds(lb=-inf, ub=inf)

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
        self._hyperparameters = model.get_hyperparameters()
        self._order = order or list(self._hyperparameters.keys())

        # Create mapping (CRITICAL: index <-> key)
        # . separate concerns from fixed vs optimized hyperparameters
        self._fixed_keys = [key for key in self._order if self._hyperparameters[key].is_fixed]
        self._optimized_keys = [key for key in self._order if not self._hyperparameters[key].is_fixed]
        # . Create mapping for optimized hyperparameters only (optimizer interface)
        self._optimized_key_to_index = {key: i for i, key in enumerate(self._optimized_keys)}
        self._optimized_index_to_key = {i: key for key, i in self._optimized_key_to_index.items()}
        # . Create full mapping (model interface)
        self._key_to_index = {key: i for i, key in enumerate(self._order)}
        self._index_to_key = {i: key for key, i in self._key_to_index.items()}
        
        # Initialize array from initial values
        # self._array is the current (latest) accepted hyperparameter values in array form
        # . This array contains ONLY optimized hyperparameters, not fixed ones
        self._array = np.array([self._hyperparameters[key].value for key in self._optimized_keys])
        # . Store fixed hyperparameter values separately
        self._fixed_values = {key: self._hyperparameters[key].value for key in self._fixed_keys}

        # State management
        self._buffer: np.ndarray | None = None  # Tentative values (not accepted)
        self._iteration = 0
        self._history: list[tuple[int, np.ndarray, float]] = []  # (iter, array, f)

        # HPM Configuration
        self._config = config or HyperparameterManagerConfig()

        # Public API for : Array <-> Dict conversion
        def array_to_dict(self, array: np.ndarray | None = None) -> dict[str, float]:
            """
            Convert array to dict with model-compatible keys.

            The length of the array must match the number of 
            optimized hyperparameters. If array is None, uses 
            current accepted values.
            
            Parameters
            ----------
            array : np.ndarray, optional
                Array to convert. If None, uses current accepted values.
            
            Returns
            -------
            dict[str, float]
                Dict with keys matching Model.hyperparameters.
            
            Raises
            ------
            ValueError
                If array length does not match the number of optimized hyperparameters.
                
            Example
            -------
            >>> manager.array_to_dict(np.array([1.0, 2.0]))
            {"tau_iid": 1.0, "tau_queen": 2.0}
            """
            if array is None:
                array = self._array  # Use accepted values
            
            if len(array) != len(self._optimized_keys):
                raise ValueError("Array length does not match the number of optimized hyperparameters")
            
            return {key: array[self._key_to_index[key]] for key in self._order}

        def dict_to_array(self, d: dict[str, float]) -> np.ndarray:
            """
            Convert dict to array matching manager's ordering.
            
            Parameters
            ----------
            d : dict[str, float]
                Dict with keys matching Model.hyperparameters.
            
            Returns
            -------
            np.ndarray
                Array in manager's order.
            
            Raises  
            ------
            ValueError
                If dict keys do not match manager's order or if length mismatch.
            ValueError
                If dict length does not match the number of optimized hyperparameters.
                
            Example
            -------
            >>> manager.dict_to_array({"tau_queen": 2.0, "tau_iid": 1.0})
            np.array([1.0, 2.0])  # Order follows manager._order
            """
            for key in d.keys():
                if key not in self._order:
                    raise ValueError(f"Key '{key}' not in manager's order")

            if len(d) != len(self._optimized_keys):
                raise ValueError("Dict length does not match the number of optimized hyperparameters")
            
            return np.array([d[key] for key in self._order])

        # Public API for : Optimizer Getters
        def get_array(self) -> np.ndarray:
            """Get current accepted values as array (for scipy x0).
            
            Returns only optimized hyperparameters (not fixed ones).
            """
            return self._array.copy()
        
        def get_bounds(self) -> list[tuple[float, float]]:
            """
            Get bounds as list of (lb, ub) tuples matching array order.
            
            Returns only bounds for optimized hyperparameters (not fixed ones).

            Returns
            -------
            list[tuple[float, float]]
                Bounds in same order as get_array().
            
            Example
            -------
            >>> manager.get_bounds()
            [(1e-6, 1e6), (1e-6, 1e6), (1e-6, 1e6)]
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
            """
            if len(array) != len(self._optimized_keys):
                raise ValueError("Provided buffered array length does not match the number of optimized hyperparameters")
            
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
            """Get full history of accepted iterations."""
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

        