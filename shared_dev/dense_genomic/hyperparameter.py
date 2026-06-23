import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from copy import deepcopy


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter."""
    name: str
    initial_value: float
    # No bounds - all parameters live in (-inf, +inf)
    
    def __post_init__(self):
        # Ensure name is a string
        if not isinstance(self.name, str):
            raise TypeError(f"Parameter name must be string, got {type(self.name)}")


class HyperparameterManager:
    """
    Manages hyperparameters for Bayesian optimization with scipy.
    
    This class provides:
    - Read-only named access to hyperparameters
    - Conversion to/from 1D numpy arrays for scipy.optimize.minimize
    - Automatic history tracking with iteration numbers
    - Only optimizer can modify values via update_from_array()
    
    Design principles:
    - Hyperparameters are read-only for practitioners
    - Only scipy optimizer modifies values through array interface
    - History tracking is automatic and stores all successful updates
    """
    
    def __init__(self, configs: List[HyperparameterConfig], track_history: bool = True):
        """
        Initialize the hyperparameter manager.
        
        Args:
            configs: List of hyperparameter configurations
            track_history: Whether to track history (default: True)
        """
        # Use OrderedDict to preserve order for consistent array conversion
        self._configs = OrderedDict()
        self._values = OrderedDict()
        self._param_names = []
        
        for config in configs:
            if config.name in self._configs:
                raise ValueError(f"Duplicate hyperparameter name: {config.name}")
            self._configs[config.name] = config
            self._values[config.name] = config.initial_value
            self._param_names.append(config.name)
        
        # History tracking
        self._track_history = track_history
        self._history: List[Dict[str, Any]] = [] if track_history else None
        self._history_array: Optional[np.ndarray] = None  # Dense 2D array (iterations x params)
        self._iteration_counter = 0
        self._last_update_array: Optional[np.ndarray] = None  # Store last array used
        
        # Record initial state if tracking
        if self._track_history:
            self._append_to_history(iteration=0, note="initialization")
    
    # ========== READ-ONLY ACCESS ==========
    
    def __getitem__(self, name: str) -> float:
        """
        Read-only access to hyperparameter value by name.
        
        Raises:
            KeyError: If hyperparameter name doesn't exist
        """
        if name not in self._values:
            raise KeyError(f"Hyperparameter '{name}' not found. Available: {list(self._values.keys())}")
        return self._values[name]
    
    def __contains__(self, name: str) -> bool:
        """Check if hyperparameter exists."""
        return name in self._values
    
    def __len__(self) -> int:
        """Number of hyperparameters."""
        return len(self._values)
    
    def __iter__(self):
        """Iterate over hyperparameter names in order."""
        return iter(self._param_names)
    
    def get(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Safe read-only access with default value."""
        return self._values.get(name, default)
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current values as a dictionary (copy)."""
        return {name: self._values[name] for name in self._param_names}
    
    def get_names(self) -> List[str]:
        """Get list of hyperparameter names in order."""
        return self._param_names.copy()
    
    # ========== ARRAY INTERFACE (FOR OPTIMIZER) ==========
    
    def get_array(self) -> np.ndarray:
        """
        Get hyperparameters as a 1D numpy array for optimizer.
        
        Returns:
            1D numpy array of hyperparameter values in consistent order
        """
        return np.array([self._values[name] for name in self._param_names], dtype=float)
    
    def _set_array(self, array: np.ndarray) -> None:
        """
        Internal method to set values from array.
        Only called by update_from_array().
        
        Args:
            array: 1D array of values in same order as get_array()
        """
        if len(array) != len(self._param_names):
            raise ValueError(
                f"Expected array of length {len(self._param_names)}, got {len(array)}"
            )
        
        # Update values (no validation since we're in (-inf, +inf))
        for i, name in enumerate(self._param_names):
            self._values[name] = float(array[i])
        
        self._last_update_array = array.copy()
    
    def update_from_array(self, array: np.ndarray, 
                          iteration: Optional[int] = None,
                          note: str = "") -> None:
        """
        Update hyperparameters from array and record history.
        
        This is the ONLY method that modifies hyperparameter values.
        Designed to be called by scipy.optimize.minimize after each iteration.
        
        Args:
            array: 1D array of values in same order as get_array()
            iteration: Optional iteration number (auto-increments if not provided)
            note: Optional note for this update (e.g., "after gradient step")
        """
        self._set_array(array)
        
        if self._track_history:
            # Use provided iteration or auto-increment
            if iteration is None:
                iteration = self._iteration_counter
            self._append_to_history(iteration, note)
    
    def _append_to_history(self, iteration: int, note: str = "") -> None:
        """Append current values to history."""
        if not self._track_history:
            return
        
        # Store as dict with metadata
        current_values = {name: self._values[name] for name in self._param_names}
        entry = {
            'iteration': iteration,
            'values': current_values,
            'note': note,
            'array': self._last_update_array.copy() if self._last_update_array is not None else None
        }
        self._history.append(entry)
        
        # Update dense array for efficient access
        array_values = self.get_array()
        if self._history_array is None:
            self._history_array = np.array([array_values])
        else:
            self._history_array = np.vstack([self._history_array, array_values])
        
        self._iteration_counter = iteration + 1  # Next iteration number
    
    # ========== HISTORY ACCESS ==========
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get full history as list of dictionaries with metadata.
        
        Returns:
            List of dicts with keys: 'iteration', 'values', 'note', 'array'
        """
        if not self._track_history:
            raise ValueError("History tracking is disabled")
        return deepcopy(self._history)
    
    def get_history_array(self) -> np.ndarray:
        """
        Get history as dense 2D numpy array (iterations x hyperparameters).
        
        Returns:
            2D array where rows are iterations and columns are hyperparameters
            in the same order as get_array()
        """
        if not self._track_history:
            raise ValueError("History tracking is disabled")
        return self._history_array.copy() if self._history_array is not None else np.empty((0, len(self)))
    
    def get_last_iteration(self) -> int:
        """Get the number of recorded iterations."""
        return self._iteration_counter
    
    def get_last_update(self) -> Optional[Dict[str, Any]]:
        """Get the most recent update entry."""
        if not self._track_history or not self._history:
            return None
        return deepcopy(self._history[-1])
    
    def get_historical_array(self, iteration: int) -> np.ndarray:
        """
        Get the full array of hyperparameters at a specific iteration.
        
        Args:
            iteration: Iteration number (0-indexed)
            
        Returns:
            1D numpy array of hyperparameter values at that iteration
        """
        if not self._track_history:
            raise ValueError("History tracking is disabled")
        if iteration < 0 or iteration >= len(self._history):
            raise IndexError(f"Iteration {iteration} out of range (0 to {len(self._history)-1})")
        
        # Return from history array if available, otherwise reconstruct
        if self._history_array is not None and iteration < len(self._history_array):
            return self._history_array[iteration].copy()
        else:
            # Fallback: reconstruct from values dict
            return np.array([self._history[iteration]['values'][name] 
                           for name in self._param_names])
    
    def get_historical_dict(self, iteration: int) -> Dict[str, float]:
        """
        Get all hyperparameter values at a specific iteration as a dict.
        
        Args:
            iteration: Iteration number (0-indexed)
            
        Returns:
            Dictionary mapping hyperparameter names to values at that iteration
        """
        if not self._track_history:
            raise ValueError("History tracking is disabled")
        if iteration < 0 or iteration >= len(self._history):
            raise IndexError(f"Iteration {iteration} out of range (0 to {len(self._history)-1})")
        
        return deepcopy(self._history[iteration]['values'])
    
    def get_historical_by_name(self, name: str) -> np.ndarray:
        """
        Get the entire history of a specific hyperparameter as an array.
        
        Args:
            name: Name of the hyperparameter
            
        Returns:
            1D numpy array of values across all iterations
        """
        if name not in self._values:
            raise KeyError(f"Hyperparameter '{name}' not found")
        if not self._track_history:
            raise ValueError("History tracking is disabled")
        
        # Get the index of this parameter
        param_idx = self._param_names.index(name)
        
        # Extract from history array
        if self._history_array is not None and len(self._history_array) > 0:
            return self._history_array[:, param_idx].copy()
        else:
            # Fallback: extract from history dicts
            return np.array([entry['values'][name] for entry in self._history])
    
    def get_history_dataframe(self):
        """
        Get history as a pandas DataFrame (if pandas is available).
        
        Returns:
            pandas DataFrame with hyperparameters as columns and iterations as rows
        """
        try:
            import pandas as pd
            df = pd.DataFrame(self._history_array, columns=self._param_names)
            df.index.name = 'iteration'
            return df
        except ImportError:
            raise ImportError("pandas is not installed. Install it with: pip install pandas")
    
    # ========== UTILITY METHODS ==========
    
    def reset(self) -> None:
        """
        Reset hyperparameters to initial values and clear history.
        Useful for restarting optimization.
        """
        # Reset values
        for name in self._param_names:
            self._values[name] = self._configs[name].initial_value
        
        # Reset history
        if self._track_history:
            self._history = []
            self._history_array = None
            self._iteration_counter = 0
            self._last_update_array = None
            # Record initial state
            self._append_to_history(iteration=0, note="reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for inspection/export."""
        return {
            'configs': [
                {
                    'name': c.name,
                    'initial_value': c.initial_value,
                }
                for c in self._configs.values()
            ],
            'current_values': self.get_current_values(),
            'param_names': self._param_names,
            'track_history': self._track_history,
            'last_iteration': self._iteration_counter,
            'history': self._history if self._track_history else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterManager':
        """Reconstruct from dictionary."""
        configs = [HyperparameterConfig(**c_data) for c_data in data['configs']]
        manager = cls(configs, track_history=data.get('track_history', True))
        
        # Restore values if needed
        if 'current_values' in data:
            # Use internal method to bypass read-only restriction
            for name, value in data['current_values'].items():
                if name in manager._values:
                    manager._values[name] = float(value)
        
        # Restore history if present
        if data.get('track_history', True) and 'history' in data and data['history'] is not None:
            manager._history = deepcopy(data['history'])
            # Rebuild history array
            if manager._history:
                # Extract arrays from history entries
                arrays = []
                for entry in manager._history:
                    if 'array' in entry and entry['array'] is not None:
                        arrays.append(entry['array'])
                    else:
                        # Fallback: reconstruct from values dict
                        values = [entry['values'][name] for name in manager._param_names]
                        arrays.append(np.array(values))
                
                if arrays:
                    manager._history_array = np.vstack(arrays)
                manager._iteration_counter = len(manager._history)
        
        return manager
    
    def __repr__(self) -> str:
        """String representation showing current values and status."""
        values_str = ', '.join([f"{name}={self._values[name]:.6g}" for name in self._param_names])
        history_info = f", {self._iteration_counter} iterations" if self._track_history else ""
        return f"HyperparameterManager({values_str}{history_info})"
    
    def __str__(self) -> str:
        """Pretty string representation."""
        lines = ["HyperparameterManager:"]
        lines.append("  Current values:")
        for name in self._param_names:
            lines.append(f"    {name}: {self._values[name]:.6g}")
        if self._track_history:
            lines.append(f"  History: {self._iteration_counter} iterations recorded")
        return "\n".join(lines)


# ========== EXAMPLE USAGE WITH SCIPY ==========

def example_usage():
    """Demonstrate usage with scipy.optimize.minimize."""
    import numpy as np
    from scipy.optimize import minimize
    
    # Define hyperparameters
    configs = [
        HyperparameterConfig("learning_rate", 0.01),
        HyperparameterConfig("batch_size", 32),
        HyperparameterConfig("dropout_rate", 0.5),
        HyperparameterConfig("l2_regularization", 1e-4),
    ]
    
    # Initialize manager
    hp = HyperparameterManager(configs, track_history=True)
    print("Initial state:")
    print(hp)
    print(f"\nArray for optimizer: {hp.get_array()}")
    
    # Define objective function that uses hyperparameters
    def objective(x):
        # Update hyperparameters with optimizer's proposed values
        hp.update_from_array(x, note=f"iteration_{hp.get_last_iteration()}")
        
        # Now use the hyperparameters in your objective
        lr = hp['learning_rate']
        bs = hp['batch_size']
        dr = hp['dropout_rate']
        l2 = hp['l2_regularization']
        
        # CORRECTED: Use a loss function that's well-defined for all real values
        # This is a quadratic loss with a known minimum at the initial values
        # All terms are squares, so they're always non-negative and well-defined
        loss = (
            (lr - 0.01)**2 * 1000 +           # Scale learning rate appropriately
            (bs - 32)**2 * 0.1 +               # Batch size term
            (dr - 0.5)**2 * 10 +               # Dropout term
            (np.log(np.abs(l2) + 1e-8) - np.log(1e-4))**2  # Log-scale for L2
        )
        
        # Print progress
        print(f"Iter {hp.get_last_iteration()-1}: loss={loss:.6f}, lr={lr:.6f}, bs={bs:.1f}, l2={l2:.2e}")
        return loss
    
    # Initial guess (must be same length as hyperparameters)
    x0 = hp.get_array()
    
    # Run optimization
    print("\n--- Starting optimization ---")
    result = minimize(
        objective, 
        x0, 
        method='L-BFGS-B',
        options={'maxiter': 20, 'disp': False}
    )
    
    print(f"\n--- Optimization complete ---")
    print(f"Success: {result.success}")
    print(f"Final loss: {result.fun:.6f}")
    print(f"Number of iterations: {result.nit}")
    print(f"\nFinal hyperparameters:")
    print(hp)
    
    # Access history
    history_array = hp.get_history_array()
    print(f"\nHistory shape: {history_array.shape} (iterations x hyperparameters)")
    print("Last 5 iterations (values):")
    last_5 = history_array[-5:] if len(history_array) >= 5 else history_array
    for i, row in enumerate(last_5):
        iter_num = len(history_array) - len(last_5) + i
        print(f"  Iter {iter_num}: {row}")
    
    # Example of read-only access
    print(f"\nRead learning_rate: {hp['learning_rate']:.6f}")
    print(f"Read batch_size: {hp['batch_size']:.1f}")
    
    # This would raise an error (read-only):
    try:
        hp['learning_rate'] = 0.02
    except TypeError as e:
        print(f"\nCorrectly prevented modification: {e}")

if __name__ == "__main__":
    example_usage()
