## Hyperparameter Architecture: Model ↔ Optimizer Interface

You've identified the core tension correctly. Let me break down the **three distinct spaces** that need to be coordinated:

### The Three Spaces

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. MODEL SPACE (dict-based, named)                              │
│    - Keys: "sigma_st", "tau_iid", "tau_queen", etc.             │
│    - Values: Hyperparameter objects with .name, .value, .bounds │
│    - Used by: Model.assemble_prior_precision_matrix()           │
│    - Order: Irrelevant (dict lookup by key)                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. MANAGER SPACE (ordered array + mapping)                      │
│    - Keys: Ordered list ["tau_iid", "tau_queen", "sigma_st"]    │
│    - Values: np.ndarray [τ₁, τ₂, τ₃]                            │
│    - Used by: HyperparameterManager internally                  │
│    - Order: CRITICAL (index → key mapping)                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. OPTIMIZER SPACE (scipy array interface)                      │
│    - Keys: Implicit indices [0, 1, 2, ...]                      │
│    - Values: np.ndarray [τ₁, τ₂, τ₃]                            │
│    - Used by: scipy.optimize.minimize                           │
│    - Order: CRITICAL (same as manager)                          │
└─────────────────────────────────────────────────────────────────┘
```

### The Core Challenge

The **HyperparameterManager** is the **bilingual bridge** between these spaces:

1. **Model → Manager**: "I need hyperparameters as a dict with keys like 'sigma_st'"
2. **Optimizer → Manager**: "I need hyperparameters as an ordered array [τ₁, τ₂, τ₃]"
3. **Manager → Model**: "Here's the dict your model expects"
4. **Manager → Optimizer**: "Here's the array your optimizer expects"

---

## Detailed Workflow: Accepted vs Rejected Iterations

Let me sketch the complete workflow with state transitions:

### Initial Setup

```python
# Step 1: Model defines hyperparameters with names
model = GenomicModel(config)
# model.hyperparameters = {
#     "tau_iid": Hyperparameter(name="tau_iid", value=1.0, bounds=(1e-6, 1e6)),
#     "tau_queen": Hyperparameter(name="tau_queen", value=1.0, bounds=(1e-6, 1e6)),
#     "prec_regression": Hyperparameter(name="prec_regression", value=1.0, bounds=(1e-6, 1e6))
# }

# Step 2: Manager creates ordered mapping
hp_manager = HyperparameterManager(
    hyperparameters=model.hyperparameters,
    order=["tau_iid", "tau_queen", "prec_regression"]  # CRITICAL: defines array index → key mapping
)

# Manager internal state:
# hp_manager._key_to_index = {"tau_iid": 0, "tau_queen": 1, "prec_regression": 2}
# hp_manager._index_to_key = {0: "tau_iid", 1: "tau_queen", 2: "prec_regression"}
# hp_manager._array = np.array([1.0, 1.0, 1.0])  # current accepted values
# hp_manager._history = []  # list of accepted iterations
# hp_manager._buffer = None  # buffered perturbed values (not yet accepted)
```

### Optimization Loop with Rejected Iterations

Here's what happens during optimization:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Iteration 0: Initial point                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ scipy calls: wrapped_objective(x=[1.0, 1.0, 1.0])                          │
│   ↓                                                                          │
│   1. Manager buffers this as current "tentative" point                     │
│      hp_manager._buffer = np.array([1.0, 1.0, 1.0])                       │
│   ↓                                                                          │
│   2. objective_fn converts buffer → dict for model:                        │
│      hp_dict = {"tau_iid": 1.0, "tau_queen": 1.0, "prec_regression": 1.0} │
│   ↓                                                                          │
│   3. Model assembles Q_prior with hp_dict                                  │
│   ↓                                                                          │
│   4. Compute INLA approximation f = -123.45                                │
│   ↓                                                                          │
│   5. Return f to scipy                                                     │
│   ↓                                                                          │
│   6. scipy accepts point (first iteration, always accepted)               │
│   ↓                                                                          │
│   7. callback(xk=[1.0, 1.0, 1.0]) triggers:                               │
│      hp_manager._history.append((0, [1.0, 1.0, 1.0], f=-123.45))          │
│      hp_manager._array = np.array([1.0, 1.0, 1.0])  # update accepted     │
│      hp_manager._buffer = None  # clear buffer                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Iteration 1: Rejected perturbation                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ scipy computes finite difference:                                           │
│   x_perturbed = [1.0001, 1.0, 1.0]  (h=1e-4)                               │
│   ↓                                                                          │
│   1. Manager buffers this perturbation:                                    │
│      hp_manager._buffer = np.array([1.0001, 1.0, 1.0])                    │
│   ↓                                                                          │
│   2. objective_fn converts buffer → dict:                                  │
│      hp_dict = {"tau_iid": 1.0001, "tau_queen": 1.0, ...}                 │
│   ↓                                                                          │
│   3. Model assembles Q_prior with hp_dict                                  │
│   ↓                                                                          │
│   4. Compute f = -123.44 (slightly worse)                                  │
│   ↓                                                                          │
│   5. Return f to scipy                                                     │
│   ↓                                                                          │
│   6. scipy REJECTS point (line search or bound constraint)                │
│   ↓                                                                          │
│   7. NO callback called — buffer remains unchanged                         │
│   ↓                                                                          │
│   8. Next perturbation: x_perturbed = [0.9999, 1.0, 1.0]                   │
│   ↓                                                                          │
│   9. Manager updates buffer: hp_manager._buffer = [0.9999, 1.0, 1.0]     │
│   ↓                                                                          │
│   10. objective_fn uses buffer → dict                                      │
│   ↓                                                                          │
│   11. Compute f = -123.40 (better!)                                        │
│   ↓                                                                          │
│   12. Return f to scipy                                                    │
│   ↓                                                                          │
│   13. scipy ACCEPTS point                                                  │
│   ↓                                                                          │
│   14. callback(xk=[0.9999, 1.0, 1.0]) triggers:                           │
│       hp_manager._history.append((1, [0.9999, 1.0, 1.0], f=-123.40))     │
│       hp_manager._array = np.array([0.9999, 1.0, 1.0])  # update accepted │
│       hp_manager._buffer = None  # clear buffer                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## HyperparameterManager Interface Specification

Here's the clean interface that resolves all your concerns:

```python
class Hyperparameter:
    """Represents a single hyperparameter with metadata."""
    name: str              # Unique identifier (e.g., "sigma_st", "tau_iid")
    value: float           # Current value
    is_fixed: bool         # If True, not optimized
    bounds: Bounds         # (lb, ub) for optimization
    # NOTE: name is the key — enforced to match dict key in model.hyperparameters


class HyperparameterManager:
    """
    Bridge between Model (dict-based) and Optimizer (array-based).
    
    Key design principles:
    1. Manager owns the ordered array representation
    2. Only ACCEPTED iterations update manager._array and _history
    3. Buffer holds tentative/perturbed values (not yet accepted)
    4. name field in Hyperparameter must match dict key in Model
    """
    
    # ── Initialization ─────────────────────────────────────────────────────
    def __init__(
        self,
        hyperparameters: dict[str, Hyperparameter],  # Model's dict
        order: list[str] | None = None,              # Explicit ordering
        track_history: bool = True,
        checkpoint_frequency: int = 10
    ):
        """
        Initialize manager from Model's hyperparameters.
        
        Parameters
        ----------
        hyperparameters : dict[str, Hyperparameter]
            Model's hyperparameters dict. Keys are used as names.
        order : list[str], optional
            Explicit order of hyperparameters in array. If None,
            uses dict insertion order (Python 3.7+).
        track_history : bool, default True
            If False, skip history tracking (for performance).
        checkpoint_frequency : int, default 10
            How often to checkpoint history to disk.
        """
        # Validate: all names match dict keys
        for key, hp in hyperparameters.items():
            assert hp.name == key, f"Hyperparameter.name '{hp.name}' must match dict key '{key}'"
        
        # Store metadata
        self._hyperparameters = hyperparameters
        self._order = order or list(hyperparameters.keys())
        
        # Create mapping (CRITICAL: index ↔ key)
        self._key_to_index = {key: i for i, key in enumerate(self._order)}
        self._index_to_key = {i: key for key, i in self._key_to_index.items()}
        
        # Initialize array from initial values
        self._array = np.array([hyperparameters[key].value for key in self._order])
        
        # State management
        self._buffer: np.ndarray | None = None  # Tentative values (not accepted)
        self._iteration = 0
        self._track_history = track_history
        self._checkpoint_frequency = checkpoint_frequency
        self._history: list[tuple[int, np.ndarray, float]] = []  # (iter, array, f)
    
    # ── Array ↔ Dict Conversion ────────────────────────────────────────────
    def array_to_dict(self, array: np.ndarray | None = None) -> dict[str, float]:
        """
        Convert array to dict with model-compatible keys.
        
        Parameters
        ----------
        array : np.ndarray, optional
            Array to convert. If None, uses current accepted values.
        
        Returns
        -------
        dict[str, float]
            Dict with keys matching Model.hyperparameters.
        
        Example
        -------
        >>> manager.array_to_dict(np.array([1.0, 2.0]))
        {"tau_iid": 1.0, "tau_queen": 2.0}
        """
        if array is None:
            array = self._array  # Use accepted values
        
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
        
        Example
        -------
        >>> manager.dict_to_array({"tau_queen": 2.0, "tau_iid": 1.0})
        np.array([1.0, 2.0])  # Order follows manager._order
        """
        return np.array([d[key] for key in self._order])
    
    # ── Getters for Optimizer ──────────────────────────────────────────────
    def get_array(self) -> np.ndarray:
        """Get current accepted values as array (for scipy x0)."""
        return self._array.copy()
    
    def get_bounds(self) -> list[tuple[float, float]]:
        """
        Get bounds as list of (lb, ub) tuples matching array order.
        
        Returns
        -------
        list[tuple[float, float]]
            Bounds in same order as get_array().
        
        Example
        -------
        >>> manager.get_bounds()
        [(1e-6, 1e6), (1e-6, 1e6), (1e-6, 1e6)]
        """
        return [self._hyperparameters[key].bounds for key in self._order]
    
    # ── State Management (Buffered) ────────────────────────────────────────
    def buffer_update(self, array: np.ndarray) -> None:
        """
        Buffer a perturbed array (tentative, not yet accepted).
        
        Called by optimizer BEFORE each objective evaluation.
        The buffer is only committed on accepted iterations.
        """
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
        
        # Record history if enabled
        if self._track_history:
            if len(self._history) % self._checkpoint_frequency == 0:
                self._checkpoint_history()
            
            self._history.append((self._iteration, self._array.copy(), f or -np.inf))
        
        # Cleanup
        self._buffer = None
        self._iteration += 1
    
    def reject_buffer(self) -> None:
        """
        Discard buffered values (scipy rejected the perturbation).
        
        No state change — buffer remains for next perturbation.
        """
        self._buffer = None  # Clear buffer, wait for next perturbation
    
    # ── History Access ─────────────────────────────────────────────────────
    def get_history(self) -> list[tuple[int, np.ndarray, float]]:
        """Get full history of accepted iterations."""
        return self._history.copy()
    
    def get_last_iteration(self) -> int:
        """Get last accepted iteration number."""
        return self._iteration - 1 if self._history else 0
    
    # ── Model Integration ──────────────────────────────────────────────────
    def update_model(self, model: StatisticalModel) -> None:
        """
        Update model's hyperparameters with manager's current values.
        
        Called after optimization completes.
        """
        hp_dict = self.array_to_dict()
        
        for key, value in hp_dict.items():
            model.hyperparameters[key].value = value
    
    # ── Checkpointing ──────────────────────────────────────────────────────
    def _checkpoint_history(self) -> None:
        """Save history to disk (implementation-specific)."""
        # Save self._history to disk (e.g., np.save, pickle, etc.)
        pass
    
    def checkpoint(self, path: Path) -> None:
        """Save entire manager state to disk."""
        state = {
            "array": self._array,
            "iteration": self._iteration,
            "history": self._history,
            "order": self._order
        }
        np.save(path, state, allow_pickle=True)
    
    @classmethod
    def load_checkpoint(cls, path: Path, hyperparameters: dict[str, Hyperparameter]) -> "HyperparameterManager":
        """Restore manager from checkpoint."""
        state = np.load(path, allow_pickle=True).item()
        
        manager = cls(hyperparameters=hyperparameters, order=state["order"])
        manager._array = state["array"]
        manager._iteration = state["iteration"]
        manager._history = state["history"]
        
        return manager
```

---

## Optimizer Integration Pattern

Here's how the optimizer wraps everything:

```python
def optimize(
    model: StatisticalModel,
    hp_manager: HyperparameterManager,
    objective_fn: callable,
    gradient_strategy: GradientStrategy,
    callback: callable | None = None
) -> OptimizeResult:
    """
    Optimize hyperparameters using INLA objective function.
    
    Parameters
    ----------
    model : StatisticalModel
        The statistical model to optimize.
    hp_manager : HyperparameterManager
        Bridge between model (dict) and optimizer (array).
    objective_fn : callable
        INLA objective function f(hp_values, model) → float.
    gradient_strategy : GradientStrategy
        Strategy for computing gradients (finite diff, AD, etc.).
    callback : callable, optional
        User-defined callback after each accepted iteration.
    
    Returns
    -------
    OptimizeResult
        scipy.optimize.OptimizeResult.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    x0 = hp_manager.get_array()
    bounds = hp_manager.get_bounds()
    
    # ── Wrapped Objective (pure function) ──────────────────────────────────
    def wrapped_objective(x: np.ndarray) -> float:
        """
        Pure objective function — no state mutation.
        
        Uses buffered values (not committed to manager yet).
        """
        # Buffer the perturbation (tentative, not accepted)
        hp_manager.buffer_update(x)
        
        # Convert to dict for model
        hp_dict = hp_manager.array_to_dict(array=x)
        
        # Compute objective (model uses hp_dict)
        return objective_fn(hp_dict, model)
    
    # ── Wrapped Jacobian (if needed) ───────────────────────────────────────
    def wrapped_jacobian(x: np.ndarray) -> np.ndarray:
        """Compute gradient using gradient strategy."""
        hp_manager.buffer_update(x)
        return gradient_strategy.compute(objective_fn, x, model)
    
    # ── Scipy Callback (accepted iterations only) ──────────────────────────
    def scipy_callback(xk: np.ndarray):
        """
        Called by scipy AFTER each accepted iteration.
        
        Commits buffered values to manager state.
        """
        # Commit the accepted point
        hp_manager.commit_buffer()
        
        # User callback for logging
        if callback is not None:
            callback(xk, hp_manager.array_to_dict())
    
    # ── Run Optimization ───────────────────────────────────────────────────
    result = minimize(
        fun=wrapped_objective,
        x0=x0,
        jac=wrapped_jacobian if gradient_strategy.supports_autodiff else None,
        bounds=bounds,
        method='L-BFGS-B',
        callback=scipy_callback
    )
    
    # ── Final Update ───────────────────────────────────────────────────────
    # Commit final point
    hp_manager.buffer_update(result.x)
    hp_manager.commit_buffer()
    
    # Update model with optimized hyperparameters
    hp_manager.update_model(model)
    
    return result
```

---

## Key Design Principles Resolving Your Concerns

### 1. **Name/Key Enforcement**
```python
# In __init__:
for key, hp in hyperparameters.items():
    assert hp.name == key, f"Hyperparameter.name '{hp.name}' must match dict key '{key}'"
```
- Model defines `hyperparameters = {"sigma_st": Hyperparameter(name="sigma_st", ...)}`
- Manager validates that `hp.name == key`
- Array index `i` → key `self._order[i]` → `Hyperparameter.name`

### 2. **Only Accepted Iterations Update State**
```python
# Optimizer flow:
wrapped_objective(x_perturbed)  # buffers, computes f, returns f
# ↓
scipy REJECTS → no callback → buffer discarded
# OR
scipy ACCEPTS → callback(xk) → commit_buffer() → state updated
```
- Rejected perturbations never touch `_array` or `_history`
- Buffer is cleared on rejection

### 3. **Model Gets Dict, Optimizer Gets Array**
```python
# Model interface:
model.assemble_prior_precision_matrix(hp_dict)  # {"sigma_st": 1.0, ...}

# Optimizer interface:
minimize(wrapped_objective, x0=array)  # [1.0, 2.0, ...]

# Manager bridges both:
hp_dict = manager.array_to_dict(array)  # Convert for model
array = manager.dict_to_array(hp_dict)  # Convert for optimizer
```

### 4. **Order Preservation**
```python
# Manager stores explicit order:
self._order = ["tau_iid", "tau_queen", "prec_regression"]
self._key_to_index = {"tau_iid": 0, "tau_queen": 1, "prec_regression": 2}

# Array index 0 always → "tau_iid", index 1 → "tau_queen", etc.
# This order is fixed at initialization
```

### 5. **Buffered Updates**
```python
# Tentative perturbations:
hp_manager.buffer_update(x_perturbed)  # Store in _buffer
objective_fn(...)  # Use _buffer for conversion

# Accepted:
hp_manager.commit_buffer()  # _array = _buffer, clear _buffer

# Rejected:
hp_manager.reject_buffer()  # Clear _buffer, wait for next
```

---

## Summary: The Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Model.hyperparameters = {"tau_iid": HP(name="tau_iid", value=1.0), ...}    │
│                                                                              │
│ Manager = HyperparameterManager(                                            │
│     hyperparameters=model.hyperparameters,                                  │
│     order=["tau_iid", "tau_queen", "prec_regression"]                       │
│ )                                                                            │
│                                                                              │
│ Manager._key_to_index = {"tau_iid": 0, "tau_queen": 1, "prec_regression": 2}│
│ Manager._array = [1.0, 1.0, 1.0]  # accepted values                        │
│ Manager._buffer = None  # tentative values (not yet accepted)              │
└─────────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. OPTIMIZATION LOOP                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ scipy calls wrapped_objective(x=[1.0001, 1.0, 1.0])                        │
│   ├─> Manager.buffer_update([1.0001, 1.0, 1.0])  # Store in _buffer       │
│   ├─> hp_dict = Manager.array_to_dict([1.0001, 1.0, 1.0])                 │
│   │     → {"tau_iid": 1.0001, "tau_queen": 1.0, "prec_regression": 1.0}   │
│   ├─> Model.assemble_prior_precision_matrix(hp_dict)  # Use dict          │
│   ├─> Compute f = -123.44                                                   │
│   └─> Return f to scipy                                                     │
│                                                                              │
│ scipy REJECTS point (worse objective)                                       │
│   └─> NO callback → Manager._buffer discarded, _array unchanged            │
│                                                                              │
│ scipy calls wrapped_objective(x=[0.9999, 1.0, 1.0])                        │
│   ├─> Manager.buffer_update([0.9999, 1.0, 1.0])                           │
│   ├─> hp_dict = {"tau_iid": 0.9999, "tau_queen": 1.0, ...}                │
│   ├─> Model.assemble_prior_precision_matrix(hp_dict)                       │
│   ├─> Compute f = -123.40 (better!)                                         │
│   └─> Return f to scipy                                                     │
│                                                                              │
│ scipy ACCEPTS point                                                         │
│   └─> callback(xk=[0.9999, 1.0, 1.0]) → Manager.commit_buffer()           │
│         ├─> Manager._array = [0.9999, 1.0, 1.0]  # Update accepted         │
│         ├─> Manager._history.append((iter, [0.9999, 1.0, 1.0], f))        │
│         └─> Manager._buffer = None  # Clear buffer                         │
└─────────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. OPTIMIZATION COMPLETE                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ result = minimize(...)                                                      │
│                                                                              │
│ hp_manager.commit_buffer()  # Final point                                  │
│ hp_manager.update_model(model)  # Write optimized HPs to model              │
│                                                                              │
│ Model.hyperparameters now has optimized values:                             │
│ {"tau_iid": 0.9999, "tau_queen": 1.0, "prec_regression": 1.0}              │
└─────────────────────────────────────────────────────────────────────────────┘
```

This architecture cleanly separates concerns:
- **Model**: Works with dict `{"sigma_st": 1.0, ...}`
- **Optimizer**: Works with array `[1.0, 2.0, ...]`
- **Manager**: Bridges both, tracks only accepted iterations
- **Buffer**: Holds tentative values until acceptance