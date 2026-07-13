# Design Decisions and Tracking

## Architecture Decisions


### 2. Optimization is External to StatisticalModel

**Decision:** The INLA optimization is a standalone function/module, not a method on `StatisticalModel`.

**Rationale:**
- Tying `optimize()` to `StatisticalModel` would couple INLA machinery to the abstract base class
- Every model subclass would inherit INLA-specific methods
- The model's public API would mix statistical concepts with computational ones
- Swapping optimizers (MCMC, variational) would require modifying the model

**Bridging Layer:**
```python
# The objective function receives the model explicitly
def objective(hp_values: np.ndarray, model: StatisticalModel) -> float:
    hp_named = hp_manager.get_named_values()  # array → dict
    Q_prior = model.assemble_prior_precision_matrix(hp_named)
    A = model.assemble_design_matrix()
    # ... compute INLA approximation
    return f
```

The `HyperparameterManager` is the bridge between scipy's array-based interface and the model's dict-based interface.

### 3. Gradient Strategy: Separate from Objective

**Decision:** Finite differences is the initial gradient strategy, architected so AD can be swapped in later.

**Design:**
```
GradientStrategy (abstract)
├── finite_difference(objective, x, model, stencil=3) → grad
├── backward_difference(objective, x, model, stencil=3) → grad
└── autodiff(objective, x, model) → grad
```

The objective function always returns `(f, terms_dict)` — the same for both strategies. The gradient strategy decides how to compute the Jacobian:
- **Finite diff:** calls `objective()` N times with perturbed x
- **AD:** wraps internal computation in an AD-traceable backend

The 4 INLA terms share computation internally (prior precision → conditional precision). This sharing is an implementation detail of `objective()`, not exposed in the gradient interface.

### 4. Checkpointing

**Decision:** Checkpoint the `HyperparameterManager` state, not the model.

**Rationale:**
- The model is a passive recipient of optimized HPs
- Checkpoint = manager state + iteration info
- Resume = restore manager → call optimize() → manager updates model HPs at end


### 5. HyperparameterManager and accepted points only

#### Option A: Manager Updates on Every `objective()` Call (Current)

**Pros:**
- Simple — manager state always reflects the current evaluation point
- The objective function can always read the "correct" HP values via `hp['name']`
- No state management in the optimizer

**Cons:**
- Scipy evaluates `objective()` at rejected perturbations (finite diff, line search) — the manager's state gets polluted
- History tracking records rejected points as "iterations"
- If the objective is expensive, the manager is doing work for points that will be rejected

#### Option B: Manager Updates Only on Accepted Points (Recommended)

**Pros:**
- Manager state reflects only valid optimization progress
- History is clean — only accepted iterations
- The objective function receives HP values as an explicit parameter (no hidden state mutation)
- Cleaner separation: the optimizer owns the acceptance/rejection logic

**Cons:**
- The objective function signature changes — instead of `hp['name']`, you'd pass HP values explicitly: `objective(hp_values, model)` where `hp_values` is already the accepted array
- Slightly more plumbing in the optimizer wrapper to call `manager.update_from_array()` after acceptance
- The objective function can't use `hp['name']` for reading — it must use the passed array or a named dict

#### Recommendation: Option B

The objective function should be a pure function: `objective(hp_values, model) -> f`. No hidden state mutation. The HyperparameterManager's `update_from_array` is called by the **optimizer wrapper** after scipy accepts a new point. This means:

```python
def optimize(model, hp_manager, objective_fn, gradient_strategy, callback=None):
    x = hp_manager.get_array()
    
    def wrapped_objective(x):
        # Pure function — no state mutation
        return objective_fn(x, model)
    
    def callback(xk):
        # Called by scipy after each accepted iteration
        hp_manager.update_from_array(xk, note=f"accepted_iter_{hp_manager.get_last_iteration()}")
    
    result = minimize(wrapped_objective, x, callback=callback, ...)
    
    # Final update
    hp_manager.update_from_array(result.x, note="optimization_complete")
    
    # Write optimized HPs back to model
    model.hyperparameters = hp_manager.get_current_values()
```

This makes the lifecycle explicit:
```
scipy calls wrapped_objective(x) → pure computation, no side effects
scipy accepts x → callback(x) → manager updates → history recorded
scipy converges → manager writes to model
```

---


**Usage pattern in the optimization loop:**

```python
cache = CacheManager(tier_config)

# Register components once during model setup
for name, component in model.prior_components.items():
    cache.register(component, name)

def objective(hp_values, model):
    # Ensure components are in active memory
    for key in cache.keys():
        cache.restore(key)
    
    # Assemble Q_prior (components are in tier_0)
    Q_prior = model.assemble_prior_precision_matrix(hp_named)
    
    # Components are no longer needed — evict them
    cache.evict_all()
    
    # Now build Q_cond — Q_prior and Q_cond coexist in memory
    Q_cond = build_conditional(Q_prior, ...)
    
    # Compute the INLA approximation
    f = compute_inla_approximation(Q_prior, Q_cond, ...)
    
    return f
```

**Key design points:**
- The model doesn't know about the cache — it just assembles matrices
- The cache manager owns the eviction/loading lifecycle
- Components implement `store_in_cache()` and `restore_from_cache()` as a protocol
- The optimization loop (or INLA module) decides when to evict

---

## Pending Decisions / Future Optimizations

### A. Cache Strategy — Memory Management

**Current problem:** Component matrices (iid, queen, regression) can be large. During INLA, both prior and conditional precision matrices must coexist in memory.

**Design:** Components are cached to a lower memory tier once they are no longer needed:
- GPU mode: move components to host memory
- CPU mode: cache components to disk

**Implementation:** The caching should live at the component level, not wrapped around assembly methods. The current `restore_from_cache() → assemble → store_in_cache()` pattern is incorrect — it restores state that is immediately discarded.

**Later:** Implement this after the DenseMatrix pipeline works end-to-end.

Here the idea is to use a `CacheableObject(ABC)` base class that Objects that implements the required methods inherits from. We then use a `CacheManager` at the DALIA level to handle dynamic caching of any `CacheableObject` in the framework.

```Python
class CacheableObject(ABC):
    @abstractmethod
    def store_in_cache(): ...

    @abstractmethod
    def restore_from_cache(): ...
```

```python
class CacheManager:
    """
    Manages caching of objects to lower memory tiers.
    
    Objects must implement:
    - store_in_cache() → save current state to lower tier
    - restore_from_cache() → load state from lower tier
    
    This is done by inheriting from `CacheableObject`.

    Design:
    - The cache manager owns the lifecycle, not the model
    - objects are evicted when no longer needed
    - objects are restored on-demand when needed again
    - The model never knows about caching
    """
    
    def __init__(self, tier_config: CacheTierConfig):
        """
        Initialize with tier configuration.
        
        Tiers:
        - tier_0: fast memory (RAM or GPU memory)
        - tier_1: slower storage (disk or host memory for GPU)
        """
        ...
    
    def register(self, object: CacheableObject, name: str) -> str:
        """
        Register a object with the cache manager.
        
        Returns a unique key for this object.
        The object is initially in tier_0 (active).
        """
        ...
    
    def evict(self, key: str) -> None:
        """
        Move a object from tier_0 to tier_1.
        
        Calls object.store_in_cache() internally.
        The object is no longer in active memory.
        """
        ...
    
    def restore(self, key: str) -> None:
        """
        Restore a object from tier_1 to tier_0.
        
        Calls object.restore_from_cache() internally.
        If the object is already in tier_0, this is a no-op.
        """
        ...
    
    def is_active(self, key: str) -> bool:
        """Check if a object is currently in tier_0."""
        ...
    
    def evict_all(self) -> None:
        """Evict all registered objects. Used after assembly is complete."""
        ...
    
    def status(self) -> dict:
        """Return status of all registered objects."""
        ...
```

### B. Matrix Type Selection

**Current state:** All components use `DenseMatrix` as a temporary shortcut.

**Future:** Replace with appropriate structured types:
- `DiagonalMatrix` for iid, regression components
- `SparseMatrix` or `DenseMatrix` for queen component (depends on sparsity pattern)
- `BlockMatrix` for the assembled prior precision matrix

**Impact on assembly:** `_load_prior_components` should return the appropriate matrix type per component, not all `DenseMatrix`.

### C. Incremental Matrix Updates

**Observation:** INLA computation is dominated by linear solver calls (O(n³)), while matrix assembly is BLAS (O(n) for diagonal scaling). Assembly is not the bottleneck — but it's still wasted work.

**Design:** Implement multiplicative update for block-diagonal scaling:
```
If τ → τ', only update: block *= (τ' / τ)
```
This is O(n²) for diagonal blocks vs O(n³) for full reassembly.

**Later:** Add `update_prior_precision_matrix(hp_name, new_value)` method to the model. The optimizer can call this between iterations instead of full reassembly.

### D. Design Matrix Caching

**Current state:** `assemble_design_matrix()` is called every time it's needed.

**Future:** Cache the assembled design matrix after first construction. The design matrix is not HP-dependent, so it only needs to be assembled once.

**Later:** Add internal caching to `assemble_design_matrix()`.

### E. Stencil Configuration for Finite Differences

**Current state:** Not yet implemented.

**Future:** Support configurable stencil widths:
- 3-point central difference: O(h²) accuracy
- 5-point central difference: O(h⁴) accuracy
- Trade-off: accuracy vs number of objective evaluations

**Later:** Add `stencil_width` parameter to the finite difference gradient strategy.

---

