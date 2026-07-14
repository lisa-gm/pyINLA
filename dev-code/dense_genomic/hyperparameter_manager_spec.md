## Hyperparameter Architecture: Model ↔ Optimizer Interface

> **Status note (2026-07-13):** `Hyperparameter` and `HyperparameterManager` now exist in
> `hyperparameter.py` and already implement the fixed/optimized split described below.
> This document is kept as a **forward-looking spec**: it assumes the existing classes,
> flags what still needs fixing in them, and focuses on how they get puzzled together
> with the model, the objective, the gradient strategy, and the optimizer — the parts
> of the pipeline that don't exist yet. It intentionally does not re-explain what is
> already legible in the code.

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

## Blocking Fixes (do these before writing new code on top)

These aren't design questions — they will raise exceptions or silently corrupt
state the first time the pipeline actually runs. Fix in this order, since later
items depend on earlier ones being correct.

3. **Fix the optimized/full index mismatch in `array_to_dict`/`dict_to_array`.**
   `_array` only has `len(self._optimized_keys)` entries, but both methods loop
   over `self._order` (fixed + optimized) and index with `self._key_to_index`
   (built over the full order). Both should loop/index over `_optimized_keys` /
   `_optimized_key_to_index` consistently.

4. **Give the model a way to see the *full* hyperparameter dict (fixed + optimized merged).**
   `GenomicModel._assemble_prior_precision_matrix` needs every key
   (`tau_iid`, `tau_queen`, `prec_regression`), not just the ones being
   optimized. `_fixed_values` is stored but never merged back in anywhere.
   Suggest adding `HyperparameterManager.get_full_dict(array=None) -> dict[str, float]`
   that merges `_fixed_values` with `array_to_dict(array)` — this becomes the
   thing that actually gets passed to `model.assemble_prior_precision_matrix`.

5. **Decide whether the Manager owns copies or references of the Model's `Hyperparameter` objects.**
   `model.get_hyperparameters()` returns a shallow dict copy — same
   `Hyperparameter` instances. `commit_buffer()` mutates
   `self._hyperparameters[key].value` in place, which means it mutates the
   model's own objects immediately, before `update_model()` is ever called.
   This contradicts the stated goal ("model does not track optimization
   state"). Either deep-copy `Hyperparameter` objects at
   `HyperparameterManager.__init__` (so `update_model()` becomes the real,
   deliberate sync point back to the model), or explicitly drop that isolation
   guarantee from the docstring and design around shared references instead.

---

## Next Steps: Wiring the Rest of the Pipeline

The manager and model classes are the *foundation*; the interesting remaining
work is in how the objective, gradient strategy, and optimizer loop consume
them. This section sketches that wiring — treat it as a proposal, not a
finished spec, since `inla.py`/`main.py` are still drafts.

### 1. `objective()` must be a pure function of `(hp_values, model)`

`inla.objective` currently calls `manager.get_named_values()` — an undefined
name, and (more importantly) the wrong idea: an objective used for finite
differences must recompute `f` from the exact perturbed point it's given, not
from whatever the manager currently has committed. The fix is mechanical but
important to get right:

```python
def objective(hp_full_dict: dict[str, float], model: StatisticalModel) -> float:
    Q_prior = model.assemble_prior_precision_matrix(hp_full_dict)
    ...
```

The caller (the optimizer wrapper, not `objective` itself) is responsible for
turning the raw `x: np.ndarray` scipy hands it into `hp_full_dict` via
`manager.get_full_dict(array=x)` (see fix #4 above). This keeps `objective`
trivially testable in isolation, independent of `HyperparameterManager`.

### 2. Where does `Q_cond = Q_prior - θ·AᵀA` (or similar) belong?

`inla.py` currently stubs this out with `...`. This is a modeling decision, not
a hyperparameter-management one, but it affects the interface: does
`StatisticalModel` gain a `assemble_conditional_precision_matrix(hp_full_dict)`
method (keeping the "given the likelihood, know how to combine Q_prior and A"
logic inside the model, consistent with your existing `assemble_*` pattern), or
does `objective()` compute it inline using only `Q_prior`/`A` as building
blocks? Given your note in `model.py` that "the conditional precision matrix
... depends on the type of prior ... hence should not be part of it" — worth
deciding now whether that logic instead belongs on the *likelihood* object
(see `saved_taxonomy/likelihoods/`) rather than duplicated inside `inla.py`.

### 3. `GradientStrategy` doesn't exist yet as a class — only as a bare function

`design_decisions.md` describes an abstract `GradientStrategy` with
`finite_difference`/`backward_difference`/`autodiff` variants, and
`mode_finding_and_ad.md` explains why AD breaks for non-Gaussian likelihoods
(data-dependent Newton iteration count, non-differentiable convergence check).
Concretely, before the pipeline can run end-to-end you need:

- An actual `GradientStrategy` ABC (or a `Protocol`) with a single
  `compute(objective_fn, x, model) -> np.ndarray` method, so `optimize()` can
  stay agnostic to which strategy is active.
- A decision on whether `finite_difference_gradient` in `inla.py` becomes a
  method on a `FiniteDifferenceStrategy` class, or stays a free function that
  a thin `GradientStrategy` wraps. Given `L-BFGS-B` already supports numeric
  differentiation out of the box (`jac=None` + `bounds`), also worth asking:
  do you need your *own* finite-difference implementation at all for the
  Gaussian-likelihood path, or only once AD/implicit-differentiation is
  introduced for non-Gaussian likelihoods?

### 4. The optimizer wrapper (in `main.py`) is the piece that ties everything together

None of this exists yet beyond a signature stub. Concretely it needs to:

- Pull `x0 = hp_manager.get_array()` and `bounds = hp_manager.get_bounds()`
  (optimized-only, per fix #3).
- Wrap `objective` so scipy's `x` gets converted via
  `hp_manager.get_full_dict(array=x)` before being passed to `objective`.
- Wrap `jac` similarly if/when a `GradientStrategy` other than scipy's built-in
  numerical differencing is used.
- Register a `callback(xk)` that commits `xk` (per fix #6) and, only then, asks
  the `HyperparameterManagerConfig` checkpointing logic whether to flush to
  disk.
- After `minimize()` returns, call something like
  `hp_manager.commit_buffer(result.x)` followed by `hp_manager.update_model(model)`
  as the single deliberate sync point from optimizer state back into the model
  — consistent with fix #5.
- `main.py`'s current call to `HyperparameterManager(hyperparameters=[...],
  config=...)` also needs to be reconciled with the real constructor, which
  takes `model` (not a raw list of `Hyperparameter`) and derives
  `model.get_hyperparameters()` from it — decide whether `main.py` should
  build `Hyperparameter`s and attach them to `GenomicModelConfig.hyperparameters`
  instead of constructing them ad hoc for the manager.

### 5. Config plumbing between `main.py`, `GenomicModelConfig`, and `StatisticalModelConfig` needs to be reconciled

Not a `HyperparameterManager` concern per se, but it blocks running anything:
`GenomicModelConfig` isn't itself a `@dataclass` (so its extra fields like
`iid_prior_n` aren't real dataclass fields), `StatisticalModelConfig.__post_init__`
references `self.config.hyperparameters` where it should reference
`self.hyperparameters`, and `main.py` passes keyword arguments
(`dataset_path`, `n_observations`) that don't exist on either config while
omitting the ones that do (`path_to_model_components`, `path_to_observations`,
`hyperparameters`). Worth resolving before wiring the optimizer, since
`optimize()` needs a working `model` instance to run against.

### 6. Checkpointing/resume path is unexercised

`HyperparameterManagerConfig` has the knobs (`checkpoint_hpm`,
`checkpoint_hpm_every`, ...) and `_checkpoint()`/`load_checkpoint()` exist, but
nothing calls `load_checkpoint()` from `main.py`, and `_checkpoint()` pickles
`self._hyperparameters` (live `Hyperparameter` objects, possibly containing
non-trivial `Bounds` objects) via `np.save(..., allow_pickle=True)` — confirm
this round-trips correctly once `Hyperparameter` is a real dataclass (fix #2),
and decide whether resume should reconstruct a fresh `HyperparameterManager`
from `model.get_hyperparameters()` and then overwrite `_array`/`_iteration`/
`_history` from the checkpoint (current approach), or reconstruct everything
from the checkpoint alone without touching the model at all.

This architecture cleanly separates concerns:
- **Model**: Works with dict `{"sigma_st": 1.0, ...}`
- **Optimizer**: Works with array `[1.0, 2.0, ...]`
- **Manager**: Bridges both, tracks only accepted iterations
- **Buffer**: Holds tentative values until acceptance