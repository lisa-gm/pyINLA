## Hyperparameter Architecture: Model ↔ Optimizer Interface

> **Status note (2026-07-14):** `Hyperparameter` (`hp_dataclass.py`) and `HyperparameterManager`
> (`hp_manager.py`) are stable and implement the fixed/optimized split described below.
> `model.py`, `main.py` config plumbing, and `StatisticalModelConfig.__post_init__` are also
> now consistent (see "Blocking Fixes", all resolved). The frontier has moved to `inla.py`,
> which still `exit()`s right after assembling `Q_prior`/`A` — the marginal-likelihood terms
> and everything depending on `dalia.backend` (Cholesky, solves, log-det) is unwritten.
> This document stays a **forward-looking spec**: it does not re-explain what's already
> legible in the code, only how the not-yet-written pieces should be puzzled together.

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

## Blocking Fixes — Status

The items originally listed here (config field mismatches, `__post_init__` referencing
the wrong attribute, `GenomicModelConfig` not being a real dataclass, constructor
argument mismatches between `main.py` and `HyperparameterManager`) are **resolved** in
the current `model.py`/`main.py`/`hp_manager.py`. Nothing left to fix at that layer.

Two new blocking items surfaced while reviewing the backend integration path — these
will raise or silently misbehave the moment `inla.py` calls past the `exit()` stub:

1. **`DenseSolver._compute_selected_inverse` is broken in both branches.**
   `overwrite_factors=True` calls `self._set_library(self._target)`, which does not
   exist anywhere in `DenseSolver`/`LinearSolver` → `AttributeError`. The
   `overwrite_factors=False` branch references `xp_la` without ever assigning it in
   that branch → `NameError`. Since marginal variances (needed for the latent field's
   posterior marginals, not just the mode) are exactly a selected/full inverse call,
   this must be fixed before the marginals can be computed at all. Fix: define
   `_set_library` (presumably returning `scipy.linalg` vs `cupyx.scipy.linalg` based on
   `self._target`) and call it unconditionally at the top of the method, not only in
   the overwrite branch.

2. **`objective()`'s buffer/accept coupling is implicit.** `inla.objective` calls
   `hpm.buffer_update(hp_values)` on every call (including line-search trials and
   finite-difference perturbations), and `main.minimize_callback` calls
   `hpm.commit_buffer(f=...)` with no array argument — it commits whatever is
   currently sitting in `_buffer`. This is correct only if the last `x` scipy passed
   to `fun`/`jac` before firing the callback is exactly `intermediate_result.x`, which
   holds for `L-BFGS-B` today but isn't a contract (would break under e.g.
   `trust-constr`, which evaluates `fun` and `jac` on different schedules). Fix:
   `minimize_callback` should call `hpm.buffer_update(intermediate_result.x)`
   immediately before `commit_buffer()`, making acceptance explicit rather than
   order-dependent.

---

## Next Steps: Wiring the Rest of the Pipeline

The manager and model classes are the *foundation*; the interesting remaining
work is in how the objective, gradient strategy, and optimizer loop consume
them. This section sketches that wiring — treat it as a proposal, not a
finished spec, since `inla.py`/`main.py` are still drafts.

### 1. ~~`objective()` must be a pure function of `(hp_values, model)`~~ — resolved

`inla.objective` now converts the raw array to a dict via
`hpm.convert_array_to_dict(array=hp_values, include_fixed=True)` before calling
`marginal_log_likelihood_approximation(hp_dict, model)`, and
`marginal_log_likelihood_approximation` itself only takes `(hp_dict, model)` — no
hidden manager state read inside the pure computation. This matches the
original proposal. Remaining wrinkle: `objective()` still calls `hpm.buffer_update`
as a side effect on every evaluation (including line-search/finite-diff
perturbations) — see Blocking Fix #2 above for why that specific coupling still
needs tightening.

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

### 4. ~~The optimizer wrapper (in `main.py`)~~ — implemented, one gap remains

`fit_model()` in `main.py` now does essentially this: it builds `hpm`, passes
`hpm.get_latest_hyperparameters(format="array", include_fixed=False)` as `x0` and
`hpm.get_bounds()` to `minimize`, and registers `minimize_callback` which calls
`hpm.commit_buffer(f=intermediate_result.fun)`. After `minimize()` returns it calls
`hpm.update_model(model)`. The one open gap is the same one noted in Blocking Fix
#2: `minimize_callback` should call `hpm.buffer_update(intermediate_result.x)`
before `commit_buffer()` rather than relying on the last `objective()`/`jac` call
having already populated the buffer with the accepted point.

Still genuinely open: `fit_model` never calls `HyperparameterManager.load_checkpoint`
— resuming from a checkpoint isn't wired into `main.py`'s control flow yet (see #7
below), so `checkpoint_hpm=True` currently only ever writes, never reads.

### 5. ~~Config plumbing between `main.py`, `GenomicModelConfig`, and `StatisticalModelConfig`~~ — resolved

`GenomicModelConfig` is now a real `@dataclass` (inherits `StatisticalModelConfig`'s
fields correctly), `StatisticalModelConfig.__post_init__` references
`self.hyperparameters`, and `main.py`'s `GenomicModelConfig(...)` call now uses the
actual field names (`path_to_model_components`, `path_to_observations`,
`hyperparameters`, `iid_prior_n`, ...). No action needed here anymore.

### 6. `finite_difference_gradient`'s `stencil=5` branch is a silent no-op

The `elif stencil == 5:` body in `inla.finite_difference_gradient` is just `...` —
`grad[i]` is left at `0.0` for every hyperparameter. Passing `stencil=5` today does not
raise; it silently hands `scipy.optimize.minimize` an all-zero gradient, which reads as
immediate (false) convergence at `x0`. Either implement the 5-point stencil or raise
`NotImplementedError` in that branch until it exists — leaving it as `...` is strictly
worse than raising, because it fails silently rather than loudly.

### 7. Checkpointing/resume path is unexercised

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

---

## Wiring `dalia.backend` into `inla.py`: what has to be decided next

This is the actual next step — `marginal_log_likelihood_approximation` currently builds
`Q_prior` and `A` and then `exit()`s. Getting past that stub requires resolving how the
four INLA terms consume `Matrix`/`LinearSolver`, not just filling in the `...` stubs.

### 1. Where does `Q_cond = Q_prior - θ·AᵀA` get built, and with what primitive?

This was flagged as an open question in `model.py`'s own docstring ("the conditional
precision matrix ... depends on the type of prior ... hence should not be part of it")
and is still open. Two sub-decisions, not one:

- **Ownership**: a method on `StatisticalModel`
  (`assemble_conditional_precision_matrix`), a method on the likelihood object (see
  `shared_dev/saved_taxonomy/likelihoods/`), or inline in `inla.py`. Given the
  Gaussian-only path right now, inlining in `inla.py` is the pragmatic short-term
  choice — but decide now whether that's a placeholder or the permanent home, since it
  changes where a future non-Gaussian likelihood's Newton step would plug in.
- **Primitive used for `AᵀA`**: `A.T @ A` through the `Matrix` API resolves to `gemm`
  via `dispatch_matmul` — general matrix-matrix product, computing the full result and
  discarding the fact that `AᵀA` is symmetric. `dalia.backend.BLAS.syherk` (rank-k
  symmetric update) already exists but has no `Matrix`-level entry point (no operator,
  no `Operation.SYRK` in the dispatch enum). Before writing the `Q_cond` assembly,
  decide whether to (a) add a `Matrix.rank_k_update()` / dedicated dispatch path wired
  to `syherk`, or (b) accept the ~2x FLOP cost of general `gemm` for now and revisit
  once profiling justifies it. Option (a) is the more consistent long-term choice given
  the framework already forked `syherk.py` for this purpose.

### 2. Selected inverse is required, not optional, for marginals

`compute_conditional_latent`/marginal variances need `LinearSolver.selected_inverse()`,
which is currently broken (see Blocking Fixes #1 above). Beyond the bug fix, decide:
- Do you need the **full** inverse (current `_compute_selected_inverse` behavior for
  `DenseMatrix`) or an actual **selected** inverse restricted to `Q_prior`'s sparsity
  pattern? For the dense pipeline this is moot (full ≈ selected), but the naming and
  the eventual `SparseSolver._compute_selected_inverse` (currently
  `NotImplementedError`) suggest the API was designed with sparse selected-inversion in
  mind. Worth deciding whether `inla.py`'s dense path should call the same
  `selected_inverse()` name now so the sparse swap-in later is a no-op at the call site.

### 3. `A` is hyperparameter-independent but reassembled every objective call

`assemble_design_matrix()` takes no hyperparameters and is deterministic, yet
`marginal_log_likelihood_approximation` (called `2N+1` times per outer iteration via
finite differences) reassembles it from disk every time. This was already flagged as
"Pending Decision D" for later — it stops being deferrable once the solver is wired in,
since `A` is about to be a `syherk`/`gemm` operand `2N+1` times per outer iteration.
Cache it once (e.g. inside `GenomicModel.__init__` or lazily on first call) before
wiring the solver, not after — retrofitting is more disruptive once `inla.py` depends
on a specific calling convention.

### 4. Layout consistency through `Matrix` arithmetic vs. what `DenseSolver` expects

`DenseMatrix.__init__` forces Fortran order only at construction
(`force_order=True` by default); `wrap_result()` — used after every `+`/`-`/`*`/`@` —
constructs results with `force_order=False`, so `Q_cond` built via `Q_prior - theta *
(A.T @ A)` has no guaranteed memory layout. `LinearSolver.solve()` additionally forces
any `Matrix` right-hand side through `.toarray()`, which hardcodes `order="C"`. Neither
is wrong, but every mismatch between the F-order Cholesky factors and a C-order `b`
costs a silent BLAS-level copy. Once `Q_cond` is on the hot path (`2N+1` solves per
outer iteration), decide whether `_assemble_prior_precision_matrix`/`Q_cond` assembly
should explicitly request Fortran order for solver-bound matrices, rather than relying
on whatever `wrap_result` happens to produce.

### 5. Symmetry/PD sanity checks belong at the assembly boundary, not after `cholesky` fails

Manual block-offset slicing in `_assemble_prior_precision_matrix` is easy to get subtly
wrong once a 4th component is added (off-by-one offset, mismatched block, etc.). A
cheap `assert np.allclose(Q.toarray(), Q.toarray().T)` (or a symmetry-aware assembly
that only ever writes the upper/lower triangle) right after assembly, before it reaches
`DenseSolver.factorize()`, turns a cryptic `LinAlgError` deep in LAPACK into an
immediate, localized assertion failure.