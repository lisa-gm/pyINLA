# DALIA Backend: Matrix System Development Summary

## Project Overview

**DALIA** is a statistical learning framework for Bayesian inference using Integrated Nested Laplace Approximation (INLA). The project implements advanced statistical modeling with support for:
- Spatial and temporal models (SPDE)
- Random effects (autoregressive, random walk)
- Various likelihood families (Gaussian, Poisson, Binomial)
- Mixed-effects models with complex covariance structures

## Repository Architecture

```
DALIA/
├── src/
│   ├── backend/                    # Core computational infrastructure
│   │   ├── datastructures/
│   │   │   └── matrix/            # ← Current development focus
│   │   │       ├── core/          # Matrix base classes
│   │   │       └── dispatch/      # Operation routing system
│   │   ├── linalg/                # Linear algebra solvers
│   │   └── multiprocessing/       # Parallel computation
│   │
│   ├── statistical_modeling_toolbox/
│   │   ├── latent_models/         # Statistical model components
│   │   ├── likelihoods/           # Observation models
│   │   └── priors/                # Prior distributions
│   │
│   └── dalia/                      # High-level inference engine
│       ├── mode_finding/          # Posterior mode optimization
│       ├── integration/           # Laplace approximation
│       └── model_fitting/         # Model parameter estimation
│
└── tests/
    └── backend/components/datastructures/matrix/
```

## Current Development: Matrix System

### Motivation

Statistical models in DALIA produce various matrix types:
- **Dense**: Small covariance matrices, design matrices
- **Sparse**: Large precision matrices from spatial models (SPDE)
- **Structured**: Block-tridiagonal with arrowhead (temporal AR models)

**Core requirement**: Automatic dispatch to optimal BLAS kernels based on operand types while maintaining type consistency across operations.

### Design Goals

1. **Polymorphic operations**: `Matrix @ Matrix` should automatically choose:
   - Dense × Dense → Dense BLAS (GEMM)
   - Sparse × Sparse → Sparse CSR multiplication
   - Dense × Sparse → Mixed format multiplication
   - Structured × Vector → Exploit block structure

2. **Type consistency**: All operations must return `Matrix` subclasses, not raw numpy/scipy arrays

3. **External compatibility**: Support mixed operations with `scipy.sparse` and `numpy.ndarray`

4. **Transparent API**: Users shouldn't need to think about underlying representation

## Architecture Implemented

### Class Hierarchy

```python
Matrix (ABC)                    # Base class with operator overloading
├── DenseMatrix                 # Wraps numpy.ndarray
├── SparseMatrix                # Wraps scipy.sparse matrices (csr/csc/coo)
└── StructuredMatrix (future)   # Block-tridiagonal, arrowhead, etc.
```

**Key design choice**: Wrapper pattern, not inheritance from numpy/scipy.
- Each Matrix wraps `._data` attribute (numpy array or scipy sparse matrix)
- Operators unwrap, dispatch, then wrap result

### Dispatch System

**Location**: `src/backend/datastructures/matrix/dispatch/`

```python
# User code
result = matrix_a @ matrix_b

# Internal flow
Matrix.__matmul__
  → unwrap operands to raw numpy/scipy
  → blas_dispatch(Operation.MATMUL, left_data, right_data)
      → dispatch_matmul(left_data, right_data)
          → type inspection: isinstance(left, np.ndarray), sp.issparse(left)
          → route to: sparse × sparse, dense × sparse, etc.
          → return raw result
  → wrap_result(result_data) → SparseMatrix or DenseMatrix
  → return wrapped Matrix
```

**Operations implemented**:
- `__matmul__` / `__rmatmul__`: Matrix multiplication (`@`)
- `__add__` / `__radd__`: Element-wise addition (`+`)
- `.T`: Transpose property (view for dense, copy for sparse)

**Test coverage**: 57/57 passing (matmul, add, transpose)

## Critical Technical Challenge: Array Protocol Blocking

### Problem Encountered

When mixing Matrix with external types:
```python
scipy_matrix = scipy.sparse.csr_matrix(...)
dense_matrix = DenseMatrix(...)

# This should return DenseMatrix but was returning numpy.ndarray
result = scipy_matrix @ dense_matrix
```

**Root cause**: Python operator precedence with external types.

### Discovery Process

1. **Initial observation**: `scipy.sparse + Matrix` worked correctly, but `scipy.sparse @ Matrix` returned unwrapped `numpy.ndarray`

2. **Hypothesis**: Array protocol allowing implicit conversion
   - Tried blocking `__array__()` with explicit method
   - scipy caught TypeError and propagated it (didn't trigger `__rmatmul__`)

3. **Key insight from user**: Removing `__getattr__` made everything work
   - `__getattr__` was delegating to `_data.__array__()`
   - scipy successfully converted Matrix to array internally
   - Never called `Matrix.__rmatmul__()`

4. **Investigation of scipy's conversion chain**:
   ```python
   # Logging attribute accesses revealed:
   __array_struct__      # Buffer protocol for memory access
   __array_interface__   # Dict-based array protocol
   __array__            # Standard conversion method
   ```

### Solution Implemented

Block all three array protocol attributes in `__getattr__`:

```python
def __getattr__(self, name):
    """Delegate to underlying data, blocking array protocol."""
    if name in ('__array__', '__array_struct__', '__array_interface__'):
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Use .toarray() for explicit conversion."
        )
    return getattr(self._data, name)
```

**Effect**:
- When scipy tries `np.asanyarray(Matrix)`, all conversion paths fail
- scipy's `__matmul__` returns `NotImplemented`
- Python calls `Matrix.__rmatmul__(scipy_matrix, self)`
- Result is properly wrapped as Matrix

**Additional class attribute**:
```python
__array_ufunc__ = None  # Disable numpy universal functions
```

### Design Philosophy: Explicit vs Implicit Conversion

**Choice made**: Follow pandas/PyTorch pattern, not numpy pattern

| Library | Conversion | Example |
|---------|------------|---------|
| numpy | Implicit | `np.asarray(x)` just works |
| pandas | Explicit | `df.to_numpy()` required |
| PyTorch | Explicit | `tensor.numpy()` required |
| **DALIA Matrix** | **Explicit** | **`matrix.toarray()`** |

**Rationale**:
- **Type consistency**: All operations return Matrix types
- **Predictability**: No surprise unwrapping to raw arrays
- **Performance**: Explicit conversion forces users to think about when leaving Matrix ecosystem

**Tradeoff**:
- ✅ Consistent types throughout statistical model computations
- ✅ Clear dispatch to optimal kernels
- ⚠️ Some numpy/matplotlib functions require `.toarray()` or `._data` access

### Code Location

**Core implementation**:
- `src/backend/datastructures/matrix/core/matrix.py` (base class)
- `src/backend/datastructures/matrix/core/dense.py`
- `src/backend/datastructures/matrix/core/sparse.py`
- `src/backend/datastructures/matrix/core/utils.py` (wrap_result, toarray helpers)

**Dispatch system**:
- `src/backend/datastructures/matrix/dispatch/dispatcher.py`
- `src/backend/datastructures/matrix/dispatch/operations.py` (Operation enum)
- `src/backend/datastructures/matrix/dispatch/matmul.py`
- `src/backend/datastructures/matrix/dispatch/add.py`

**Tests**:
- `tests/backend/components/datastructures/matrix/test_matrix_matmul.py` (24 tests)
- `tests/backend/components/datastructures/matrix/test_matrix_add.py` (24 tests)
- `tests/backend/components/datastructures/matrix/test_matrix_transpose.py` (9 tests)

## Implementation Status

### ✅ Complete

1. **Matrix base class**:
   - Abstract base with operator overloading
   - Array protocol blocking via `__getattr__`
   - `.toarray()` for explicit conversion
   - `.T` transpose property

2. **Subclasses**:
   - `DenseMatrix` wrapping `numpy.ndarray`
   - `SparseMatrix` wrapping `scipy.sparse` (csr/csc/coo)

3. **Operations**:
   - Matrix multiplication (`@`) with all combinations
   - Addition (`+`) with all combinations
   - Right-hand operators (`__rmatmul__`, `__radd__`)
   - Transpose (view for dense, copy for sparse)

4. **External type support**:
   - Mixed operations with `scipy.sparse.csr_matrix/csc_matrix/coo_matrix`
   - Mixed operations with `numpy.ndarray`
   - Proper type wrapping in all cases

5. **Test coverage**: 57/57 passing
   - All dense/sparse combinations
   - External scipy/numpy types
   - Transpose semantics

### 🚧 Pending

1. **Additional operators**:
   - Subtraction (`__sub__`, `__rsub__`)
   - Element-wise multiplication (`__mul__`, `__rmul__`)
   - Division (`__truediv__`, `__rtruediv__`)
   - In-place operators (`__iadd__`, `__imatmul__`, etc.)

2. **Structured matrices**:
   - `BlockTridiagonalMatrix` (for Spatio-temporal models)
   - `ArrowheadMatrix` (for Spatio-temporal models)
   - Custom dispatch exploiting block structure

3. **Sparse format standardization**:
   - `SparseMatrix.__init__` should convert to canonical format (CSR?)
   - Or maintain format flexibility with explicit conversion methods

4. **Vector handling**:
   - 1-D array support (currently assumes 2-D matrices)
   - Matrix-vector multiplication dispatch

5. **Linear solvers**:
   - Cholesky decomposition
   - Solve systems: `A \ b`
   - Inverse/Selected inversion
   - Eigendecomposition (Hessian?)

6. **Performance optimization**:
   - Benchmark dispatch overhead
   - Consider caching type checks
   - Profile memory copies

## Key Learnings for Future Development

### 1. Scipy's array conversion is aggressive

Scipy tries multiple conversion protocols before returning `NotImplemented`:
1. Direct `__matmul__` on operand
2. `np.asanyarray()` → `__array__()`
3. `__array_struct__` (buffer protocol)
4. `__array_interface__` (dict protocol)

**Must block all three** to force fallback to right-hand operators.

### 2. `__getattr__` interacts with protocols

Special methods accessed via `__getattr__` are found by `getattr()` but not by Python's internal method resolution for operators. However, numpy/scipy explicitly call `getattr()` to check protocols.

### 3. View vs copy semantics

Be explicit about when operations return views:
- `.T` on dense → view (like numpy)
- `.T` on sparse → copy (scipy limitation)
- `.toarray()` on dense → view (via `np.asarray`)
- `.toarray()` on sparse → copy (necessary for conversion)

### 4. Test external types extensively

Don't just test `Matrix @ Matrix`. Test:
- `scipy.sparse @ Matrix`
- `Matrix @ numpy.ndarray`
- `numpy.ndarray @ Matrix`
- All scipy sparse formats (csr, csc, coo)

### 5. Lazy imports for circular dependency

In `utils.py`:
```python
def wrap_result(data):
    from .dense import DenseMatrix  # Import inside function
    from .sparse import SparseMatrix
    # ...
```

Avoids `matrix.py` → `utils.py` → `dense.py` → `matrix.py` cycles.

## Integration with DALIA Statistical Models

### Example Usage Pattern

```python
# Spatial model generates sparse precision matrix
Q = spatial_model.precision_matrix()  # Returns SparseMatrix

# Design matrix is dense
X = DenseMatrix(design_matrix)

# Mixed operation automatically dispatches correctly
XtQX = X.T @ Q @ X  # Sparse × Dense → Dense result

# Solve system (future)
L = cholesky(Q)          # Should return SparseMatrix
x = solve(L, b)          # Should return DenseMatrix -> To be decided given the dimmensions of the right-hand side and the current state of the decision regarding Vector handling
```

### Why This Matters for INLA

1. **Computational efficiency**: SPDE models produce ~10⁶ × 10⁶ precision matrices with <0.01% non-zeros. Sparse operations are essential.

2. **Type safety**: Laplace approximation involves many matrix operations. Losing sparsity accidentally would cause memory explosion.

3. **Code clarity**: Statistical code shouldn't worry about when to use `scipy.sparse.csr_matrix.dot()` vs `numpy.dot()`. Dispatch handles it.

4. **Future extensibility**: Structured matrices (block-tridiagonal from ST models) need specialized O(n) solvers (n number of time steps), not O(n³) dense solvers.

## Questions for Future Consideration

1. **Slice returns**: Should `matrix[0:2, 0:2]` return Matrix or unwrapped array?

2. **In-place operations**: Should `matrix += other` mutate `._data` or create new Matrix?

3. **Format conversions**: Explicit `.to_csr()` / `.to_dense()` methods vs automatic conversion?

4. **Error handling**: What happens if dispatch encounters incompatible shapes? Raise custom exceptions or let numpy/scipy errors propagate?

5. **Memory management**: When should operations create copies vs views? Current behavior is inconsistent (transpose is view for dense, copy for sparse).


## References

- NEP 18 (Array Function Protocol): https://numpy.org/neps/nep-0018-array-function-protocol.html
- NEP 13 (Array UFuncs): https://numpy.org/neps/nep-0013-ufunc-overrides.html
- Scipy sparse matrix documentation: https://docs.scipy.org/doc/scipy/reference/sparse.html
- Pandas design rationale for explicit conversion: https://pandas.pydata.org/docs/user_guide/basics.html

---

**Summary**: The Matrix system provides a type-safe, dispatch-based linear algebra foundation for DALIA's statistical models. Key achievement is maintaining type consistency across operations with external scipy/numpy types by carefully blocking array protocol conversion paths. The system is ready for integration into statistical model implementations while additional operators and structured matrix types remain to be developed.
