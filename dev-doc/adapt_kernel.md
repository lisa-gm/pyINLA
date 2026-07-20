# BLAS Kernel Adaptation Prompt

## Context

You are working on a BLAS wrapper library that provides a unified interface for Matrix datastructures. Each BLAS kernel follows a strict internal convention for:
- Argument order (wrapper signature → internal host kernel signature)
- Flow order (validation → extraction → sanitization → allocation → dispatch → return)
- Naming patterns (private helper `_xxx_host`, parameter sanitization, etc.)
- Error message style and type hierarchy

## Reference Implementation

Below is an already-implemented, reviewed, and tested BLAS kernel. Study it carefully. Your job is to adapt the **Target Kernel** (further below) to follow the **exact same conventions**.

### Reference: Wrapper (`{ref_kernel}.py`)

```python
{paste the full reference wrapper file here}
```

### Reference: Tests (`test_blas_{ref_kernel}.py`)

```python
{paste the full reference test file here}
```

## Target Kernel to Adapt

Below is the current state of the target kernel. It may have inconsistencies with the reference (e.g., wrong argument order, missing shape validation, different allocation strategy, different return convention, etc.). Adapt it to match the reference conventions.

### Target: Wrapper (`{target_kernel}.py`)

```python
{paste the full current target wrapper file here}
```

### Target: Tests (`test_blas_{target_kernel}.py`)

```python
{paste the full current target test file here}
```

## Rules

1. **Do NOT touch any accelerator/GPU/device code** (cupy, cublas, nvmath, etc.) — leave it exactly as-is, commented out if it already is.

2. **Argument order in the host kernel** must match the reference pattern exactly:
   - Data arrays first (`a`, `b`, `c`), then scalars (`alpha`, `beta`), then flags (`trans_a`, `trans_b`, `uplo`, etc.).
   - No `overwrite_c` parameter — always hard-code `overwrite_c=True` in the host kernel call.

3. **Flow order in the wrapper** must match the reference:
   - Type assertions (`isinstance` checks)
   - DenseMatrix-only guard
   - Extract `._data` arrays
   - Sanitize `hw_target`
   - Sanitize trans/uplo flags (`.upper()`)
   - If `c` is provided: extract `c_data` + shape assertions → if not: allocate `c_data = np.zeros(..., order="F")`
   - Dispatch to `_xxx_host(...)` (in-place, no return value)
   - If `c` was `None`, wrap result in `DenseMatrix(data=c_data, hw_target="host")` and return it

4. **Host kernel** must be in-place only — it receives a pre-allocated `c` array, modifies it, and returns `None`. Document it as:
   ```
   Returns
   -------
   None
   - The result is stored in the `c` array, which is modified in-place.
   ```

5. **Tests must mirror the reference pattern:**
   - Full type annotations on all parameters
   - Same docstring structure (`Parameters`, `Assert`, `Notes`)
   - `xp = np` / `xp = cp` pattern for reference computation
   - `pytest.skip` guard for complex+accelerator without nvmath
   - Same comment patterns (`# . make operands`, `# . only test in-place for now`, etc.)

6. **Error messages** should be specific but follow the same phrasing style as the reference.

7. **Preserve the `trans` mapping dicts** in the host kernel (e.g., `{"N": 0, "T": 1, "C": 2}`) — they are the proper interface to `scipy.linalg.blas`.

## Output

Produce the adapted wrapper and test files. Use `replace_string_in_file` / `insert_edit_into_file` to apply changes directly.
```

-