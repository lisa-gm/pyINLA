# Matrix
- Should the slicing in getitem return a Matrix object or a numpy array / sparse matrix?
- How should I handle vectors? (i.e. 1-D arrays)

## Operators and dispatching system
- [X] matmul
- [X] rmatmul
- [x] add
- [ ] sub
- [ ] mul
- [ ] div

## Matrix properties
- [X] shape
- [X] ndim
- [ ] nnz (?)
- [ ] dtype

- [X] Transpose


## Matrix types
- [ ] Structured matrices
    - [ ] Diagonal matrices
    - [ ] BT / BTA


# Sparse Matrices
- [X] choose a specific sparse format for ._data, then ensure __init__ converts to that format.


# Implementation pipeline:
1. ~~Get sparse matrix to have a default underlying ones : csr probably.~~
2. ~~Fixe the implementation for sub when scipy sparse matrices are involved.~~
3. Look at the linear solver part
4. look at the GPU implementation part


Then get a rought understanding of how the following would work:
- io and parallel io
- distributed matrices, dense, sparse
- distributed operations, on dense matrices, BLAS and Linear solvers?
- multiprocessing, can I use mpi4py and nccl4py seemlessly, does it solve past problems? Can I re-use the scafold I did for the distributed backend?

In terms of CPU perofrmances:
- Multithreading handling? All process based or we can do thread based parallelism?
- Even nested thread parallelism?



```
# 1. Class attributes (if any)
# 2. Initialization
# 3. Special representation methods
# 4. Properties (grouped together)
# 5. Comparison operators (if needed)
# 6. Arithmetic operators (standard order)
# 7. Right-hand operators (same order as above)
# 8. In-place operators (if supported)
# 9. Other special methods
# 10. Public methods
# 11. Private/protected methods (start with _)
```