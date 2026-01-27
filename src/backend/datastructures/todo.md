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
- [ ] choose a specific sparse format for ._data, then ensure __init__ converts to that format.