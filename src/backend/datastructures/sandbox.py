




def _dispatch_matmul(left, right):
    # What checks do you perform?
    # How do you determine which backend to call?
    # Sketch the logic:
    
    if ???:  # left is sparse, right is dense
        return _spmm_sparse_dense(left, right)
    elif ???:  # left is dense, right is sparse
        return _spmm_dense_sparse(left, right)
    # ... etc



class Matrix:
    def __init__(self, data):
        self._data = data

    # Expose methods you need:
    def diagonal(self):
        return self._data.diagonal()
    
    def tocsr(self):
        return Matrix(self._data.tocsr())
    
    # Or, use __getattr__ for automatic delegation:
    def __getattr__(self, name):
        return getattr(self._data, name)

    @staticmethod
    def _infer_type(data):
        """Determine what type of matrix this should be"""
        if sp.issparse(data):
            return "sparse"
        elif isinstance(data, np.ndarray):
            return "dense"
        elif hasattr(data, '__matmul__'):  # Duck typing for custom types
            return "structured"
        else:
            raise TypeError(f"Unknown matrix type: {type(data)}")
    
    @classmethod
    def wrap(cls, data):
        """Factory: wrap data in appropriate Matrix subclass"""
        # Based on type, return SparseMatrix, DenseMatrix, etc.
        ...

    def __new__(cls, data):
        # Automatically create the right subclass
        if sp.issparse(data):
            return super().__new__(SparseMatrix)
        elif isinstance(data, np.ndarray):
            return super().__new__(DenseMatrix)
        elif isinstance(data, BlockTridiagonalMatrix):
            return super().__new__(StructuredMatrix)
        # etc.