import scipy.sparse as sp
import numpy as np


class MySparseCSR(sp.csr_matrix):
    def __matmul__(self, other):
        print(f"MySparseCSR @ {type(other).__name__}")
        result = super().__matmul__(other)
        print(f"Result type: {type(result)}")
        return result


# Test it
A = MySparseCSR([[1, 2], [3, 4]])

E = np.array([[1, 2], [3, 4]])
F = E @ A  # numpy array @ MySparseCSR

print(type(F))

G = A @ E  # MySparseCSR @ numpy array

print(type(G))
