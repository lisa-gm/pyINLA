import os

import numpy as np
from pyinla import xp, sp

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
#from pyinla.utils import scaled_logit, bdiag_tiling, get_host

import time

import matplotlib.pyplot as plt

path = os.path.dirname(__file__)

# def compute_temp_izip(A, B):
#     """
#     Perform the computation equivalent to the C++ OpenMP code.

#     Parameters:
#     - A: Sparse matrix (CSR format).
#     - B: Sparse matrix (CSR format).

#     Returns:
#     - temp: Updated sparse matrix.
#     """
#     # Convert A to COO format for easy iteration
#     A_coo = A.tocoo()
#     temp_data = np.zeros(A_coo.nnz, dtype=A_coo.data.dtype)
#     temp_row = A_coo.row.copy()
#     temp_col = A_coo.col.copy()

#     # Iterate over non-zero elements of A using COO format
#     for idx, (i, j, v) in izip(A_coo.row, A_coo.col, A_coo.data):
#         new_value = 0.0  # Reset the value in temp
#         # Compute the new value for temp[i, j]
#         for k, v_A in izip(A_coo.col[A_coo.row == i], A_coo.data[A_coo.row == i]):
#             new_value[i, j] += v_A * B[k, j]
#         temp_data[idx] = new_value
        

#     # Create a new sparse matrix in COO format and convert to CSR
#     temp = sp.sparse.coo_matrix((temp_data, (temp_row, temp_col)), shape=A.shape).tocsr()

def compute_temp(A, B):
    """
    Perform the computation equivalent to the C++ OpenMP code.

    Parameters:
    - temp: Sparse matrix to be updated (CSR format).
    - Ax_all: Sparse matrix (CSR format).
    - Qinv: Sparse matrix (CSR format).
    - xp: Array module (NumPy or CuPy).

    Returns:
    - temp: Updated sparse matrix.
    """
    # Ensure temp is in CSR format
    temp = A.copy().tocsr()

    # loop over rows in A 
    for k in range(temp.shape[0]):
        # loop over non-zero elements in each row of temp
        for col, value in zip(temp.indices[temp.indptr[k]:temp.indptr[k+1]], temp.data[temp.indptr[k]:temp.indptr[k+1]]):
            temp[k, col] = 0.0
            # loop over non-zero elements in each row of A
            for col_A, value_A in zip(A.indices[A.indptr[k]:A.indptr[k+1]], A.data[A.indptr[k]:A.indptr[k+1]]):
                #print("k: ", k, "col: ", col, "col_A: ", col_A, "value_A: ", value_A, "B[col_A, col]: ", B[col_A, col])
                temp[k, col] += value_A * B[col_A, col]

    return temp

def compute_diagABAt(A, B):
    """
    Computes temp where temp[i,j] = sum_k (A[i,k] * B[k,j])
    ONLY for (i,j) where A[i,j] ≠ 0 (preserving A's sparsity pattern).
    
    Args:
        A (csr_matrix): Sparse matrix (CSR format).
        B (csr_matrix): Sparse matrix (CSR format).
    
    Returns:
        csr_matrix: Result with same sparsity as A.
    """
    # Ensure CSR format for fast row/column access
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    
    # Preallocate output data
    data = xp.zeros_like(A_csr.data)
    
    # Precompute B's columns for faster access
    B_cols = [B_csr[:, j].tocsc() for j in range(B_csr.shape[1])]
    
    for i in range(A_csr.shape[0]):
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        cols_A = A_csr.indices[row_start:row_end]
        vals_A = A_csr.data[row_start:row_end]
        
        for idx, j in enumerate(cols_A):
            # Get column j of B (in CSC format for fast access)
            print("j: ", j)
            B_col_j = B_cols[j]
            
            # Find overlapping non-zeros between A[i,:] and B[:,j]
            common_k = xp.intersect1d(
                cols_A,
                B_col_j.indices,
                assume_unique=True
            )
            
            if len(common_k) > 0:
                # Extract A[i, common_k] and B[common_k, j]
                a_vals = vals_A[xp.searchsorted(cols_A, common_k)]
                b_vals = B_col_j.data[xp.searchsorted(B_col_j.indices, common_k)]
                data[row_start + idx] = xp.dot(a_vals, b_vals)
    
    return sp.sparse.csr_matrix((data, A_csr.indices.copy(), A_csr.indptr.copy()), shape=A_csr.shape)

def sparse_diag_product(A, B):
    """
    Computes diag(A @ B) efficiently for sparse matrices A and B.
    
    Args:
        A, B: Sparse matrices in CSR format (for efficient row/column access).
    
    Returns:
        NumPy array containing the diagonal of A @ B.
    """
    assert A.shape[1] == B.shape[0], "Matrix dimensions must match for multiplication"
    
    # Ensure CSR format for fast row/column access
    A_csr = A.tocsr() if not isinstance(A, sp.sparse.csr_matrix) else A
    B_csc = B.tocsc() if not isinstance(B, sp.sparse.csc_matrix) else B  # CSC for fast column access
    
    diag = xp.zeros(A.shape[0])  # Initialize diagonal
    
    # Iterate over rows of A and multiply by corresponding columns of B
    for i in range(A_csr.shape[0]):
        # Get row i of A (non-zero entries)
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        cols_A = A_csr.indices[row_start:row_end]
        vals_A = A_csr.data[row_start:row_end]
        
        # Get column i of B (non-zero entries)
        col_start = B_csc.indptr[i]
        col_end = B_csc.indptr[i + 1]
        rows_B = B_csc.indices[col_start:col_end]
        vals_B = B_csc.data[col_start:col_end]
        
        # Find matching indices where A's column j matches B's row j
        common_j = xp.intersect1d(cols_A, rows_B, assume_unique=True)
        
        if len(common_j) > 0:
            # Sum A[i,j] * B[j,i] for overlapping j
            a_vals = vals_A[xp.searchsorted(cols_A, common_j)]
            b_vals = vals_B[xp.searchsorted(rows_B, common_j)]
            diag[i] = xp.sum(a_vals * b_vals)
    
    return diag


if __name__ == "__main__":


    # Define small example matrices
    n = 30 # Size of the matrices
    m = 40
    # Ax_all_dense = np.random.rand(m, n)  # Random dense matrix
    # Ax_all = sp.sparse.csr_matrix(Ax_all_dense)  # Convert to CSR format
    # Qinv_dense = np.random.rand(n, n)  # Random dense matrix
    # Qinv = sp.sparse.csr_matrix(Qinv_dense)  # Convert to CSR format
    Ax = sp.sparse.random(m, n, density=0.05, format="csr")
    Qinv = sp.sparse.random(n, n, density=0.05, format="csr")

    # print("qinv: \n", Qinv.toarray())
    # print("Ax: \n", Ax.toarray())

    # Call the function
    time1 = time.time()
    updated_temp = compute_temp(Ax, Qinv)
    time2 = time.time()
    # print("Time taken for sparse matrix multiplication: ", time2 - time1)
    
    time1 = time.time()
    updated_temp1 = compute_diagABAt(Ax, Qinv)
    time2 = time.time()
    print("Time taken for optimized sparse matrix multiplication: ", time2 - time1)

    #print("updated_temp: \n", updated_temp.toarray())
    # print("updated_temp1: \n", updated_temp1.toarray())
    print("norm(diff):", xp.linalg.norm(updated_temp.toarray() - updated_temp1.toarray()))

    time1 = time.time()
    AxQinv_dense = Ax.toarray() @ Qinv.toarray()
    time2 = time.time()
    print("Time taken for dense matrix multiplication: ", time2 - time1)

    mask = Ax.toarray().astype(bool)
    restricted_AxQinv = AxQinv_dense * mask
    #print("AxQinv_dense: \n", AxQinv_dense)
    #print("restricted_AxQinv: \n", restricted_AxQinv)

    diff = updated_temp1.toarray() - restricted_AxQinv
    # print("diff: \n", diff)
    print("norm(diff):", xp.linalg.norm(diff))

    time1 = time.time()
    diag = sparse_diag_product(updated_temp1, Ax.T)  
    time2 = time.time()
    print("Time taken for sparse diag product: ", time2 - time1)

    time1 = time.time()
    diag_ref = (updated_temp1 @ Ax.T).diagonal()
    time2 = time.time()
    print("Time taken for dense diag product: ", time2 - time1)

    print("diff diag: ", xp.linalg.norm(diag - diag_ref))


