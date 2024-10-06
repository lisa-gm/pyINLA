from scipy.sparse import csc_matrix


def write_sparse_csc_matrix(A, filename):
    """
    Writes a sparse matrix in CSC format to a file compatible with the specified C++ format.

    Parameters:
    -----------
    A : scipy.sparse.csc_matrix
        The sparse matrix to be written to the file.
    filename : str
        The name of the output file.
    """
    # Ensure the input is a CSC matrix
    if not isinstance(A, csc_matrix):
        A = csc_matrix(A)
        print("Matrix was converted to CSC format.")

    n_rows, n_cols = A.shape
    nnz = A.nnz  # Number of non-zero entries

    # Get data from the sparse matrix
    inner_indices = A.indices  # Row indices of non-zero entries
    outer_index_ptr = A.indptr  # Column pointers
    values = A.data  # Non-zero values

    # Write to file
    with open(filename, "w") as f:
        f.write(f"{n_rows}\n")  # Write number of rows
        f.write(f"{n_cols}\n")  # Write number of columns
        f.write(f"{nnz}\n")  # Write number of non-zero entries

        # Write inner indices
        for index in inner_indices:
            f.write(f"{index}\n")

        # Write outer index pointers
        for ptr in outer_index_ptr:
            f.write(f"{ptr}\n")

        # Write values
        for value in values:
            f.write(f"{value}\n")
