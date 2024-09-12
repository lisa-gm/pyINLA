import argparse
import time

import cupy as cp
import numpy as np
from construct_precision_matrices import Q_spatio_temporal, construct_Q
from construct_precision_matrices_dev import Q_spatio_temporal_dev, construct_Q_dev
from cupyx.scipy.sparse import coo_matrix as cpcoo_matrix
from cupyx.scipy.sparse import csr_matrix as cpcsr_matrix
from cupyx.scipy.sparse import kron as cpkron
from matrix_utilities import read_CSC, read_sym_CSC
from scipy.sparse import csr_matrix, diags
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
from scipy.sparse import random
from scipy.sparse.linalg import norm


def eliminate_zeros_cupy(cupy_sparse_matrix):
    # Convert to COO format
    coo = cupy_sparse_matrix.tocoo()

    # Filter out zero entries
    mask = coo.data != 0
    coo_filtered = cpcoo_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
    )

    # Convert back to CSR format
    return coo_filtered.tocsr()


# Function to generate a random sparse matrix
def generate_sparse_matrix(rows, cols, density=0.1):
    # for large matrices, reduce density, fix to 200 non-zero elements per row
    if rows > 1000:
        density = 200 / rows
    return random(rows, cols, density=density, format="csr")


def find_nan_indices(matrix):
    # Create a boolean matrix where True indicates the presence of NaN
    nan_mask = np.isnan(matrix)

    # Find the indices of NaN values
    nan_indices = np.argwhere(nan_mask)

    return nan_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    # TODO: randomData argument doesn't work
    parser.add_argument(
        "--randomData",
        type=bool,
        default=False,
        help="generate random sparse matrices",
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=42,
        help="blocksize of diagonal blocks",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=3,
        help="number of diagonal blocks",
    )
    parser.add_argument(
        "--nb",
        type=int,
        default=3,
        help="number of diagonal blocks",
    )
    parser.add_argument(
        "--no",
        type=int,
        default=3,
        help="number of diagonal blocks",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns42_nt3_nb2",
        help="a string for the file path",
    )

    args = parser.parse_args()

    randData = args.randomData
    # diagonal block size
    ns = args.ns
    # number of ns x ns blocks
    nt = args.nt
    # number of rows in arrow
    nb = args.nb
    # number of observations
    no = args.no
    file_path = args.file_path

    n = ns * nt + nb
    print("ns: {}, nt: {}, n: {}".format(ns, nt, n))

    randData = False

    print("randData: ", randData)

    if randData:
        print("Generating random sparse matrices")

        # Generate two random sparse matrices
        nrows_1 = ns
        matrix1 = generate_sparse_matrix(nrows_1, nrows_1, density=0.1) + speye(nrows_1)

        # generate sparse tridiagonal matrix
        nrows_2 = nt

        diagonals = [
            0.5 * np.ones((nrows_2 - 1,)),
            np.ones((nrows_2,)),
            0.5 * np.ones((nrows_2 - 1,)),
        ]
        offsets = [-1, 0, 1]
        matrix2 = diags(diagonals, offsets, format="csr")
        # print(matrix2.toarray())

        # Compute their Kronecker product
        start_time = time.time()
        kronecker_product = spkron(matrix1, matrix2)
        end_time = time.time()
        print(
            "SciPy: Time Kronecker product: {:.3f} seconds".format(
                end_time - start_time
            )
        )

        # Convert matrices to CuPy sparse matrices
        matrix1_gpu = cpcsr_matrix(matrix1)
        matrix2_gpu = cpcsr_matrix(matrix2)

        # Compute their Kronecker product on GPU
        start_time = time.time()
        kronecker_product_gpu = cpkron(matrix1_gpu, matrix2_gpu)
        end_time = time.time()
        print(
            "CuPy:  Time Kronecker product: {:.3f} seconds".format(
                end_time - start_time
            )
        )

        # Pybind11: pass-by-reference not supported for sparse matrices
        # https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html

        # Print the result
        # print("Matrix 1:\n", matrix1.toarray())
        # print("Matrix 2:\n", matrix2.toarray())
        # print("Kronecker Product:\n", kronecker_product.toarray())

    else:
        print("Importing sparse matrices from file")
        # Load submatrices

        ###### spatial submatrices ######
        # c0
        file_name_c0 = f"c0_{ns}.dat"
        c0_lower = read_sym_CSC(file_path + "/" + file_name_c0)
        c0 = c0_lower + c0_lower.T
        c0.setdiag(c0_lower.diagonal())

        # g1
        file_name_g1 = f"g1_{ns}.dat"
        g1_lower = read_sym_CSC(file_path + "/" + file_name_g1)
        g1 = g1_lower + g1_lower.T
        g1.setdiag(g1_lower.diagonal())

        # g2
        file_name_g2 = f"g2_{ns}.dat"
        g2_lower = read_sym_CSC(file_path + "/" + file_name_g2)
        g2 = g2_lower + g2_lower.T
        g2.setdiag(g2_lower.diagonal())

        # g3
        file_name_g3 = f"g3_{ns}.dat"
        g3_lower = read_sym_CSC(file_path + "/" + file_name_g3)
        g3 = g3_lower + g3_lower.T
        g3.setdiag(g3_lower.diagonal())
        # print("g3:\n", g3.toarray())

        ###### temporal submatrices ######
        # M0
        file_name_M0 = f"M0_{nt}.dat"
        M0_lower = read_sym_CSC(file_path + "/" + file_name_M0)
        M0 = M0_lower + M0_lower.T
        M0.setdiag(M0_lower.diagonal())
        # print("M0:\n", M0.toarray())

        # M1
        file_name_M1 = f"M1_{nt}.dat"
        M1_lower = read_sym_CSC(file_path + "/" + file_name_M1)
        M1 = M1_lower + M1_lower.T
        M1.setdiag(M1_lower.diagonal())

        # M2
        file_name_M2 = f"M2_{nt}.dat"
        M2_lower = read_sym_CSC(file_path + "/" + file_name_M2)
        M2 = M2_lower + M2_lower.T
        M2.setdiag(M2_lower.diagonal())

        ####### covariates to form arrowhead structure #######

        # Ax
        file_name_Ax = f"Ax_{no}_{n}.dat"
        Ax = read_CSC(file_path + "/" + file_name_Ax)

        AxTAx = Ax.T @ Ax
        AxTAx.eliminate_zeros()

        theta_initial = [1, 1, 1, 1]
        theta = theta_initial

        ######## assemble the matrices - CPU Version &  GPU version ########

        theta_dev = cp.array(theta)

        # only do this once
        c0_dev = cpcsr_matrix(c0)
        g1_dev = cpcsr_matrix(g1)
        g2_dev = cpcsr_matrix(g2)
        g3_dev = cpcsr_matrix(g3)

        M0_dev = cpcsr_matrix(M0)
        M1_dev = cpcsr_matrix(M1)
        M2_dev = cpcsr_matrix(M2)

        ## is the GPU getting too full??
        Ax_dev = cpcsr_matrix(Ax)

        # AxTAx_dev = cpcsr_matrix(AxTAx)

        # Multiplication introduces zero elements in csr structure
        # DONT USE AxTAx_dev.eliminate_zeros() on device -> corrupts the matrix
        AxTAx_dev = Ax_dev.T @ Ax_dev

        num_zeros = cp.sum(AxTAx_dev.data == 0)
        print("Number of zeros in AxTAx.data: ", num_zeros)

        AxTAx_dev = eliminate_zeros_cupy(AxTAx_dev)

        num_zeros = cp.sum(AxTAx_dev.data == 0)
        print("Number of zeros in AxTAx.data after elimination: ", num_zeros)

        AxTAx_from_dev = AxTAx_dev.get()
        print("norm(AxTAx - AxTAx_from_dev): ", norm(AxTAx - AxTAx_from_dev))

        print("size(data(AxTAx)): ", AxTAx.data.shape)
        print("size(data(AxTAx_dev)): ", AxTAx_dev.data.shape)

        Ax_memory = Ax.data.nbytes + Ax.indices.nbytes + Ax.indptr.nbytes
        print(f"Ax allocated memory     : {Ax_memory / (1024**2):.2f} MB")

        Ax_dev_memory = (
            Ax_dev.data.nbytes + Ax_dev.indices.nbytes + Ax_dev.indptr.nbytes
        )
        print(f"Ax    allocated GPU memory: {Ax_dev_memory / (1024**2):.2f} MB")

        AxTAx_memory = AxTAx.data.nbytes + AxTAx.indices.nbytes + AxTAx.indptr.nbytes
        print(f"AxTAx allocated memory  : {AxTAx_memory / (1024**2):.2f} MB")

        AxTAx_dev_memory = (
            AxTAx_dev.data.nbytes + AxTAx_dev.indices.nbytes + AxTAx_dev.indptr.nbytes
        )
        print(f"AxTAx allocated GPU memory: {AxTAx_dev_memory / (1024**2):.2f} MB")

        # initial value of theta
        print("theta_initial: ", theta_initial)
        theta_dev = cp.array(theta_initial)

        print("Constructing Q ...")

        m = 1
        for i in range(m):
            print("\nIteration: ", i)
            # make this a reference and not a copy ...
            t_start = time.time()
            Qst = Q_spatio_temporal(theta[1:], c0, g1, g2, g3, M0, M1, M2)
            Qst = csr_matrix(Qst)

            Q = construct_Q(nb, theta, Qst, AxTAx)
            t_end = time.time()

            # print("Q:\n", Q[1:10,1:10].toarray())
            print(
                "SciPy: Total time to construct Q: {:.3f} seconds".format(
                    t_end - t_start
                )
            )
            # print("Qst[1:10,1:10]:\n", Qst[:10,:10].toarray())

            t_start = time.time()
            Qst_dev = Q_spatio_temporal_dev(
                theta_dev[1:], c0_dev, g1_dev, g2_dev, g3_dev, M0_dev, M1_dev, M2_dev
            )
            Q_dev = construct_Q_dev(nb, theta_dev, Qst_dev, AxTAx_dev)
            t_end = time.time()
            print(
                "CuPy:  Total time to construct Q: {:.3f} seconds".format(
                    t_end - t_start
                )
            )

            theta = theta + np.random.rand(len(theta))
            theta_dev = cp.array(theta)
            # theta_dev = theta_dev + cp.random.rand(theta_dev.shape[0])

        # Qst_from_dev = Qst_dev.get() #Q_dev.get()
        # print("nnz(Qst_from_dev): ", Qst_from_dev.nnz)
        # print("norm(Qst - Qst_from_dev): ", norm(Qst - Qst_from_dev))

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        print(
            "\nGPU Memory pool used            : {:.2f} MB".format(
                mempool.used_bytes() / 1024**2
            )
        )
        print(
            "GPU mempool total               : {:.2f} MB".format(
                mempool.total_bytes() / 1024**2
            )
        )
        print(
            "GPU Pinned memory n free blocks : {:.2f} MB".format(
                pinned_mempool.n_free_blocks() / 1024**2
            )
        )

        # theoretical memory usage
        Q_memory = Q.data.nbytes + Q.indices.nbytes + Q.indptr.nbytes
        print(f"\nQ allocated memory    : {Q_memory / (1024**2):.2f} MB")

        Qst_memory = Qst.data.nbytes + Qst.indices.nbytes + Qst.indptr.nbytes
        print(f"Qst allocated memory    : {Qst_memory / (1024**2):.2f} MB")

        AxTAx_memory = AxTAx.data.nbytes + AxTAx.indices.nbytes + AxTAx.indptr.nbytes
        print(f"AxTAx allocated memory  : {AxTAx_memory / (1024**2):.2f} MB")

        Ax_memory = Ax.data.nbytes + Ax.indices.nbytes + Ax.indptr.nbytes
        print(f"Ax allocated memory     : {Ax_memory / (1024**2):.2f} MB")

        print(
            f"Sum of allocated memory : {(Q_memory + Qst_memory + AxTAx_memory + Ax_memory) / (1024**2):.2f} MB"
        )

        Q_from_dev = Q_dev.get()
        print("\nnorm(Q - Q_from_dev): ", norm(Q - Q_from_dev))

        # find maximum difference between the two matrices
        diff = Q - Q_from_dev
        max_diff = np.max(np.abs(diff))
        print("max_diff: ", max_diff)

        # isequal = np.allclose(Q.toarray(), Q_from_dev.toarray())
