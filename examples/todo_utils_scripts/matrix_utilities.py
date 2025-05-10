######## Matrix Utilities ########

import numpy as np
import os
import scipy.sparse as sp

from scipy.sparse import csc_matrix


def load_matrices_data_component_from_npz(ns, nt, nb, no, file_path):
    n = ns * nt + nb
    ####### data #######

    file_name_y = f"y.npy"
    y = np.loadtxt(file_path + "/" + file_name_y)

    ####### covariates to form arrowhead structure #######

    # Ax
    Ax = sp.load_npz(os.path.join(file_path, "Ax.npz"))

    return y, Ax


def load_matrices_spatial_temporal_model_from_npz(ns, nt, file_path):

    ###### spatial submatrices ######
    # c0
    file_name_c0 = f"c0.npz"
    c0 = sp.load_npz(file_path + "/" + file_name_c0)

    # g1
    file_name_g1 = f"g1.npz"
    g1 = sp.load_npz(file_path + "/" + file_name_g1)

    # g2
    file_name_g2 = f"g2.npz"
    g2 = sp.load_npz(file_path + "/" + file_name_g2)

    # g3
    file_name_g3 = f"g3.npz"
    g3 = sp.load_npz(file_path + "/" + file_name_g3)

    ###### temporal submatrices ######
    # M0
    file_name_M0 = f"m0.npz"
    M0 = sp.load_npz(file_path + "/" + file_name_M0)

    # M1
    file_name_M1 = f"m1.npz"
    M1 = sp.load_npz(file_path + "/" + file_name_M1)

    # M2
    file_name_M2 = f"m2.npz"
    M2 = sp.load_npz(file_path + "/" + file_name_M2)

    return c0, g1, g2, g3, M0, M1, M2


def load_matrices_data_component_from_dat(ns, nt, nb, no, file_path):
    n = ns * nt + nb
    ####### data #######

    file_name_y = f"y_{no}_1.dat"
    y = np.loadtxt(file_path + "/" + file_name_y)

    ####### covariates to form arrowhead structure #######

    # Ax
    file_name_Ax = f"Ax_{no}_{n}.dat"
    Ax = read_CSC(file_path + "/" + file_name_Ax)

    return y, Ax


def load_matrices_spatial_temporal_model_from_dat(ns, nt, file_path):

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

    return c0, g1, g2, g3, M0, M1, M2


def load_matrices_spatial_model_from_dat(ns, file_path):

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

    return (
        c0,
        g1,
        g2,
    )


def read_sym_CSC(filename):
    with open(filename, "r") as f:
        n = int(f.readline().strip())
        n = int(f.readline().strip())
        nnz = int(f.readline().strip())

        inner_indices = np.zeros(nnz, dtype=int)
        outer_index_ptr = np.zeros(n + 1, dtype=int)
        values = np.zeros(nnz, dtype=float)

        for i in range(nnz):
            inner_indices[i] = int(f.readline().strip())

        for i in range(n + 1):
            outer_index_ptr[i] = int(f.readline().strip())

        for i in range(nnz):
            values[i] = float(f.readline().strip())

    # Create the lower triangular CSC matrix
    A = csc_matrix((values, inner_indices, outer_index_ptr), shape=(n, n))

    return A


def read_CSC(filename):
    with open(filename, "r") as f:
        nrows = int(f.readline().strip())
        ncols = int(f.readline().strip())
        nnz = int(f.readline().strip())

        inner_indices = np.zeros(nnz, dtype=int)
        outer_index_ptr = np.zeros(ncols + 1, dtype=int)
        values = np.zeros(nnz, dtype=float)

        for i in range(nnz):
            inner_indices[i] = int(f.readline().strip())

        for i in range(ncols + 1):
            outer_index_ptr[i] = int(f.readline().strip())

        for i in range(nnz):
            values[i] = float(f.readline().strip())

    # Create CSC matrix
    A = csc_matrix((values, inner_indices, outer_index_ptr), shape=(nrows, ncols))

    return A
