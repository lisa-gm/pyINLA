import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import math

from scipy.sparse import csc_matrix, save_npz, block_diag

from pyinla import sp, xp

from matrix_utilities import (
    read_CSC,
    read_sym_CSC,
    load_matrices_spatial_model_from_dat,
    load_matrices_spatial_temporal_model_from_dat,
)


def construct_Qprior_spatial(range_s, sigma_s, c0, g1, g2):

    gamma_s, gamma_e = _interpretable2compute_s(range_s, sigma_s)
    print("gamma s: ", gamma_s, "gamma e: ", gamma_e)
    exp_gamma_s = np.exp(gamma_s)
    exp_gamma_e = np.exp(gamma_e)

    Q = pow(exp_gamma_e, 2) * (
        pow(exp_gamma_s, 4) * c0 + 2 * pow(exp_gamma_s, 2) * g1 + g2
    )

    return Q


def _interpretable2compute_s(
    r_s: float, sigma_e: float, dim_spatial_domain: int = 2
) -> tuple:
    if dim_spatial_domain != 2:
        raise ValueError("Only 2D spatial domain is supported for now.")

    alpha = 2
    nu_s = alpha - dim_spatial_domain / 2
    gamma_s = 0.5 * np.log(8 * nu_s) - r_s
    gamma_e = 0.5 * (
        sp.special.gamma(nu_s)
        - (
            sp.special.gamma(alpha)
            + 0.5 * dim_spatial_domain * np.log(4 * np.pi)
            + 2 * nu_s * gamma_s
            + 2 * sigma_e
        )
    )

    return gamma_s, gamma_e


def construct_Q_spatio_temporal(
    r_s, r_t, sigma_st, c0, g1, g2, g3, m0, m1, m2
) -> sp.sparse.coo_matrix:
    """Construct the prior precision matrix."""

    gamma_s, gamma_t, gamma_st = _interpretable2compute_st(
        r_s,
        r_t,
        sigma_st,
        dim_spatial_domain=2,
    )

    print(
        f"Thetas used in Qprior construction: gamma_s: {gamma_s}, gamma_t: {gamma_t}, gamma_st: {gamma_st}"
    )

    exp_gamma_s = xp.exp(gamma_s)
    exp_gamma_t = xp.exp(gamma_t)
    exp_gamma_st = xp.exp(gamma_st)

    q1s = pow(exp_gamma_s, 2) * c0 + g1
    q2s = pow(exp_gamma_s, 4) * c0 + 2 * pow(exp_gamma_s, 2) * g1 + g2
    q3s = (
        pow(exp_gamma_s, 6) * c0
        + 3 * pow(exp_gamma_s, 4) * g1
        + 3 * pow(exp_gamma_s, 2) * g2
        + g3
    )

    # withsparseKroneckerProduct", color_id=0):
    Q_prior: sp.sparse.spmatrix = sp.sparse.csc_matrix(
        pow(exp_gamma_st, 2)
        * (
            sp.sparse.kron(m0, q3s)
            + exp_gamma_t * sp.sparse.kron(m1, q2s)
            + pow(exp_gamma_t, 2) * sp.sparse.kron(m2, q1s)
        )
    )

    return Q_prior


def _interpretable2compute_st(
    r_s: float, r_t: float, sigma_st: float, dim_spatial_domain: int = 2
) -> tuple:
    if dim_spatial_domain != 2:
        raise ValueError("Only 2D spatial domain is supported for now.")

    # manifold = "sphere"
    manifold = "plane"

    # Assumes alphas as fixed for now
    alpha_s = 2
    alpha_t = 1
    alpha_e = 1

    # Implicit assumption that spatial domain is 2D
    alpha = alpha_e + alpha_s * (alpha_t - 0.5)

    nu_s = alpha - 1
    nu_t = alpha_t - 0.5

    gamma_s = 0.5 * xp.log(8 * nu_s) - r_s
    gamma_t = r_t - 0.5 * xp.log(8 * nu_t) + alpha_s * gamma_s

    if manifold == "sphere":
        cR_t = sp.special.gamma(nu_t) / (
            sp.special.gamma(alpha_t) * pow(4 * math.pi, 0.5)
        )
        c_s = 0.0
        for k in range(50):
            c_s += (2.0 * k + 1.0) / (
                4.0 * math.pi * pow(pow(xp.exp(gamma_s), 2) + k * (k + 1), alpha)
            )
        gamma_st = 0.5 * xp.log(cR_t) + 0.5 * xp.log(c_s) - 0.5 * gamma_t - sigma_st

    elif manifold == "plane":
        c1_scaling_constant = pow(4 * math.pi, 1.5)
        c1 = (
            sp.special.gamma(nu_t)
            * sp.special.gamma(nu_s)
            / (
                sp.special.gamma(alpha_t)
                * sp.special.gamma(alpha)
                * c1_scaling_constant
            )
        )
        gamma_st = 0.5 * xp.log(c1) - 0.5 * gamma_t - nu_s * gamma_s - sigma_st
    else:
        raise ValueError("Manifold not supported: ", manifold)

    return gamma_s, gamma_t, gamma_st


def construct_Qprior_coreg(
    num_vars, type, ns, nt, nb, theta, c0, g1, g2, g3=None, m0=None, m1=None, m2=None
):

    if type == "spatial":
        sigma1 = np.exp(theta[0])
        Q1 = construct_Qprior_spatial(theta[1], np.log(1), c0, g1, g2)
        print("Q1[:10, :10] = \n", Q1[:10, :10].toarray())
        # print(Q1.todense())
        print(Q1.shape)
        sigma2 = np.exp(theta[2])
        Q2 = construct_Qprior_spatial(theta[3], np.log(1), c0, g1, g2)
        print("Q2[:10, :10] = \n", Q2[:10, :10].toarray())
        # print(Q2.todense())
        print(Q2.shape)

        if num_vars == 2:
            lambda1 = theta[4]

        if num_vars == 3:
            sigma3 = np.exp(theta[4])
            print("theta[5]: ", theta[5])
            Q3 = construct_Qprior_spatial(theta[5], np.log(1), c0, g1, g2)
            # print(Q3.todense())
            print(Q3.shape)
            lambda1 = theta[6]
            lambda2 = theta[7]
            lambda3 = theta[8]

    elif type == "spatio-temporal":
        sigma1 = np.exp(theta[0])
        Q1 = construct_Q_spatio_temporal(
            theta[1], theta[2], np.log(1), c0, g1, g2, g3, m0, m1, m2
        )
        print("Q1[:5, :5] = \n", Q1[:5, :5].toarray())

        sigma2 = np.exp(theta[3])
        Q2 = construct_Q_spatio_temporal(
            theta[4], theta[5], np.log(1), c0, g1, g2, g3, m0, m1, m2
        )
        print("Q2[:5, :5] = \n", Q2[:5, :5].toarray())

        if num_vars == 2:
            lambda1 = theta[6]

        if num_vars == 3:
            sigma3 = np.exp(theta[6])
            Q3 = construct_Q_spatio_temporal(
                theta[7], theta[8], np.log(1), c0, g1, g2, g3, m0, m1, m2
            )
            print("Q3[:5, :5] = \n", Q3[:5, :5].toarray())

            lambda1 = theta[9]
            lambda2 = theta[10]
            lambda3 = theta[11]

    # assemble Qprior
    if num_vars == 2:
        Q = sp.sparse.vstack(
            [
                sp.sparse.hstack(
                    [
                        (1 / sigma1**2) * Q1 + (lambda1**2 / sigma2**2) * Q2,
                        (-lambda1 / sigma2**2) * Q2,
                    ]
                ),
                sp.sparse.hstack([(-lambda1 / sigma2**2) * Q2, (1 / sigma2**2) * Q2]),
            ]
        )

    elif num_vars == 3:
        Q = sp.sparse.vstack(
            [
                sp.sparse.hstack(
                    [
                        (1 / sigma1**2) * Q1
                        + (lambda1**2 / sigma2**2) * Q2
                        + (lambda3**2 / sigma3**2) * Q3,
                        (-lambda1 / sigma2**2) * Q2
                        + (lambda2 * lambda3 / sigma3**2) * Q3,
                        -lambda3 / sigma3**2 * Q3,
                    ]
                ),
                sp.sparse.hstack(
                    [
                        (-lambda1 / sigma2**2) * Q2
                        + (lambda2 * lambda3 / sigma3**2) * Q3,
                        (1 / sigma2**2) * Q2 + (lambda2**2 / sigma3**2) * Q3,
                        -lambda2 / sigma3**2 * Q3,
                    ]
                ),
                sp.sparse.hstack(
                    [
                        -lambda3 / sigma3**2 * Q3,
                        -lambda2 / sigma3**2 * Q3,
                        (1 / sigma3**2) * Q3,
                    ]
                ),
            ]
        )

    else:
        raise ValueError("Invalid number of variables")

    return Q


def generate_permutation(n, num_vars):
    """
    Generate a 1D array in the form [0, n, 1, n+1, 2, n+2, ...].
    or for 3 variables [0, n, 2n, 1, n+1, 2n+1, 2, n+2, 2n+2, ...].

    Parameters
    ----------
    n : int
        The size of the sequence.

    Returns
    -------
    np.ndarray
        The generated permutation array.
    """
    idx_first_mat = np.arange(n)
    idx_second_mat = np.arange(n, 2 * n)

    if num_vars == 2:
        perm = np.empty((2 * n,), dtype=int)
        perm[0::2] = idx_first_mat
        perm[1::2] = idx_second_mat

    elif num_vars == 3:
        idx_third_mat = np.arange(2 * n, 3 * n)
        perm = np.empty((3 * n,), dtype=int)
        perm[0::3] = idx_first_mat
        perm[1::3] = idx_second_mat
        perm[2::3] = idx_third_mat
    else:
        raise ValueError("Invalid number of variables")

    return perm


def generate_permutation_matrix(n, num_vars):
    """
    Generate a permutation matrix using the permutation scheme [0, n, 1, n+1, 2, n+2, ...].

    Parameters
    ----------
    n : int
        Size of the matrix.

    Returns
    -------
    sp.spmatrix
        The permutation matrix.
    """
    perm = generate_permutation(n, num_vars)
    # print("Permutation indices:", perm)

    # Vectorized creation of the permutation matrix
    n = len(perm)
    row_indices = np.arange(n)
    col_indices = perm
    data = np.ones(n, dtype=int)

    permutation_matrix = sp.sparse.coo_matrix(
        (data, (row_indices, col_indices)), shape=(n, n)
    ).tocsc()

    permutation_matrix2 = sp.sparse.csc_matrix((len(perm), len(perm)), dtype=int)

    for i in range(len(perm)):
        permutation_matrix2[i, perm[i]] = 1

    assert np.all(
        permutation_matrix.toarray() == permutation_matrix2.toarray()
    ), "Permutation matrices do not match!"

    return permutation_matrix


def generate_block_permutation_matrix(n_blocks, block_size, num_vars):
    """
    Generate a permutation matrix using the permutation scheme [0, n, 1, n+1, 2, n+2, ...].

    Parameters
    ----------
    n : int
        Size of the matrix.

    Returns
    -------
    sp.spmatrix
        The permutation matrix.
    """
    perm = generate_permutation(n_blocks, num_vars)

    n_perm_mat = len(perm) * block_size
    permutation_matrix = sp.sparse.csc_matrix((n_perm_mat, n_perm_mat), dtype=int)

    # print("permutation vector:")
    for i in range(len(perm)):
        permutation_matrix[
            i * block_size : (i + 1) * block_size,
            perm[i] * block_size : (perm[i] + 1) * block_size,
        ] = sp.sparse.eye(block_size)
        seq = list(range(perm[i] * block_size, (perm[i] + 1) * block_size))

    return permutation_matrix


# def generate_block_permutation_matrix(n_blocks, block_size, num_vars):
#     """
#     Generate a permutation matrix using the permutation scheme [0, n, 1, n+1, 2, n+2, ...].

#     Parameters
#     ----------
#     n : int
#         Size of the matrix.

#     Returns
#     -------
#     sp.spmatrix
#         The permutation matrix.
#     """
#     perm = generate_permutation(n_blocks, num_vars)

#     n_perm_mat = len(perm) * block_size
#     permutation_matrix = sp.csc_matrix((n_perm_mat,n_perm_mat), dtype=int)

#     print("permutation vector:")
#     for i in range(len(perm)):
#         permutation_matrix[i*block_size:(i+1)*block_size, perm[i]*block_size:(perm[i]+1)*block_size] = sp.eye(block_size)
#         seq = list(range(perm[i]*block_size, (perm[i]+1)*block_size))

#         print(seq)

#     return permutation_matrix


def generate_permutation_indices(n_blocks, block_size, num_vars):
    """
    Generate a permutation vector containing indices in the pattern:
    [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

    Parameters
    ----------
    n_blocks : int
        Number of blocks.
    block_size : int
        Size of each block.

    Returns
    -------
    np.ndarray
        The generated permutation vector.
    """
    indices = np.arange(n_blocks * block_size)

    first_idx = indices.reshape(n_blocks, block_size)
    second_idx = first_idx + n_blocks * block_size

    if num_vars == 2:
        perm_vectorized = np.hstack((first_idx, second_idx)).flatten()
    if num_vars == 3:
        third_idx = second_idx + n_blocks * block_size
        perm_vectorized = np.hstack((first_idx, second_idx, third_idx)).flatten()

    return perm_vectorized


def permute_block_tridiagonal_matrix(matrix, block_size, n_blocks):
    """
    Permute a block tridiagonal matrix using the permutation scheme [0:block_size, n*block_size:(n+1)*block_size, 1*blocksize:(1+1)*blocksize, (n+1)*block_size:(n+1+1)*block_size, 2*block_size:(2+1)*blocksize, ...].

    Parameters
    ----------
    matrix : sp.spmatrix
        The block tridiagonal matrix to permute.
    block_size : int
        Size of each block.
    n_blocks : int
        Number of blocks along the main diagonal.

    Returns
    -------
    sp.spmatrix
        The permuted block tridiagonal matrix.
    """
    n = block_size * n_blocks
    perm = generate_permutation(n_blocks, num_vars)
    perm = np.repeat(perm, block_size) + np.tile(np.arange(block_size), n_blocks)
    permuted_matrix = matrix[perm, :][:, perm]
    return permuted_matrix


if __name__ == "__main__":

    num_vars = 3

    type = "spatio-temporal"  # "spatial"

    ns = 354
    nt = 8

    # add more shared fixed effects later
    nb = num_vars

    no1 = 8496
    no2 = 8496
    no3 = 8496  # 0

    no_list = [no1, no2, no3]
    total_obs = sum(no_list)

    data_dir = f"/Users/usi/Documents/PhD/paper_submissions/SC_pyINLA/application/coregionalization_models/data/nv{num_vars}_ns{ns}_nt{nt}_nb{nb}"

    # load submatrices
    if type == "spatial":
        c0, g1, g2 = load_matrices_spatial_model_from_dat(ns, data_dir)

    elif type == "spatio-temporal":
        c0, g1, g2, g3, M0, M1, M2 = load_matrices_spatial_temporal_model_from_dat(
            ns, nt, data_dir
        )
    else:
        raise ValueError("Invalid model type")

    # order theta vectr for spatial
    # 2 variable
    # theta = [sigma1, range1, sigma2, range2, lambda1]
    # 3 variable
    # theta = [sigma1, range1, sigma2, range2, sigma3, range3, lambda1, lambda2, lambda3]
    # for spatial temporal model
    # 2 variable
    # theta = [sigma1, range1, range_t1, sigma2, range2, range_t2, lambda1]
    # 3 variable
    # theta = [sigma1, range_s1, range_t1, sigma2, range_s2, range_t2, sigma3, range_s3, range_t3, lambda1, lambda2, lambda3]

    sigma1 = 0.5
    sigma2 = 0.6

    lambda1 = 1.5

    if "spatial" in type:
        range1 = 2
        range2 = 3

        if num_vars == 2:
            theta = np.array([sigma1, range1, sigma2, range2, lambda1])
        elif num_vars == 3:
            sigma3 = 1
            range3 = 1

            lambda2 = -1
            lambda3 = 2
            theta = np.array(
                [
                    sigma1,
                    range1,
                    sigma2,
                    range2,
                    sigma3,
                    range3,
                    lambda1,
                    lambda2,
                    lambda3,
                ]
            )
    elif "spatio-temporal" in type:
        range_s1 = 2
        range_t1 = 1

        range_s2 = 3
        range_t2 = 2

        if num_vars == 2:
            theta = np.array(
                [
                    sigma1,
                    range_s1,
                    range_t1,
                    sigma2,
                    range_s2,
                    range_t2,
                    lambda1,
                ]
            )
        elif num_vars == 3:
            sigma3 = 1

            range_s3 = 1
            range_t3 = 1

            lambda2 = -1
            lambda3 = 2
            theta = np.array(
                [
                    sigma1,
                    range_s1,
                    range_t1,
                    sigma2,
                    range_s2,
                    range_t2,
                    sigma3,
                    range_s3,
                    range_t3,
                    lambda1,
                    lambda2,
                    lambda3,
                ]
            )

    # or alternatively load theta from file
    dim_theta = len(theta) + num_vars
    theta_file = f"{data_dir}/theta_interpretS_original_{dim_theta}_1.dat"
    theta = np.loadtxt(theta_file)

    print("theta: ", theta)

    if type == "spatial":
        Q = construct_Qprior_coreg(num_vars, type, ns, nt, nb, theta, c0, g1, g2)
    else:
        Q = construct_Qprior_coreg(
            num_vars, type, ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2
        )

    print("dimensions of Q: ", Q.shape)
    # print("Q: \n", Q.toarray())
    print("Qprior_new[:10, :10] = \n", Q[-10:, -10:].toarray())

    # Q_upper = sp.sparse.triu(Q)

    # save spy plot of Q to .png
    plt.figure()
    plt.spy(Q)
    plt.savefig("Q.png")
    plt.close()

    dim_Q = num_vars * (ns * nt)
    # load Q from file and compare
    Qprior_file = f"{data_dir}/Qprior_R_{dim_Q}_{dim_Q}.dat"
    Qprior_R = read_CSC(Qprior_file)

    print("Qprior_R[:10, :10] = \n", Qprior_R[-10:, -10:].toarray())

    # check if Q_upper and Qprior_R are equal
    diff = Q - Qprior_R
    print("max(abs(Q - Qprior_R)): ", np.max(np.abs(diff)))

    threshold = 1e-4

    # Find the indices where the absolute values of the elements in diff are greater than the threshold
    diff_abs = np.abs(diff)
    rows, cols = diff_abs.nonzero()
    values = diff_abs.data

    # Filter the indices based on the threshold
    filtered_indices = [(r, c) for r, c, v in zip(rows, cols, values) if v > threshold]

    # Print the filtered indices
    print("Indices where abs(diff) > 1e-4:", filtered_indices[:10])

    # load observation vectors
    y1_file = f"{data_dir}/y1_{no1}_1.dat"
    y1 = np.loadtxt(y1_file)

    y2_file = f"{data_dir}/y2_{no2}_1.dat"
    y2 = np.loadtxt(y2_file)

    # load projection matrices
    a1_file = f"{data_dir}/A1_{no1}_{ns*nt}.dat"
    a1 = read_CSC(a1_file)
    print("nnz(A1) = ", a1.nnz)

    print("nnz(a1[:100, :100]) = ", a1[:100, :100].nnz)
    print(a1[:10, :10].toarray())

    # Define a custom colormap where only zero is white
    colors = ["white", "blue", "green", "red", "yellow"]
    cmap = ListedColormap(colors)
    bounds = [
        -1e-9,
        1e-9,
        1e-3,
        1e-1,
        1,
        1e9,
    ]  # Adjust bounds to ensure only 0 is white
    norm = BoundaryNorm(bounds, cmap.N)

    # Visualize using matplotlib.pyplot.imshow with custom colormap
    # plt.figure(figsize=(10, 10))
    # plt.imshow(a1.toarray(), cmap=cmap, norm=norm, aspect="auto")
    # plt.colorbar()
    # plt.title("a1 submatrix")
    # plt.savefig("a1_custom_colormap.png")

    a2_file = f"{data_dir}/A2_{no2}_{ns*nt}.dat"
    a2 = read_CSC(a2_file)

    if num_vars == 3:
        a3_file = f"{data_dir}/A3_{no3}_{ns*nt}.dat"
        a3 = read_CSC(a3_file)

        y3_file = f"{data_dir}/y3_{no3}_1.dat"
        y3 = np.loadtxt(y3_file)

    if num_vars == 2:
        y = np.concatenate([y1, y2])
        A = block_diag((a1, a2))
    elif num_vars == 3:
        y = np.concatenate([y1, y2, y3])
        A = block_diag((a1, a2, a3))

    print("dimensions of y: ", y.shape)
    print("dimensions of A: ", A.shape)

    # add 1's for intercept for each submodel
    A_intercept = np.zeros((total_obs, num_vars))

    for i in range(num_vars):
        if i == 0:
            first_idx = 0
        else:
            first_idx = sum(no_list[:i])
        last_idx = sum(no_list[: i + 1])
        A_intercept[first_idx:last_idx, i] = 1

    # print("A_intercept: \n", A_intercept)

    A = sp.sparse.hstack([A, A_intercept])
    # print("A: \n", A.toarray())
    print("dim(A) = ", A.shape)

    A = csc_matrix(A)

    print("A[:10, -6:] = \n", A[:10, -6:].toarray())
    print("A[no1-3: no1+3, -6:] = \n", A[no1 - 3 : no1 + 3, -6:].toarray())

    # Calculate the diagonal values
    if num_vars == 2:
        if type == "spatial":
            prec_obs1 = np.exp(theta[5])
            prec_obs2 = np.exp(theta[6])
        elif type == "spatio-temporal":
            prec_obs1 = np.exp(theta[7])
            prec_obs2 = np.exp(theta[8])
        diag_values = np.concatenate([np.full(no1, prec_obs1), np.full(no2, prec_obs2)])
        print("prec_obs1 ", prec_obs1, "prec_obs2 ", prec_obs2)

    if num_vars == 3:
        if type == "spatial":
            prec_obs1 = np.exp(theta[9])
            prec_obs2 = np.exp(theta[10])
            prec_obs3 = np.exp(theta[11])
        elif type == "spatio-temporal":
            prec_obs1 = np.exp(theta[12])
            prec_obs2 = np.exp(theta[13])
            prec_obs3 = np.exp(theta[14])

        diag_values = np.concatenate(
            [
                np.full(no1, prec_obs1),
                np.full(no2, prec_obs2),
                np.full(no3, prec_obs3),
            ]
        )
        print(
            "prec_obs1: ",
            prec_obs1,
            "prec_obs2 ",
            prec_obs2,
            "prec_obs3",
            prec_obs3,
        )

    # Construct the diagonal matrix
    diagonal_matrix = sp.sparse.diags(
        diag_values, offsets=0, shape=(no1 + no2 + no3, no1 + no2 + no3), format="csc"
    )

    # extend Qprior
    Qprior_ext = block_diag(
        (
            Q,
            sp.sparse.diags(
                np.full(num_vars, 1e-3),
                offsets=0,
                shape=(num_vars, num_vars),
                format="csc",
            ),
        )
    )

    # Construct the precision matrix
    Qconditional = Qprior_ext + A.T @ diagonal_matrix @ A
    print("Qconditional[:10, :10] = \n", Qconditional[:10, :10].toarray())

    # load comparison
    Qcond_file = f"{data_dir}/Qconditional_R_{dim_Q+nb}_{dim_Q+nb}.dat"
    Qcond = read_CSC(Qcond_file)

    print("Qcond[:10, :10] = \n", Qcond[:10, :10].toarray())

    # check if Q_upper and Qprior_R are equal
    diff = Qconditional - Qcond
    print("max(abs(Qconditional - Qcond)): ", np.max(np.abs(diff)))

    threshold = 1e-4

    # Find the indices where the absolute values of the elements in diff are greater than the threshold
    diff_abs = np.abs(diff)
    rows, cols = diff_abs.nonzero()
    values = diff_abs.data

    # Filter the indices based on the threshold
    filtered_indices = [(r, c) for r, c, v in zip(rows, cols, values) if v > threshold]

    # Print the filtered indices
    print("Indices where abs(diff) > 1e-4:", filtered_indices[:10])

    # save spy plot of Q to .png
    plt.figure()
    plt.spy(Qconditional)
    plt.savefig("Qconditional.png")
    plt.close()

    # deal with permutation if spatial temporal matrix
    if type == "spatio-temporal":

        # both work. TODO: how to do this efficiently
        permM = generate_block_permutation_matrix(nt, ns, num_vars)

        perm = generate_permutation_indices(nt, ns, num_vars)

        # permute Q
        Qprior_perm1 = Q[perm, :][:, perm]
        Qprior_perm = permM @ Q @ permM.T

        assert np.all(
            Qprior_perm1.toarray() == Qprior_perm.toarray()
        ), "Permutation matrices do not match!"

        # save spy plot of Q
        plt.figure()  # Start a new figure
        plt.spy(Qprior_perm, markersize=0.1)
        plt.savefig("Qprior_perm.png")
        plt.close()  # Close the figure to free memory
