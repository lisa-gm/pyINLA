## functions to construct the precision matrices
import time

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse import kron as spkron

# TODO: reference passing ... no copies ...
# TODO: make sure that matrices are deleted in time or better: more global scope -> reused


def Q_spatio_temporal(theta, c0, g1, g2, g3, M0, M1, M2):
    """
    Construct the precision matrix of the spatio-temporal component
    """
    # print("theta:", theta)
    # print("c0:\n", c0.toarray())
    # print("g1:\n", g1.toarray())
    # print("g2:\n", g2.toarray())
    # print("g3:\n", g3.toarray())
    # print("M0:\n", M0.toarray())
    # print("M1:\n", M1.toarray())
    # print("M2:\n", M2.toarray())

    exp_theta1 = np.exp(theta[0])
    exp_theta2 = np.exp(theta[1])
    exp_theta3 = np.exp(theta[2])

    q1s = pow(exp_theta2, 2) * c0 + g1
    # print("q1s:\n", q1s[:10,:10].toarray())
    q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2, 2) * g1 + g2
    # print("q2s:\n", q2s[:10,:10].toarray())
    q3s = (
        pow(exp_theta2, 6) * c0
        + 3 * pow(exp_theta2, 4) * g1
        + 3 * pow(exp_theta2, 2) * g2
        + g3
    )
    # print("q3s:\n", q3s[:10,:10].toarray())

    ## c++ kron eigen seems to take ~1 sec, slightly less

    # Compute their Kronecker product
    start_time = time.time()
    # naturally return is a sparse matrix in COO format, after addition CSR  #print(type(kronecker_product))
    kronecker_product = pow(exp_theta1, 2) * (
        spkron(M0, q3s)
        + exp_theta3 * spkron(M1, q2s)
        + pow(exp_theta3, 2) * spkron(M2, q1s)
    )
    end_time = time.time()
    print("SciPy: Time Kronecker product: {:.3f} seconds".format(end_time - start_time))
    # print("kronecker product: \n", kronecker_product[:10,:10].toarray())

    return kronecker_product


def construct_Q(nb, theta, Qst, AxTAx):
    if not isspmatrix_csr(Qst):
        Qst = csr_matrix(Qst)

    ########### VERSION 1 ###########
    # extend Qst by nb rows/cols
    ## TODO SLOW
    # t_start = time.time()
    # # TODO: check bmat better
    # Q = spblock_diag([Qst, Qb])
    # t_end = time.time()
    # print("time spblock_diag: ", t_end - t_start)
    # # TODO: SLOW ...
    # t_start = time.time()
    # Q += np.exp(theta[0]) * AxTAx
    # t_end = time.time()
    # print("time Qst + AxTAx: ", t_end - t_start)

    # Q = Q.tocsr()

    # print("Q.indices: ", Q.indices)
    # print("Q.indptr: ", Q.indptr)

    ########### VERSION 2 ###########
    # seems ~2x faster for large matrices
    # when additional spatial field ... update ...
    # Manually create data, indices, and indptr for Qb
    # -> assumes Qb to be diagonal!!
    # t_start = time.time()
    Qb_data = np.full(nb, 0.001)
    Qb_indices = np.arange(nb)
    Qb_indptr = np.arange(nb + 1)

    # Extract data, indices, and indptr from Qst
    Qst_data = Qst.data
    Qst_indices = Qst.indices
    Qst_indptr = Qst.indptr
    Qst_shape = Qst.shape

    # Concatenate data, indices, and indptr to form the block diagonal matrix Q
    data = np.concatenate([Qst_data, Qb_data])
    indices = np.concatenate([Qst_indices, Qb_indices + Qst_shape[1]])
    indptr = np.concatenate([Qst_indptr, Qst_indptr[-1] + Qb_indptr[1:]])

    # print("Q2.indices: ", indices)
    # print("Q2.indptr: ", indptr)

    # Create the block diagonal matrix Q
    Q = csr_matrix(
        (data, indices, indptr), shape=(Qst_shape[0] + nb, Qst_shape[1] + nb)
    )
    # t_end = time.time()
    # print("time construct Q: ", t_end - t_start)

    # print("norm(Q1.data - Q2.data): ", np.linalg.norm(Q.data - data))
    # print("norm(Q1.indices - Q2.indices): ", np.linalg.norm(Q.indices - indices))
    # print("norm(Q1.indptr - Q2.indptr): ", np.linalg.norm(Q.indptr - indptr))

    # this is actually slow now. check GPU version ...
    # Add the AxTAx term
    Q += np.exp(theta[0]) * AxTAx

    # print("norm(Q2 - Q): ", np.linalg.norm(Q2 - Q))
    # Q_dense = Q.toarray()
    # print("Q[-5:,-5:]:\n", Q_dense[-5:,-5:])

    return Q
