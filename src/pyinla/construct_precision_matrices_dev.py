## functions to construct the precision matrices
import time

import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cpcsr_matrix
from cupyx.scipy.sparse import eye as cpeye
from cupyx.scipy.sparse import kron as cpkron

# TODO: reference passing ... no copies ...
# TODO: make sure that matrices are deleted in time or better: more global scope -> reused


def Q_spatio_temporal_dev(
    theta_dev, c0_dev, g1_dev, g2_dev, g3_dev, M0_dev, M1_dev, M2_dev
):
    ## allocate constants on gpu
    exp_theta1_dev = cp.exp(theta_dev[0])
    exp_theta2_dev = cp.exp(theta_dev[1])
    exp_theta3_dev = cp.exp(theta_dev[2])

    q1s_dev = pow(exp_theta2_dev, 2) * c0_dev + g1_dev
    # print("q1s:\n", q1s[:10,:10].toarray())
    q2s_dev = (
        pow(exp_theta2_dev, 4) * c0_dev + 2 * pow(exp_theta2_dev, 2) * g1_dev + g2_dev
    )
    # print("q2s:\n", q2s[:10,:10].toarray())
    q3s_dev = (
        pow(exp_theta2_dev, 6) * c0_dev
        + 3 * pow(exp_theta2_dev, 4) * g1_dev
        + 3 * pow(exp_theta2_dev, 2) * g2_dev
        + g3_dev
    )
    # print("q3s:\n", q3s[:10,:10].toarray())

    # Compute their Kronecker product on GPU
    start_time = time.time()
    # naturally return is a sparse matrix in COO format, after addition CSR #print(type(kronecker_product_gpu))

    kronecker_product_dev = pow(exp_theta1_dev, 2) * (
        cpkron(M0_dev, q3s_dev)
        + exp_theta3_dev * cpkron(M1_dev, q2s_dev)
        + pow(exp_theta3_dev, 2) * cpkron(M2_dev, q1s_dev)
    )
    end_time = time.time()
    print("CuPy:  Time Kronecker product: {:.3f} seconds".format(end_time - start_time))

    # kronecker_product_csr_from_gpu = kronecker_product_gpu.get()
    # print("kronecker_product_csr_from_gpu: \n", kronecker_product_csr_from_gpu[:10,:10].toarray())
    # print("norm(kronecker_product - kronecker_product_csr_from_gpu): ", np.linalg.norm(kronecker_product - kronecker_product_csr_from_gpu))

    # nnz = kronecker_product_gpu.nnz
    # print("Number of nonzeros: ", nnz)

    Qst_gpu_memory = (
        kronecker_product_dev.data.nbytes
        + kronecker_product_dev.indices.nbytes
        + kronecker_product_dev.indptr.nbytes
    )
    # total_gpu_memory = kronecker_product_dev.data.nbytes + kronecker_product_dev.row.nbytes + kronecker_product_dev.col.nbytes

    print(f"Qst    allocated GPU memory: {Qst_gpu_memory / (1024**2):.2f} MB")

    return kronecker_product_dev


def construct_Q_dev(nb, theta_dev, Qst_dev, AxTAx_dev):
    # Qst_dev = cpcsr_matrix(Qst_dev)
    # Qst = Qst[1:10,1:10]

    Qb_dev = 0.001 * cpeye(nb)
    Qb_dev = Qb_dev.tocsr()
    # print("Qb\n", Qb.toarray())

    ########### VERSION 1 ###########
    # print("\nVersion 1")
    # # extend Qst by nb rows/cols
    # ## TODO SLOW
    # t_start = time.time()
    # # DOESNT EXIST: Q_dev = cpblock_diag([Qst_dev, Qb_dev]), instead:
    # Q_dev = cpbmat([[Qst_dev, None], [None, Qb_dev]])
    # t_end = time.time()
    # print("time Qst: {:.3f} seconds".format(t_end - t_start))
    # # TODO: SLOW ...
    # t_start = time.time()
    # Q_dev += cp.exp(theta_dev[0]) * AxTAx_dev
    # t_end = time.time()
    # print("time Qst + AxTAx: {:.3f} seconds".format(t_end - t_start))

    # Q_dev = Q_dev.tocsr()

    # print("Q.indices: ", Q.indices)
    # print("Q.indptr: ", Q.indptr)

    ########### VERSION 2 ###########
    # seems ~2x faster for large matrices
    # when additional spatial field ... update ...
    # Manually create data, indices, and indptr for Qb
    # -> assumes Qb to be diagonal!!
    # t_start = time.time()
    Qb_data_dev = cp.full(nb, cp.float64(0.001))
    Qb_indices_dev = cp.arange(nb)
    Qb_indptr_dev = cp.arange(nb + 1)

    # Extract data, indices, and indptr from Qst
    Qst_data_dev = Qst_dev.data
    Qst_indices_dev = Qst_dev.indices
    Qst_indptr_dev = Qst_dev.indptr
    Qst_shape = Qst_dev.shape

    # print("Qst_shape: ", Qst_shape[1])

    # Concatenate data, indices, and indptr to form the block diagonal matrix Q
    data_dev = cp.concatenate([Qst_data_dev, Qb_data_dev])
    indices_dev = cp.concatenate([Qst_indices_dev, Qb_indices_dev + Qst_shape[1]])
    indptr_dev = cp.concatenate(
        [Qst_indptr_dev, Qst_indptr_dev[-1] + Qb_indptr_dev[1:]]
    )

    # print("Q2.indices: ", indices)
    # print("Q2.indptr: ", indptr)
    # print("\nVersion 2")
    # Create the block diagonal matrix Q
    Q_dev = cpcsr_matrix(
        (data_dev, indices_dev, indptr_dev),
        shape=(Qst_shape[0] + nb, Qst_shape[1] + nb),
    )
    # t_end = time.time()
    # print("time construct Qst2: {:.3f} seconds".format(t_end - t_start))
    # t_start = time.time()
    Q_dev += cp.exp(theta_dev[0]) * AxTAx_dev
    # t_end = time.time()
    # print("time Qst2 + AxTAx: {:.3f} seconds\n".format(t_end - t_start))

    # TODO: double check that this is correct
    # Q_dev_gpu_memory = Q_dev.data.nbytes + Q_dev.indices.nbytes + Q_dev.indptr.nbytes
    # print(f"Q    allocated GPU memory: {Q_dev_gpu_memory / (1024**2):.2f} MB")

    # print("norm(Q1.data - Q2.data): ", cp.linalg.norm(Q_dev.data - Q2_dev.data))
    # print("norm(Q1.indices - Q2.indices): ", cp.linalg.norm(Q_dev.indices - Q2_dev.indices))
    # print("norm(Q1.indptr - Q2.indptr): ", cp.linalg.norm(Q_dev.indptr - Q2_dev.indptr))

    # problem for large matrices
    # print("norm(Q2 - Q): ", cp.linalg.norm(Q2_dev - Q_dev))

    # Q_dense = Q.toarray()
    # print("Q[-5:,-5:]:\n", Q_dense[-5:,-5:])

    return Q_dev
