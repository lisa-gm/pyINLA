# import math
import os

# import matplotlib.pyplot as plt
import numpy as np
from matrix_utilities import (  # read_sym_CSC,
    load_matrices_spatial_model_from_dat,
    load_matrices_spatial_temporal_model_from_dat,
    read_CSC,
)
from scipy.sparse import csc_matrix, save_npz  # block_diag,

# from pyinla import sp, xp

if __name__ == "__main__":
    # get current path
    path = os.path.dirname(__file__)

    num_vars = 3

    type = "spatio-temporal"  # "spatial" #

    ns = 1673
    nt = 192

    # add more shared fixed effects later
    nb = num_vars
    nb_per_var = 1

    no1 = 3 * ns * nt
    no2 = 3 * ns * nt
    no3 = 3 * ns * nt  # 0

    dim_theta = 15

    no_list = [no1, no2, no3]
    total_obs = sum(no_list)

    n = num_vars * (ns * nt + nb_per_var)

    data_dir = (
        f"../../../coregionalization_models/data/nv{num_vars}_ns{ns}_nt{nt}_nb{nb}"
    )

    # load submatrices
    if type == "spatial":
        c0, g1, g2 = load_matrices_spatial_model_from_dat(ns, data_dir)

    elif type == "spatio-temporal":
        c0, g1, g2, g3, M0, M1, M2 = load_matrices_spatial_temporal_model_from_dat(
            ns, nt, data_dir
        )
    else:
        raise ValueError("Invalid model type")

    # load observation vectors
    y1_file = f"{data_dir}/y1_{no1}_1.dat"
    y1 = np.loadtxt(y1_file)

    y2_file = f"{data_dir}/y2_{no2}_1.dat"
    y2 = np.loadtxt(y2_file)

    # load projection matrices
    a1_file = f"{data_dir}/A1_{no1}_{ns*nt}.dat"
    a1 = read_CSC(a1_file)
    print("nnz(A1) = ", a1.nnz)

    # split a into random and fixed effects
    a1_random = a1[:, : ns * nt]
    a1_fixed = csc_matrix(np.ones((no_list[0], nb_per_var)))

    a2_file = f"{data_dir}/A2_{no1}_{ns*nt}.dat"
    a2 = read_CSC(a2_file)
    print("nnz(A2) = ", a1.nnz)

    a2_random = a2[:, : ns * nt]
    a2_fixed = csc_matrix(np.ones((no_list[1], nb_per_var)))

    if num_vars == 3:
        y3_file = f"{data_dir}/y3_{no3}_1.dat"
        y3 = np.loadtxt(y3_file)

        a3_file = f"{data_dir}/A3_{no3}_{ns*nt}.dat"
        a3 = read_CSC(a3_file)

        a3_random = a3[:, : ns * nt]
        a3_fixed = csc_matrix(np.ones((no_list[2], nb_per_var)))

    # set path new data directory and create necessary folders
    new_data_dir = f"{path}/inputs_nv{num_vars}_ns{ns}_nt{nt}_nb{nb}"
    os.makedirs(new_data_dir, exist_ok=True)
    print(f"Created directory {new_data_dir}")

    # save matrices
    if type == "spatial":
        for i in range(num_vars):
            os.makedirs(f"{new_data_dir}/model_{i+1}/inputs_spatial", exist_ok=True)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatial/c0.npz", c0)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatial/g1.npz", g1)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatial/g2.npz", g2)

            os.makedirs(f"{new_data_dir}/model_{i+1}/inputs_regression", exist_ok=True)

        save_npz(f"{new_data_dir}/model_1/inputs_spatial/a.npz", a1_random)
        save_npz(f"{new_data_dir}/model_2/inputs_spatial/a.npz", a2_random)

        save_npz(f"{new_data_dir}/model_1/inputs_regression/a.npz", a1_fixed)
        save_npz(f"{new_data_dir}/model_2/inputs_regression/a.npz", a2_fixed)

        if num_vars == 3:
            save_npz(f"{new_data_dir}/model_3/inputs_spatial/a.npz", a3_random)
            save_npz(f"{new_data_dir}/model_3/inputs_regression/a.npz", a3_fixed)

    elif type == "spatio-temporal":
        for i in range(num_vars):
            os.makedirs(
                f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal", exist_ok=True
            )
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal/c0.npz", c0)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal/g1.npz", g1)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal/g2.npz", g2)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal/g3.npz", g3)

            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal/m0.npz", M0)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal/m1.npz", M1)
            save_npz(f"{new_data_dir}/model_{i+1}/inputs_spatio_temporal/m2.npz", M2)

            os.makedirs(f"{new_data_dir}/model_{i+1}/inputs_regression/", exist_ok=True)

        save_npz(f"{new_data_dir}/model_1/inputs_spatio_temporal/a.npz", a1_random)
        save_npz(f"{new_data_dir}/model_2/inputs_spatio_temporal/a.npz", a2_random)

        save_npz(f"{new_data_dir}/model_1/inputs_regression/a.npz", a1_fixed)
        save_npz(f"{new_data_dir}/model_2/inputs_regression/a.npz", a2_fixed)

        if num_vars == 3:
            save_npz(f"{new_data_dir}/model_3/inputs_spatio_temporal/a.npz", a3_random)
            save_npz(f"{new_data_dir}/model_3/inputs_regression/a.npz", a3_fixed)

    np.save(f"{new_data_dir}/model_1/y.npy", y1)
    np.save(f"{new_data_dir}/model_2/y.npy", y2)

    if num_vars == 3:
        np.save(f"{new_data_dir}/model_3/y.npy", y3)

    # reference output
    theta_ref_file = f"{data_dir}/theta_interpretS_original_{dim_theta}_1.dat"
    theta_ref = np.loadtxt(theta_ref_file)
    print(f"Reference theta: {theta_ref}")

    theta_ref_py_file = (
        f"{data_dir}/theta_interpretS_original_pyINLA_perm_{dim_theta}_1.dat"
    )
    theta_ref_py = np.loadtxt(theta_ref_py_file)
    print(f"Reference theta (python order): {theta_ref_py}")

    # load reference x
    x_ref_file = f"{data_dir}/x_original_{n}_1.dat"
    x_ref = np.loadtxt(x_ref_file)
    print("x_ref[:10]: ", x_ref[:10])

    os.makedirs(f"{new_data_dir}/reference_outputs", exist_ok=True)

    # np.save(f"{new_data_dir}/reference_outputs/theta_interpretS_original_pyINLA_perm_{dim_theta}_1.npy", theta_ref_py)
    np.savetxt(
        f"{new_data_dir}/reference_outputs/theta_interpretS_original_pyINLA_perm_{dim_theta}_1.dat",
        theta_ref_py,
    )

    # save reference x
    np.savetxt(f"{new_data_dir}/reference_outputs/x_original_{n}_1.dat", x_ref)

    print("num vars: ", num_vars)
    print("type: ", type)

    # # reorder theta to python format
    # if type == "spatial" and num_vars == 3:
    #     # old order ['sigma_1' , 'r_s1', 'sigma_2', 'r_s2', 'sigma_3', 'r_s3', 'lambda_0_1', 'lambda_0_2', 'lambda_1_2' 'prec_o1', 'prec_o2', 'prec_o3']
    #     # new order  ['r_s1', 'prec_o1', 'r_s2', 'prec_o2', 'r_s3', 'prec_o3', 'sigma_1', 'sigma_2', 'sigma_3', 'lambda_0_1', 'lambda_0_2', 'lambda_1_2']
    #     theta_py = np.array([
    #         theta_ref[1],  # r_s1
    #         theta_ref[9],  # prec_o1
    #         theta_ref[3],  # r_s2
    #         theta_ref[10], # prec_o2
    #         theta_ref[5],  # r_s3
    #         theta_ref[11], # prec_o3
    #         theta_ref[0],  # sigma_1 (was sigma_0 in the previous mapping)
    #         theta_ref[2],  # sigma_2 (was sigma_1 in the previous mapping)
    #         theta_ref[4],  # sigma_3
    #         theta_ref[6],  # lambda_0_1
    #         theta_ref[7],  # lambda_0_2
    #         theta_ref[8],  # lambda_1_2
    #     ])

    # elif num_vars == 2 and type == "spatio-temporal":
    #     # old order ['sigma_1' , 'r_s1', 'r_t1', 'sigma_2', 'r_s2', 'r_t2', 'prec_o1', 'prec_o2', 'lambda_0_1']
    #     # new order  ['r_s1', 'r_t1', 'prec_o1', 'r_s2', 'r_t2', 'prec_o2', 'sigma_1', 'sigma_2', 'lambda_0_1']
    #     theta_py = np.array([
    #         theta_ref[1],  # r_s1
    #         theta_ref[2],  # r_t1
    #         theta_ref[6],  # prec_o1
    #         theta_ref[4],  # r_s2
    #         theta_ref[5],  # r_t2
    #         theta_ref[7],  # prec_o2
    #         theta_ref[0],  # sigma_1
    #         theta_ref[3],  # sigma_2
    #         theta_ref[8],  # lambda_0_1
    #     ])

    # else:
    #     raise ValueError("Invalid model type")

    # print(f"theta R: {theta_ref}")
    # print(f"Reordered theta: {theta_py}")

    # save reordered theta
    # theta_py_file = f"{new_data_dir}/reference_outputs/theta_interpretS_original_pyINLA_perm_{len(theta_py)}_1.dat"
    # np.savetxt(theta_py_file, theta_py)
    # print(f"Saved reordered theta to {theta_py_file}")
