
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from dalia.backend.datastructure import DiagonalMatrix, DenseMatrix, BlockMatrix
from .hyperparameter import HyperparameterConfig, HyperparameterManager





def assemble_prior_precision_matrix(
    iid_dim : int,
    tau_iid : float, 
    tau_queen : float,
    prec_regression : float):
    
    Q_iid = DiagonalMatrix(tau_iid)






def assemble_iid_prior_precision_matrix(
    PATH : Path,
    ):
    iid_dim : int = np.load(PATH / "iid_dim.npy")    
    return DiagonalMatrix(diag = np.ones(iid_dim))

def assemble_queen_prior_precision_matrix(
    PATH : Path,
    ):
    Q_queen : np.ndarray = np.load(PATH / "Q_queen.npy")
    return DenseMatrix(Q_queen)

def assemble_regression_prior_precision_matrix(
    PATH : Path,
    ):
    return DiagonalMatrix(diag = np.ones(1))

def update_model_precision_matrix(
    Qp_model : BlockMatrix,
    hp : "Hyperparameters"
    ) -> BlockMatrix:
    # Update the block precision matrix with the new hyperparameters.
    hp_i = hp.to_dict(iter = -1)
    hp_im1 = hp.to_dict(iter = -2)
    Qp_model.blocks[0][0] *= (hp_i["tau_iid"] / hp_im1["tau_iid"])
    Qp_model.blocks[1][1] *= (hp_i["tau_queen"] / hp_im1["tau_queen"])
    Qp_model.blocks[2][2] *= (hp_i["prec_regression"] / hp_im1["prec_regression"])
    return Qp_model







def assemble_design_matrix():
    ...

def optimize():
    ...

if __name__ == "__main__":
    dataset_path : Path = ...

    # Initialize (assemble) iid, queen, and regession precision
    # matrices.
    Q_prior_iid : DiagonalMatrix = assemble_iid_prior_precision_matrix(dataset_path)
    Q_prior_queen : DenseMatrix = assemble_queen_prior_precision_matrix(dataset_path)
    Q_prior_regression : DiagonalMatrix = assemble_regression_prior_precision_matrix(dataset_path)

    # Assemble the block precision matrix.
    Qp_model : BlockMatrix = BlockMatrix(
        blocks = [[Q_prior_iid, None, None],
                  [None, Q_prior_queen, None],
                  [None, None, Q_prior_regression]]
    )

    # Define the hyperparameters for the model.
    #theta : Hyperparameters = Hyperparameters(
    #    tau_iid = np.load(dataset_path / "tau_iid.npy"),
    #    tau_queen = np.load(dataset_path / "tau_queen.npy"),
    #    prec_regression = np.load(dataset_path / "prec_regression.npy")
    #)

    hp : Hyperparameters = Hyperparameters(
        hyperparameters_dict = {
            "tau_iid" : np.load(dataset_path / "tau_iid.npy"),
            "tau_queen" : np.load(dataset_path / "tau_queen.npy"),
            "prec_regression" : np.load(dataset_path / "prec_regression.npy")
        }
    )

    # Update the model precision matrix with the hyperparameters.
    Qp_model = update_model_precision_matrix(Qp_model, hp)

