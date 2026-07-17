
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dalia.backend.datastructure import BlockMatrix, DenseMatrix, DiagonalMatrix

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
    # Initialize the iid precision matrix of given dimension
    iid_dim : int = np.load(PATH / "iid_dim.npy")    
    return DiagonalMatrix(diag = np.ones(iid_dim))

def assemble_queen_prior_precision_matrix(
    PATH : Path,
    ):
    # Directly load the precision matrix given by the dataset
    Q_queen : np.ndarray = np.load(PATH / "Q_queen.npy")
    return DenseMatrix(Q_queen)

def assemble_regression_prior_precision_matrix(
    PATH : Path,
    ):
    precision_regression : float = np.load(PATH / "prec_regression.npy")
    regression_dim : int = np.load(PATH / "regression_dim.npy")
    return DiagonalMatrix(diag = precision_regression * np.ones(regression_dim))

def update_model_precision_matrix(
    Qp_model : BlockMatrix,
    hp : HyperparameterManager
    ) -> BlockMatrix:
    # Update the block precision matrix with the new hyperparameters.
    # the scaling factors are computed as the ratio of the new hyperparameter value to the previous one.
    iid_scaling = hp.get_hyperparameter_value("tau_iid") / hp.get_previous_hyperparameter_value("tau_iid")
    Qp_model.blocks[0][0] *= (hp_i["tau_iid"] / hp_im1["tau_iid"])
    Qp_model.blocks[1][1] *= (hp_i["tau_queen"] / hp_im1["tau_queen"])
    Qp_model.blocks[2][2] *= (hp_i["prec_regression"] / hp_im1["prec_regression"])
    return Qp_model



def objective():
    ...





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

    # Initialize the hyperparameter manager with the hyperparameter configurations.
    configs = [
        HyperparameterConfig("tau_iid", (float)np.load(dataset_path / "tau_iid.npy")),
        HyperparameterConfig("tau_queen", (float)np.load(dataset_path / "tau_queen.npy")),
    ]
    hp = HyperparameterManager(configs, track_history=True)

    # Update the model precision matrix with the hyperparameters.
    Qp_model = update_model_precision_matrix(Qp_model, hp)

