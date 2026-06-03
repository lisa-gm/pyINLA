# load .csv dataframe and save everything in DALIA suitable format

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def construct_site_data(
    df: pd.DataFrame,
    intercept_mode: str = "none",
    hospital_order: list | None = None,
    covariate_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # stress is the response variable y; covariates are either user-selected or all non-hospital columns
    y = df["stress"].to_numpy()
    if covariate_columns is None:
        covariate_columns = [
            col for col in df.columns if col not in {"stress", "hospital"}
        ]
    else:
        invalid_columns = [
            col
            for col in covariate_columns
            if col not in df.columns or col in {"stress", "hospital"}
        ]
        if invalid_columns:
            raise ValueError(
                "Invalid covariate columns: "
                f"{invalid_columns}. They must exist in the dataframe and cannot be 'stress' or 'hospital'."
            )

    if not covariate_columns:
        raise ValueError("No covariate columns selected.")

    x_cov = df[covariate_columns].to_numpy()

    if intercept_mode == "global":
        x_regression = np.hstack((np.ones((x_cov.shape[0], 1)), x_cov))
        x_generic = None
        print("Using intercept_mode='global': added one global intercept column")
    elif intercept_mode == "site_specific":
        x_site, hospitals = build_site_intercept_projection_matrix(
            df, hospital_order=hospital_order
        )
        x_regression = x_cov
        x_generic = x_site
        print(
            f"Using intercept_mode='site_specific': added {len(hospitals)} site-specific intercept columns"
        )
    elif intercept_mode == "none":
        x_regression = x_cov
        x_generic = None
        print("Using intercept_mode='none': no intercept column added")
    else:
        raise ValueError(
            f"Unknown intercept_mode: {intercept_mode}. Use 'global', 'site_specific', or 'none'."
        )

    return y, x_regression, x_generic


def build_site_intercept_projection_matrix(
    df: pd.DataFrame,
    hospital_column: str = "hospital",
    hospital_order: list | None = None,
) -> tuple[np.ndarray, list]:
    """Build a site-specific intercept projection matrix A.

    A[i, j] = 1 if observation y_i belongs to hospital j, otherwise 0.
    """
    if hospital_order is None:
        hospitals = sorted(df[hospital_column].unique().tolist())
    else:
        hospitals = list(hospital_order)
    hospital_to_col = {hospital: idx for idx, hospital in enumerate(hospitals)}

    A = np.zeros((df.shape[0], len(hospitals)), dtype=float)
    for row_idx, hospital in enumerate(df[hospital_column].tolist()):
        col_idx = hospital_to_col[hospital]
        A[row_idx, col_idx] = 1.0

    return A, hospitals


def save_dataset(
    base_dir: Path,
    y: np.ndarray,
    x_regression: np.ndarray,
    x_generic: np.ndarray | None,
    fit_mask: np.ndarray | None = None,
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    if fit_mask is None:
        y_fit = y
        x_regression_fit = x_regression
        x_regression_predict = None
        x_generic_fit = x_generic
        x_generic_predict = None
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape[0] != y.shape[0]:
            raise ValueError(
                "fit_mask length must match number of rows in y and design matrices."
            )
        predict_mask = ~fit_mask

        y_fit = y[fit_mask]
        x_regression_fit = x_regression[fit_mask, :]
        x_regression_predict = x_regression[predict_mask, :]

        if x_generic is not None:
            x_generic_fit = x_generic[fit_mask, :]
            x_generic_predict = x_generic[predict_mask, :]
        else:
            x_generic_fit = None
            x_generic_predict = None

    np.save(base_dir / "y.npy", y_fit)

    inputs_regression_dir = base_dir / "inputs_regression"
    inputs_regression_dir.mkdir(parents=True, exist_ok=True)
    np.save(inputs_regression_dir / "a.npy", x_regression_fit)
    if x_regression_predict is not None:
        np.save(inputs_regression_dir / "apr.npy", x_regression_predict)
        sp.save_npz(
            inputs_regression_dir / "apr.npz",
            sp.csr_matrix(x_regression_predict),
        )

    if x_generic_fit is not None:
        inputs_generic_dir = base_dir / "inputs_generic"
        inputs_generic_dir.mkdir(parents=True, exist_ok=True)
        np.save(inputs_generic_dir / "a.npy", x_generic_fit)
        if x_generic_predict is not None:
            np.save(inputs_generic_dir / "apr.npy", x_generic_predict)
            sp.save_npz(
                inputs_generic_dir / "apr.npz",
                sp.csr_matrix(x_generic_predict),
            )
        q_generic = sp.identity(x_generic_fit.shape[1], format="csr")
        sp.save_npz(inputs_generic_dir / "q.npz", q_generic)


### load .csv file
data_type = "nurses_hom"
family = "gaussian"
# Set to None to use all available covariates (except stress and hospital).
# Example for age-only: covariate_columns = ["age"]
covariate_columns: list[str] | None = ["age"]

current_dir = Path(__file__).resolve().parent

folder_path = current_dir / "data"
file_path = folder_path / f"data_{data_type}_{family}_with_fit_indicator.csv"
data = pd.read_csv(file_path)

first_column = data.columns[0]
data[first_column] = data[first_column].astype("string")

print(data.head())
print(
    f"Using covariates: {covariate_columns if covariate_columns is not None else 'all'}"
)

if "in_initial_fit" not in data.columns:
    raise ValueError("Expected column 'in_initial_fit' in input data.")

non_na_fit_values = set(
    data.loc[data["in_initial_fit"].notna(), "in_initial_fit"].unique()
)
if non_na_fit_values != {1.0}:
    raise ValueError(
        "Column 'in_initial_fit' must contain only 1 or NA values. "
        f"Found non-NA values: {sorted(non_na_fit_values)}"
    )

fit_mask = data["in_initial_fit"].to_numpy() == 1.0
print(
    f"Rows in initial fit (a.npy): {int(fit_mask.sum())}; "
    f"prediction rows (apr.npy): {int((~fit_mask).sum())}"
)

# 1) Joint datasets with explicit intercept type in folder name
joint_folder_global = current_dir / f"joint_{data_type}_{family}_global_intercept"
joint_folder_site_specific = (
    current_dir / f"joint_{data_type}_{family}_site_specific_intercept"
)
hospital_values = sorted(data["hospital"].unique().tolist())

y_joint_global, x_joint_global_regression, x_joint_global_generic = construct_site_data(
    data,
    intercept_mode="global",
    hospital_order=hospital_values,
    covariate_columns=covariate_columns,
)
save_dataset(
    joint_folder_global,
    y_joint_global,
    x_joint_global_regression,
    x_joint_global_generic,
    fit_mask=fit_mask,
)

print(
    f"Saved joint y to {joint_folder_global / 'y.npy'} with shape {y_joint_global.shape}"
)
print(
    f"Saved joint regression X to {joint_folder_global / 'inputs_regression' / 'a.npy'} with shape {x_joint_global_regression.shape}"
)

y_joint_site_specific, x_joint_regression, x_joint_generic = construct_site_data(
    data,
    intercept_mode="site_specific",
    hospital_order=hospital_values,
    covariate_columns=covariate_columns,
)
save_dataset(
    joint_folder_site_specific,
    y_joint_site_specific,
    x_joint_regression,
    x_joint_generic,
    fit_mask=fit_mask,
)

print(
    f"Saved joint y to {joint_folder_site_specific / 'y.npy'} with shape {y_joint_site_specific.shape}"
)
print(
    f"Saved joint regression X to {joint_folder_site_specific / 'inputs_regression' / 'a.npy'} with shape {x_joint_regression.shape}"
)
if x_joint_generic is not None:
    print(
        f"Saved joint generic X to {joint_folder_site_specific / 'inputs_generic' / 'a.npy'} with shape {x_joint_generic.shape}"
    )

# 2) Split datasets by hospital with explicit intercept type in folder name
split_root_global = current_dir / f"split_{data_type}_{family}_global_intercept"
split_root_site_specific = (
    current_dir / f"split_{data_type}_{family}_site_specific_intercept"
)

for idx, hospital in enumerate(hospital_values, start=1):
    row_positions = np.flatnonzero(data["hospital"].to_numpy() == hospital)

    y_h_global = y_joint_global[row_positions]
    x_h_global_regression = x_joint_global_regression[row_positions, :]
    fit_mask_h = fit_mask[row_positions]
    hospital_dir_global = split_root_global / f"hospital_{idx}"
    save_dataset(
        hospital_dir_global,
        y_h_global,
        x_h_global_regression,
        None,
        fit_mask=fit_mask_h,
    )
    print(
        f"Saved global hospital_{idx} ({hospital}) y to {hospital_dir_global / 'y.npy'} with shape {y_h_global.shape}"
    )
    print(
        f"Saved global hospital_{idx} ({hospital}) regression X to {hospital_dir_global / 'inputs_regression' / 'a.npy'} with shape {x_h_global_regression.shape}"
    )

    y_h_site = y_joint_site_specific[row_positions]
    x_h_site_regression = x_joint_regression[row_positions, :]
    x_h_site_generic = None
    if x_joint_generic is not None:
        x_h_site_generic = x_joint_generic[row_positions, :]
    hospital_dir_site = split_root_site_specific / f"hospital_{idx}"
    save_dataset(
        hospital_dir_site,
        y_h_site,
        x_h_site_regression,
        x_h_site_generic,
        fit_mask=fit_mask_h,
    )
    print(
        f"Saved site-specific hospital_{idx} ({hospital}) y to {hospital_dir_site / 'y.npy'} with shape {y_h_site.shape}"
    )
    print(
        f"Saved site-specific hospital_{idx} ({hospital}) regression X to {hospital_dir_site / 'inputs_regression' / 'a.npy'} with shape {x_h_site_regression.shape}"
    )
    if x_h_site_generic is not None:
        print(
            f"Saved site-specific hospital_{idx} ({hospital}) generic X to {hospital_dir_site / 'inputs_generic' / 'a.npy'} with shape {x_h_site_generic.shape}"
        )
