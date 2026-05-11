# load .csv dataframe and save everything in DALIA suitable format

import os
from pathlib import Path

import numpy as np
import pandas as pd


def build_design_and_response(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # stress is the response variable y, all remaining non-hospital columns are covariates X
    y = df["stress"].to_numpy()
    x_cov = df.drop(columns=["stress", "hospital"]).to_numpy()
    x = np.hstack((np.ones((x_cov.shape[0], 1)), x_cov))
    return y, x


def save_dataset(base_dir: Path, y: np.ndarray, x: np.ndarray) -> None:
    inputs_dir = base_dir / "inputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    np.save(base_dir / "y.npy", y)
    np.save(inputs_dir / "a.npy", x)


### load .csv file
data_type = "Nurses"
family = "gaussian"

folder_path = "/Users/lisa/icloud/uni/repositories/federated_learning/confeR/paper/data/summarized"
file_path = os.path.join(folder_path, f"data_{data_type}_{family}.csv")
data = pd.read_csv(file_path)

first_column = data.columns[0]
data[first_column] = data[first_column].astype("string")

print(data.head())

current_dir = Path(__file__).resolve().parent

# 1) Joint dataset (renamed): joint_nurses_hom_gaussian
joint_folder = current_dir / f"joint_{data_type}_{family}"
y_joint, x_joint = build_design_and_response(data)
save_dataset(joint_folder, y_joint, x_joint)
print(f"Saved joint y to {joint_folder / 'y.npy'} with shape {y_joint.shape}")
print(
    f"Saved joint X to {joint_folder / 'inputs' / 'a.npy'} with shape {x_joint.shape}"
)

# 2) Split dataset by hospital: split_nurses_hom_gaussian/hospital_*/
split_root = current_dir / f"split_{data_type}_{family}"
hospital_values = list(data["hospital"].dropna().unique())
hospital_values.sort()

for idx, hospital in enumerate(hospital_values, start=1):
    hospital_df = data[data["hospital"] == hospital].copy()
    y_h, x_h = build_design_and_response(hospital_df)
    hospital_dir = split_root / f"hospital_{idx}"
    save_dataset(hospital_dir, y_h, x_h)
    print(
        f"Saved hospital_{idx} ({hospital}) y to {hospital_dir / 'y.npy'} with shape {y_h.shape}"
    )
    print(
        f"Saved hospital_{idx} ({hospital}) X to {hospital_dir / 'inputs' / 'a.npy'} with shape {x_h.shape}"
    )
