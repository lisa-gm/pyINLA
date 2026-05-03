# load .csv dataframe and save everything in DALIA suitable format

import os
import numpy as np
import pandas as pd
from pathlib import Path


### load .csv file
data_type = "nurses_hom"
family = "gaussian"

folder_path = "/Users/lisa/icloud/uni/repositories/federated_learning/confeR/paper/data/summarized"
file_path = os.path.join(folder_path, f"data_{data_type}_{family}.csv")
data = pd.read_csv(file_path)

first_column = data.columns[0]
data[first_column] = data[first_column].astype("string")

print(data.head())

### ignore hospital column for now
### stress is the response variable, i.e. y, the other columns are the covariates, i.e. X
y = data["stress"].values
X = data.drop(columns=["stress", "hospital"]).values

## add all ones for the intercept
X = np.hstack((np.ones((X.shape[0], 1)), X))

# mkdir full dataset
folder_name = f"{data_type}_{family}"
current_dir = Path(__file__).resolve().parent
output_dir = current_dir / folder_name
sub_output_dir = output_dir / "inputs"
output_dir.mkdir(parents=True, exist_ok=True)
sub_output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "y.npy", y)
np.save(sub_output_dir / "a.npy", X)

print(f"Saved y to {output_dir / 'y.npy'} with shape {y.shape}")
print(f"Saved X to {sub_output_dir / 'a.npy'} with shape {X.shape}")
