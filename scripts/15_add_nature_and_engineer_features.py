import os
import pandas as pd
import numpy as np

base_dir = os.path.expanduser("~/CalibraNeo_Project")
proc_dir = os.path.join(base_dir, "data_processed")
master_path = os.path.join(proc_dir, "MASTER_CALIBRANEO_DATASET.csv")

print("=== 1. INSPECTING THE 2,317 NATURE CLINICAL VACCINE ROWS (nature_vaccine_S1.csv) ===")
nat_s1_path = os.path.join(proc_dir, "nature_vaccine_S1.csv")
df_nat = pd.read_csv(nat_s1_path, skiprows=1) # Row 0 was the table title!
print(f"Nature S1 Shape: {df_nat.shape[0]} rows x {df_nat.shape[1]} cols")
print("Real Column Headers in Nature S1:")
print(list(df_nat.columns))
print("\nFirst 2 rows of Nature S1:")
print(df_nat.head(2).to_string())
