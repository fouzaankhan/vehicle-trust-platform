# inspect_training_target.py

import pandas as pd
from src.utils.config_loader import load_config

cfg = load_config()

print("CONFIG TARGET =", cfg["features"]["target"])

df = pd.read_csv("data/processed/vehicle_sales_features.csv")

target = cfg["features"]["target"]

print("\nTARGET COLUMN:", target)

print(df[target].describe())

print("\nFirst 10 values:")
print(df[target].head(10))