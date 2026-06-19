# inspect_features.py

import pandas as pd

df = pd.read_csv("data/processed/vehicle_sales_features.csv")

cols = [
    "make_median_price",
    "model_median_price"
]

for c in cols:
    if c in df.columns:
        print("\n", c)
        print(df[c].describe())