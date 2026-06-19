# test_target_values.py
import pandas as pd

df = pd.read_csv("data/processed/vehicle_sales_features.csv")

print(df["price"].head())
print(df["price"].describe())

if "log_price" in df.columns:
    print("\nLOG PRICE:")
    print(df["log_price"].head())
    print(df["log_price"].describe())