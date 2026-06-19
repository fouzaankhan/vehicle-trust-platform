import pandas as pd

df = pd.read_csv("data/processed/vehicle_sales_features.csv")

print(df["price"].describe())
print(df["price"].head())