import pandas as pd
from src.utils.config_loader import load_config

cfg = load_config()

df = pd.read_csv("data/processed/vehicle_sales_features.csv")

target = cfg["features"]["target"]

print("Target mean:", df[target].mean())
print("Target min :", df[target].min())
print("Target max :", df[target].max())