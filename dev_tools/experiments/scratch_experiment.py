# inspect_model_predictions.py

import pandas as pd
import joblib

model = joblib.load("models/price_model_v1.joblib")

df = pd.read_csv("data/processed/vehicle_sales_features.csv")

feature_cols = joblib.load("models/feature_cols_v1.joblib")

sample = df[feature_cols].head(10)

preds = model.predict(sample)

print("Predictions:")
print(preds)

print("\nActual log_price:")
print(df["log_price"].head(10).values)