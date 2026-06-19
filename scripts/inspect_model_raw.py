# inspect_model_raw.py

import pandas as pd
import joblib

model = joblib.load("models/price_model_v1.joblib")

df = pd.read_csv("data/processed/vehicle_sales_features.csv")

features = joblib.load("models/feature_cols_v1.joblib")

X = df[features].head(1000)

preds = model.predict(X)

print("Mean prediction:", preds.mean())
print("Min prediction :", preds.min())
print("Max prediction :", preds.max())

print("\nFirst 10 predictions:")
print(preds[:10])