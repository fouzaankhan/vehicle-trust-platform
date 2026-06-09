import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


class PricePredictor:
    """
    Loads trained model and feature cols.
    Accepts a raw listing dict, engineers features, returns prediction.
    """

    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(__name__)
        self.model = joblib.load("models/price_model_v1.joblib")
        self.feature_cols = joblib.load("models/feature_cols_v1.joblib")
        self.logger.info("PricePredictor loaded successfully.")

    def predict(self, listing: dict) -> dict:
        features = self._engineer_features(listing)
        feature_df = pd.DataFrame([features])[self.feature_cols]
        log_price_pred = self.model.predict(feature_df)[0]
        price_pred = float(np.expm1(log_price_pred))

        return {
            "predicted_price": round(price_pred, 2),
            "features_used": features
        }

    def _engineer_features(self, listing: dict) -> dict:
        current_year = 2024
        year = listing["year"]
        km_driven = listing["km_driven"]
        condition = listing.get("condition", 25)  # default mid condition

        vehicle_age = max(current_year - year, 0)
        log_km = float(np.log1p(km_driven))
        age_km_ratio = km_driven / (vehicle_age + 1)
        km_per_year = km_driven / (vehicle_age + 1)

        transmission_map = {"automatic": 0, "manual": 1}
        transmission_encoded = transmission_map.get(
            listing.get("transmission", "automatic").lower(), 0
        )

        if condition <= 20:
            condition_band_encoded = 0
        elif condition <= 35:
            condition_band_encoded = 1
        else:
            condition_band_encoded = 2

        # Median prices — loaded from saved training data stats
        make_median = self._get_make_median(listing.get("make", ""))
        model_median = self._get_model_median(listing.get("model", ""))

        return {
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "log_km": log_km,
            "condition": condition,
            "age_km_ratio": age_km_ratio,
            "km_per_year": km_per_year,
            "make_median_price": make_median,
            "model_median_price": model_median,
            "sale_month": listing.get("sale_month", 6),
            "transmission_encoded": transmission_encoded,
            "condition_band_encoded": condition_band_encoded
        }

    def _get_make_median(self, make: str) -> float:
        # Load precomputed medians from training data
        try:
            medians = joblib.load("models/make_medians.joblib")
            return float(medians.get(make.title(), medians.get("Ford", 13000)))
        except Exception:
            return 13000.0  # global fallback

    def _get_model_median(self, model_name: str) -> float:
        try:
            medians = joblib.load("models/model_medians.joblib")
            return float(medians.get(model_name.lower(), 13000))
        except Exception:
            return 13000.0