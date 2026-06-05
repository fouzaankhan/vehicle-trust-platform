import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


class FeatureEngineer:
    """
    Builds ML-ready features from cleaned vehicle listing data.
    All transformations are deterministic and logged.
    """

    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(__name__)
        self.make_median_map = {}
        self.model_median_map = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use this on TRAINING data.
        Learns median maps from data, then transforms.
        """
        self.logger.info(f"fit_transform started. Input shape: {df.shape}")
        self._learn_median_maps(df)
        df = self._build_all_features(df)
        self.logger.info(f"fit_transform complete. Output shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use this on NEW/TEST data.
        Uses already-learned median maps, does not relearn.
        """
        self.logger.info(f"transform started. Input shape: {df.shape}")
        df = self._build_all_features(df)
        self.logger.info(f"transform complete. Output shape: {df.shape}")
        return df

    # ------------------------------------------------------------------ #
    # Learn median prices per make and model from training data
    # This is target encoding — must be learned only on train set
    # ------------------------------------------------------------------ #
    def _learn_median_maps(self, df: pd.DataFrame):
        self.make_median_map = df.groupby("make")["price"].median().to_dict()
        self.model_median_map = df.groupby("model")["price"].median().to_dict()
        self.logger.info(f"Learned median maps: {len(self.make_median_map)} makes, {len(self.model_median_map)} models.")

    # ------------------------------------------------------------------ #
    # Build all features in sequence
    # ------------------------------------------------------------------ #
    def _build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._age_features(df)
        df = self._usage_features(df)
        df = self._market_features(df)
        df = self._condition_features(df)
        df = self._log_transforms(df)
        df = self._encode_categoricals(df)
        return df

    def _age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        sale_year = df["sale_year"].fillna(2015)  # fallback for missing sale_year
        df["vehicle_age"] = (sale_year - df["year"]).clip(lower=0)
        df["age_km_ratio"] = df["km_driven"] / (df["vehicle_age"] + 1)
        self.logger.info("Built: vehicle_age, age_km_ratio")
        return df

    def _usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price_per_km"] = df["price"] / (df["km_driven"] + 1)
        df["km_per_year"] = df["km_driven"] / (df["vehicle_age"] + 1)
        self.logger.info("Built: price_per_km, km_per_year")
        return df

    def _market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # MMR deviation — how far is listing price from market reference?
        # MMR = Manheim Market Report, a real industry benchmark
        df["mmr_deviation"] = (df["price"] - df["mmr"]) / (df["mmr"] + 1)
        df["mmr_deviation"] = df["mmr_deviation"].clip(-1, 1)  # cap extreme outliers

        # Target encoding — encode brand and model as their median price
        global_median = df["price"].median()
        df["make_median_price"] = df["make"].map(self.make_median_map).fillna(global_median)
        df["model_median_price"] = df["model"].map(self.model_median_map).fillna(global_median)

        # How does this listing compare to its make's typical price?
        df["price_vs_make_median"] = df["price"] / (df["make_median_price"] + 1)

        self.logger.info("Built: mmr_deviation, make_median_price, model_median_price, price_vs_make_median")
        return df

    def _condition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # condition column is 1–49 scale in this dataset
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce")
        df["condition"] = df["condition"].fillna(df["condition"].median())

        # Bucket into bands
        df["condition_band"] = pd.cut(
            df["condition"],
            bins=[0, 20, 35, 49],
            labels=["low", "mid", "high"]
        )
        self.logger.info("Built: condition (cleaned), condition_band")
        return df

    def _log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        # Log transforms fix right-skewed distributions for tree models
        df["log_km"] = np.log1p(df["km_driven"])
        df["log_price"] = np.log1p(df["price"])  # this becomes our training TARGET
        self.logger.info("Built: log_km, log_price")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transmission: already lowercased in cleaning step
        transmission_map = {"automatic": 0, "manual": 1}
        df["transmission_encoded"] = df["transmission"].map(transmission_map).fillna(0)

        # Condition band
        condition_map = {"low": 0, "mid": 1, "high": 2}
        df["condition_band_encoded"] = df["condition_band"].map(condition_map).fillna(1)

        self.logger.info("Built: transmission_encoded, condition_band_encoded")
        return df


# ------------------------------------------------------------------ #
# Run directly to build and save feature-engineered dataset
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    input_path = "data/processed/vehicle_sales_clean.csv"
    output_path = "data/processed/vehicle_sales_features.csv"

    df = pd.read_csv(input_path)
    engineer = FeatureEngineer()
    df_features = engineer.fit_transform(df)

    df_features.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Shape: {df_features.shape}")
    print(f"\nNew columns added:")
    original_cols = pd.read_csv(input_path, nrows=0).columns.tolist()
    new_cols = [c for c in df_features.columns if c not in original_cols]
    print(new_cols)