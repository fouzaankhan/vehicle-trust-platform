import pandas as pd
import numpy as np
import yaml
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


class VehicleDataCleaner:
    """
    Production-style cleaning pipeline for vehicle listing data.
    Each step logs rows before/after so we have full traceability.
    """

    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(__name__)
        self.report = {}  # tracks what each step drops

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Starting cleaning pipeline. Input shape: {df.shape}")

        df = self._rename_columns(df)
        df = self._drop_missing_targets(df)
        df = self._filter_price_range(df)
        df = self._filter_year_range(df)
        df = self._filter_odometer(df)
        df = self._clean_transmission(df)
        df = self._clean_make_model(df)
        df = self._parse_sale_date(df)
        df = self._remove_duplicates(df)
        df = self._reset_index(df)

        self.logger.info(f"Cleaning complete. Output shape: {df.shape}")
        self._print_report()
        return df

    # ------------------------------------------------------------------ #
    # STEP A — Standardize column names
    # ------------------------------------------------------------------ #
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "sellingprice": "price",
            "odometer": "km_driven",
            "saledate": "sale_date"
        }
        df = df.rename(columns=rename_map)
        self.logger.info("Columns renamed.")
        return df

    # ------------------------------------------------------------------ #
    # STEP B — Drop rows where target (price) is missing
    # You flagged this — missing targets cannot be imputed, must be dropped
    # ------------------------------------------------------------------ #
    def _drop_missing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["price"])
        dropped = before - len(df)
        self.report["missing_price_dropped"] = dropped
        self.logger.info(f"Dropped {dropped} rows with missing price.")
        return df

    # ------------------------------------------------------------------ #
    # STEP C — Remove unrealistic prices
    # You flagged prices of 1 — these are data errors, not real sales
    # ------------------------------------------------------------------ #
    def _filter_price_range(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        min_p = self.config["data"]["min_price"]
        max_p = self.config["data"]["max_price"]
        df = df[(df["price"] >= min_p) & (df["price"] <= max_p)]
        dropped = before - len(df)
        self.report["price_out_of_range_dropped"] = dropped
        self.logger.info(f"Dropped {dropped} rows outside price range [{min_p}, {max_p}].")
        return df

    # ------------------------------------------------------------------ #
    # STEP D — Filter unrealistic years
    # ------------------------------------------------------------------ #
    def _filter_year_range(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        min_y = self.config["data"]["min_year"]
        max_y = self.config["data"]["max_year"]
        df = df[(df["year"] >= min_y) & (df["year"] <= max_y)]
        dropped = before - len(df)
        self.report["year_out_of_range_dropped"] = dropped
        self.logger.info(f"Dropped {dropped} rows outside year range [{min_y}, {max_y}].")
        return df

    # ------------------------------------------------------------------ #
    # STEP E — Filter unrealistic odometer readings
    # ------------------------------------------------------------------ #
    def _filter_odometer(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        max_km = self.config["data"]["max_km_driven"]
        df = df[df["km_driven"] <= max_km]
        df = df[df["km_driven"] >= 0]
        dropped = before - len(df)
        self.report["odometer_out_of_range_dropped"] = dropped
        self.logger.info(f"Dropped {dropped} rows with invalid odometer.")
        return df

    # ------------------------------------------------------------------ #
    # STEP F — Clean transmission
    # You flagged 65K+ missing — we impute with mode, don't drop
    # ------------------------------------------------------------------ #
    def _clean_transmission(self, df: pd.DataFrame) -> pd.DataFrame:
        before_missing = df["transmission"].isnull().sum()
        mode_val = df["transmission"].mode()[0]
        df["transmission"] = df["transmission"].fillna(mode_val)
        df["transmission"] = df["transmission"].str.strip().str.lower()
        self.report["transmission_imputed"] = int(before_missing)
        self.logger.info(f"Imputed {before_missing} missing transmission values with '{mode_val}'.")
        return df

    # ------------------------------------------------------------------ #
    # STEP G — Clean make and model
    # Drop rows where both are missing — unusable without these
    # ------------------------------------------------------------------ #
    def _clean_make_model(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["make", "model"])
        df["make"] = df["make"].str.strip().str.title()
        df["model"] = df["model"].str.strip().str.lower()
        dropped = before - len(df)
        self.report["missing_make_model_dropped"] = dropped
        self.logger.info(f"Dropped {dropped} rows with missing make/model.")
        return df

    # ------------------------------------------------------------------ #
    # STEP H — Parse sale_date
    # You flagged this — stored as string, needs datetime conversion
    # ------------------------------------------------------------------ #
    def _parse_sale_date(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["sale_date"] = pd.to_datetime(df["sale_date"], utc=True, errors="coerce")
            df["sale_year"] = df["sale_date"].dt.year
            df["sale_month"] = df["sale_date"].dt.month
            self.logger.info("sale_date parsed into datetime. Extracted sale_year and sale_month.")
        except Exception as e:
            self.logger.warning(f"Could not parse sale_date: {e}")
        return df

    # ------------------------------------------------------------------ #
    # STEP I — Remove duplicate listings
    # ------------------------------------------------------------------ #
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["make", "model", "year", "km_driven", "price"])
        dropped = before - len(df)
        self.report["duplicates_dropped"] = dropped
        self.logger.info(f"Dropped {dropped} duplicate rows.")
        return df

    def _reset_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        return df

    def _print_report(self):
        self.logger.info("===== CLEANING REPORT =====")
        for step, count in self.report.items():
            self.logger.info(f"  {step}: {count} rows affected")
        self.logger.info("===========================")


# ------------------------------------------------------------------ #
# Run this file directly to clean and save processed data
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    raw_path = "data/raw/vehicle_sales_raw.csv"
    output_path = "data/processed/vehicle_sales_clean.csv"

    df_raw = pd.read_csv(raw_path)
    cleaner = VehicleDataCleaner()
    df_clean = cleaner.run(df_raw)

    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to: {output_path}")
    print(f"Final shape: {df_clean.shape}")