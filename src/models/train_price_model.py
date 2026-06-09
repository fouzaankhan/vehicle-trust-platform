import pandas as pd
import numpy as np
import yaml
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


class PriceModelTrainer:

    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(__name__)
        self.model = None
        self.feature_cols = None

    def run(self):
        df = self._load_data()
        X_train, X_test, y_train, y_test = self._split(df)
        self.model = self._train(X_train, y_train, X_test, y_test)
        self._evaluate(X_test, y_test)
        self._save_model()
        self._generate_shap_report(X_test)

    # ------------------------------------------------------------------ #
    # Load and select only the feature columns defined in config
    # ------------------------------------------------------------------ #
    def _load_data(self) -> pd.DataFrame:
        path = "data/processed/vehicle_sales_features.csv"
        df = pd.read_csv(path)
        self.logger.info(f"Loaded data: {df.shape}")

        numeric = self.config["features"]["numeric"]
        categorical = self.config["features"]["categorical_encoded"]
        self.feature_cols = numeric + categorical

        # Keep only columns that actually exist in the dataframe
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        self.logger.info(f"Using {len(self.feature_cols)} features: {self.feature_cols}")

        # Drop rows with any NaN in feature columns or target
        target = self.config["features"]["target"]
        df = df[self.feature_cols + [target]].dropna()
        self.logger.info(f"Shape after dropping NaN rows: {df.shape}")
        return df

    # ------------------------------------------------------------------ #
    # Train/test split — stratify not needed for regression
    # ------------------------------------------------------------------ #
    def _split(self, df):
        target = self.config["features"]["target"]
        X = df[self.feature_cols]
        y = df[target]

        test_size = self.config["model"]["price_model"]["test_size"]
        random_state = self.config["model"]["price_model"]["random_state"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    # Train XGBoost model
    # ------------------------------------------------------------------ #
    def _train(self, X_train, y_train, X_test, y_test):
        cfg = self.config["model"]["price_model"]

        model = xgb.XGBRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"],
            early_stopping_rounds=cfg["early_stopping_rounds"],
            n_jobs=-1,
            verbosity=1
        )

        self.logger.info("Training XGBoost model with early stopping...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )
        self.logger.info(f"Best iteration: {model.best_iteration}")
        return model

    # ------------------------------------------------------------------ #
    # Evaluate on test set — report in REAL prices, not log prices
    # ------------------------------------------------------------------ #
    def _evaluate(self, X_test, y_test):
        y_pred_log = self.model.predict(X_test)

        # Convert log predictions back to real prices
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100

        metrics = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2)
        }

        self.logger.info("===== MODEL EVALUATION =====")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("============================")

        # Save metrics to reports/
        os.makedirs("reports", exist_ok=True)
        with open("reports/price_model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info("Metrics saved to reports/price_model_metrics.json")

        # Actual vs Predicted plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.1, s=1, color="steelblue")
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', linewidth=1, label="Perfect prediction")
        plt.xlabel("Actual Price ($)")
        plt.ylabel("Predicted Price ($)")
        plt.title(f"Actual vs Predicted — R²={metrics['R2']}, MAE=${metrics['MAE']:,.0f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/actual_vs_predicted.png", dpi=150)
        plt.close()
        self.logger.info("Plot saved to reports/actual_vs_predicted.png")

    # ------------------------------------------------------------------ #
    # Save model + feature column list together
    # ------------------------------------------------------------------ #
    def _save_model(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/price_model_v1.joblib")
        joblib.dump(self.feature_cols, "models/feature_cols_v1.joblib")
        self.logger.info("Model saved to models/price_model_v1.joblib")
        self.logger.info("Feature cols saved to models/feature_cols_v1.joblib")

    # ------------------------------------------------------------------ #
    # SHAP — explain what drives each prediction
    # ------------------------------------------------------------------ #
    def _generate_shap_report(self, X_test):
        self.logger.info("Generating SHAP explanations...")

        # Use a sample for speed — SHAP on 500 rows is enough for the report
        X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X_sample)

        # SHAP summary bar plot — global feature importance
        plt.figure()
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig("reports/shap_feature_importance.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # SHAP beeswarm — shows direction of each feature's effect
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig("reports/shap_beeswarm.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info("SHAP plots saved to reports/")


if __name__ == "__main__":
    trainer = PriceModelTrainer()
    trainer.run()