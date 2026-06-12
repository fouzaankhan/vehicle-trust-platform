# Vehicle Trust Intelligence Platform

## Completed

### Session 1–4

* Data ingestion and cleaning pipeline
* Feature engineering pipeline
* Exploratory data analysis
* Dataset preparation

### Session 5 — Price Prediction Model

* XGBoost regression model
* Leakage removed (`mmr`, `mmr_deviation`)
* Early stopping enabled
* Final metrics:

  * R² = 0.9064
  * MAE = 1680.56
  * RMSE = 2904.03
  * MAPE = 14.46%
* SHAP explainability reports generated

### Session 6 — FastAPI Backend

* `/health`
* `/predict/price`
* Pydantic validation
* Model loading at startup
* Prediction helper module

### Session 7 — Streamlit Dashboard

* Home page
* Analyze Listing page
* Market Analytics page
* History page
* SQLite history storage
* FastAPI integration

### Session 8 — NLP Fraud Detection

* Rule-based fraud scoring
* Scam phrase detection
* Urgency detection
* Contact information detection
* Fraud analysis API endpoint:

  * `/analyze/description`
* Sentence-transformer duplicate detection index built
* Embeddings saved successfully

## Current Status

Project is functional end-to-end:

* Price prediction
* REST API
* Dashboard
* Fraud detection
