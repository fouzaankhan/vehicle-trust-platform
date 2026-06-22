# 🚗 AI-Powered Vehicle Trust Intelligence Platform

An end-to-end machine learning application for evaluating used vehicle listings by combining **price prediction**, **seller-description fraud analysis**, **image quality checks**, and a **unified trust score** into one system.

This project was built primarily as a **learning-focused full-stack ML project** to understand how to take a trained model and turn it into a usable application with a frontend, backend, storage, and Dockerized deployment.

---

## ✨ Features

* **Fair Price Prediction** using an XGBoost regression model
* **Trust Report Generation** for complete vehicle listings
* **NLP-Based Listing Review** to flag suspicious seller descriptions
* **Image Quality Analysis** using OpenCV-based checks
* **Market Analytics Dashboard** for exploring vehicle pricing patterns
* **Analysis History Tracking** using SQLite
* **FastAPI Backend** for model serving and trust analysis
* **Streamlit Frontend** for an interactive multi-page dashboard
* **Dockerized Deployment** using Docker Compose

---

## 📌 Project Scope

This is a **portfolio / learning project**, not a production-grade automotive valuation system.

The goal of the project was to build and integrate the full pipeline:

* train a vehicle price model
* expose predictions through an API
* build a dashboard to interact with the model
* add supporting trust signals such as NLP and image checks
* store analysis history
* containerize the entire application with Docker

It is useful as an **ML engineering / applied AI project**, but the trust score and valuation outputs should **not** be treated as production-ready financial or consumer advice.

---

## 🧠 What the Platform Does

The platform evaluates a used-car listing using multiple signals:

### 1) Price Prediction

The backend estimates a **fair market price** for a vehicle based on inputs such as:

* make
* model
* year
* kilometers driven
* transmission
* condition
* sale month

### 2) Seller Description Analysis

The listing description is checked for suspicious wording or scam-like language patterns using an NLP-based fraud detector.

### 3) Image Quality Analysis

Uploaded vehicle images can be checked for quality-related issues such as blur / poor visibility, which can be treated as a supporting trust signal.

### 4) Unified Trust Score

The platform combines:

* predicted fair price
* listed price
* NLP fraud result
* image quality result

into a final **Trust Report** with:

* trust score
* risk tier
* price anomaly explanation
* fraud / image observations

---

## 🏗️ System Architecture

```text
User Input / Listing Details
          │
          ▼
    Streamlit Frontend
          │
          ▼
     FastAPI Backend
          │
 ┌────────┼────────┬───────────────┐
 ▼        ▼        ▼               ▼
Price   NLP      Image          Trust
Model  Analysis  Analysis       Engine
 │        │        │               │
 └────────┴────────┴───────────────┘
                  │
                  ▼
            Trust Report Output
                  │
                  ▼
          SQLite History Storage
```

---

## 🗂️ Project Structure

```text
vehicle-trust-platform/
│
├── frontend/                     # Streamlit frontend
│   ├── main.py
│   ├── pages/
│   │   ├── 1_Analyze_Listing.py
│   │   ├── 2_Market_Analytics.py
│   │   ├── 3_History.py
│   │   └── 4_Trust_Report.py
│   └── utils/
│       └── api_client.py
│
├── backend/                      # FastAPI backend + ML pipeline logic
│   ├── main.py
│   ├── data/
│   │   ├── clean.py
│   │   └── generate_descriptions.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── build_nlp_index.py
│   │   ├── duplicate_detector.py
│   │   ├── image_analyzer.py
│   │   ├── nlp_fraud_detector.py
│   │   ├── predict.py
│   │   ├── train_price_model.py
│   │   └── trust_engine.py
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
│
├── assets/
│   └── sample_images/            # Example vehicle images
│
├── config/
│   └── config.yaml               # Project configuration
│
├── data/
│   ├── uploads/                  # Uploaded listing images at runtime
│   └── analyses.db               # SQLite database for analysis history
│
├── models/                       # Trained model + artifacts
│   ├── price_model_v1.joblib
│   ├── feature_cols_v1.joblib
│   ├── make_medians.joblib
│   ├── model_medians.joblib
│   └── description_ids.joblib
│
├── reports/                      # Training outputs / evaluation plots
├── notebooks/                    # EDA / experimentation notebooks
├── dev_tools/                    # Inspection, testing, and scratch scripts
│   ├── testing/
│   ├── inspection/
│   └── experiments/
│
├── Dockerfile                    # Backend container
├── Dockerfile.streamlit          # Frontend container
├── docker-compose.yml            # Multi-container app runner
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

### Core Application

* **Python**
* **FastAPI** — backend API for serving predictions and trust analysis
* **Streamlit** — frontend dashboard UI
* **SQLite** — lightweight local database for storing analysis history
* **Docker + Docker Compose** — containerized deployment

### Machine Learning / Data

* **Pandas** — data handling
* **NumPy** — numerical operations
* **Scikit-learn** — preprocessing / evaluation utilities
* **XGBoost** — vehicle price prediction model
* **Joblib** — saving / loading trained model artifacts

### NLP / Image Processing

* **Custom NLP fraud detector** for suspicious listing descriptions
* **OpenCV** — image quality analysis
* **Pillow** — image handling support

### Visualization

* **Matplotlib**
* **Plotly**

### Validation / Config

* **Pydantic** — request validation for FastAPI
* **PyYAML** — config loading
* **python-dotenv** — environment variable loading

---

## 📊 Model Performance

### Price Prediction Model

* **Model:** XGBoost Regressor
* **Training dataset size:** 533,346 vehicle listings
* **Target:** vehicle sale price

| Metric |     Value |
| ------ | --------: |
| R²     |    0.9064 |
| MAE    | $1,680.56 |
| RMSE   | $2,904.03 |
| MAPE   |    14.46% |

> These results reflect the final model after removing leakage-heavy features and retraining on a more realistic feature set.

---

## ⚠️ Key Engineering Lessons / Decisions

### 1) Leakage Detection in the Price Model

An unrealistically high R² (~0.997) initially appeared during training.

That performance was misleading. Investigation showed that the model had learned from leakage-heavy market reference features, making the evaluation artificially strong. The feature pipeline was revised and the model was retrained, resulting in a much more realistic **R² of 0.9064**.

### 2) End-to-End App Refactor

The project was refactored into a clearer **frontend / backend structure**:

* **frontend/** for Streamlit UI
* **backend/** for FastAPI and ML logic

This improved maintainability and made the app structure more aligned with real deployment patterns.

### 3) Dockerized Multi-Service Setup

The app was containerized using **Docker Compose** so the frontend and backend can run together consistently without manually launching separate processes in different terminals.

### 4) Lightweight Persistence with SQLite

A small SQLite database is used to store:

* vehicle analyses
* trust report history
* listing metadata for previous runs

This keeps the project self-contained without needing a full database server.

---

# 🛠️ Installation & Setup

## Prerequisites

Make sure you have the following installed:

* **Python 3.11+**
* **Git**
* **Docker Desktop** (recommended for easiest setup)

---

# Option 1 — Run with Docker (Recommended)

This is the easiest way to run the full project.

## Step 1: Clone the Repository

```bash
git clone https://github.com/fouzaankhan/vehicle-trust-platform.git
cd vehicle-trust-platform
```

## Step 2: Create Environment File

Create a `.env` file in the project root if needed, or copy from the example:

### macOS / Linux

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Example `.env` content:

```env
PROJECT_NAME=vehicle-trust-platform
ENV=development
LOG_LEVEL=INFO
```

## Step 3: Build and Start the Application

```bash
docker compose up --build
```

This starts:

* **FastAPI backend** on `http://localhost:8000`
* **Streamlit frontend** on `http://localhost:8501`

## Step 4: Open the App

Frontend dashboard:

```text
http://localhost:8501
```

Backend health endpoint:

```text
http://localhost:8000/health
```

Expected health response:

```json
{"status":"ok","model_version":"v1"}
```

---

# Option 2 — Run Without Docker (Manual Local Run)

Use this if you want to run frontend and backend manually in separate terminals.

## Step 1: Clone the Repository

```bash
git clone https://github.com/fouzaankhan/vehicle-trust-platform.git
cd vehicle-trust-platform
```

## Step 2: Create and Activate a Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Create Environment File

Create a `.env` file in the root of the project:

```env
PROJECT_NAME=vehicle-trust-platform
ENV=development
LOG_LEVEL=INFO
```

## Step 5: Start the Backend API

Open **Terminal 1** and run:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Step 6: Start the Streamlit Frontend

Open **Terminal 2** and run:

### Windows PowerShell

```powershell
$env:API_BASE_URL="http://localhost:8000"
streamlit run frontend/main.py
```

### macOS / Linux

```bash
export API_BASE_URL=http://localhost:8000
streamlit run frontend/main.py
```

## Step 7: Open the App

Frontend:

```text
http://localhost:8501
```

Backend health check:

```text
http://localhost:8000/health
```

---

## ▶️ How to Use the App

### 1. Analyze Listing

Use the **Analyze Listing** page to enter:

* make
* model
* year
* kilometers driven
* condition
* transmission
* sale month

The app will return a predicted fair price.

### 2. Trust Report

Use the **Trust Report** page to evaluate a full listing using:

* vehicle details
* listed price
* seller description
* optional vehicle image

The app will return:

* predicted price
* trust score
* risk tier
* fraud / image / price anomaly observations

### 3. Market Analytics

Use **Market Analytics** to explore pricing patterns and summary visuals from the processed vehicle sales dataset.

### 4. History

Use **History** to view saved past analyses stored in SQLite.

---

## 💾 Runtime Data Storage

The project stores runtime data locally:

### Analysis History

Stored in:

```text
data/analyses.db
```

This SQLite database stores previous vehicle analyses and trust report history.

### Uploaded Images

Stored in:

```text
data/uploads/
```

Any image uploaded through the Trust Report flow is saved here for analysis.

---

## ⚠️ Limitations

This project has important limitations:

* The price model is trained on a specific historical vehicle sales dataset and may not generalize reliably to real-world live dealer listings.
* Trust scoring is heuristic and learning-focused, not production-calibrated.
* NLP fraud analysis is not a full fraud-detection system.
* Image analysis focuses on simple quality checks and is not a robust damage assessment engine.
* The project uses local SQLite storage rather than a production database stack.

So be clear about what this is:

* **good portfolio project**
* **good ML engineering / app integration project**
* **not a production-grade used-car trust platform**

---

## 🚀 Future Improvements

If this project were to be continued further, the next meaningful improvements would be:

* retrain on a more realistic and cleaner live-market dataset
* build stronger vehicle-specific price features
* improve NLP fraud detection beyond rule-based heuristics
* add duplicate listing detection to the production pipeline
* add proper authentication / user accounts
* replace SQLite with PostgreSQL for larger-scale usage
* deploy backend and frontend to cloud infrastructure
* add monitoring / logging / model version tracking
* improve trust score calibration using real labeled scam / safe listings

---
