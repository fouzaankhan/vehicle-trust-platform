# 🚗 AI-Powered Vehicle Trust Intelligence Platform

An end-to-end machine learning prototype for analyzing used vehicle listings by combining price estimation, listing-text fraud checks, image-quality analysis, and a rule-based trust scoring pipeline into a unified dashboard.

## Features

- Fair market price prediction using XGBoost
- NLP-based fraud detection for vehicle listings
- Image quality analysis using OpenCV
- Unified Trust Engine producing a 0–100 trust score
- FastAPI backend with REST APIs
- Streamlit dashboard for end-to-end analysis
- SQLite-based analysis history tracking

## Dataset

Source: Vehicle Sales Dataset

- Records: 533,346 listings
- Features: Vehicle age, mileage, condition, make/model statistics, transmission, and market features
- Target: Log-transformed vehicle price (`log_price`)

## Model Performance

### Price Prediction Model (XGBoost)

| Metric | Value |
|----------|----------|
| R² | 0.9064 |
| MAE | $1,680.56 |
| RMSE | $2,904.03 |
| MAPE | 14.46% |

Training dataset size: 533,346 vehicle listings.

## Architecture

Price Model (XGBoost)
        │
        ▼
NLP Fraud Detector
        │
        ▼
Image Quality Analyzer
        │
        ▼
Trust Engine
        │
        ▼
FastAPI Backend
        │
        ▼
Streamlit Dashboard

## Engineering Decisions

### 1. Data Leakage Detection and Fix

During model development, an unrealistically high R² (~0.997) was observed.

Investigation revealed target leakage through market-reference features. The feature pipeline was redesigned and the model retrained, resulting in a realistic and reliable R² of 0.9064.

### 2. PyTorch Dependency Conflict

SentenceTransformer-based duplicate listing detection was developed but disabled in production after encountering Windows-specific PyTorch DLL initialization failures (WinError 1114).

A rule-based NLP fraud detector was deployed instead, while preserving the semantic duplicate detection module for future containerized deployment.

### 3. Image Scoring Redesign

Initial HSV-based rust detection generated false positives from backgrounds and environmental objects.

The image pipeline was redesigned so that trust scoring relies on reliable image quality metrics (blur, brightness, contrast), while damage detection remains available as an experimental review signal.

## Trust Engine Validation

### Scam Scenario

- Predicted Price: $18,376
- Listed Price: $9,000
- NLP Fraud Score: 100
- Trust Score: 24/100
- Tier: LIKELY SCAM

### Legitimate Scenario

- Predicted Price: $18,376
- Listed Price: $18,000
- NLP Fraud Score: 0
- Trust Score: 100/100
- Tier: TRUSTWORTHY

## Tech Stack

- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- OpenCV
- SQLite
- Git
- Docker

## 🛠️ Installation & Setup

### Prerequisites

Make sure the following are installed on your system before running the project:

* **Python 3.11**
* **Docker Desktop**
* **Git**

You will also need the trained model files and processed dataset already present in the project structure.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/fouzaankhan/vehicle-trust-platform.git
cd vehicle-trust-platform
```

---

### Step 2: Create the Environment File

Create a `.env` file in the project root with the following content:

```env
PROJECT_NAME=vehicle-trust-platform
ENV=development
LOG_LEVEL=INFO
```

---

### Step 3: Verify Required Project Files

Before running the project, make sure these files and folders exist inside the repository:

```bash
vehicle-trust-platform/
│
├── app/
├── src/
├── data/
│   ├── processed/
│   │   ├── vehicle_sales_clean.csv
│   │   ├── vehicle_sales_features.csv
│   │   └── listing_descriptions.csv
│
├── models/
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

### Step 4: Build and Start the Application

Run the following command from the project root:

```bash
docker compose up --build
```

This will start both services:

* **FastAPI backend** on port **8000**
* **Streamlit dashboard** on port **8501**

---

### Step 5: Open the Application

After the containers start successfully, open the Streamlit dashboard in your browser:

```bash
http://localhost:8501
```

You can also verify that the backend API is running by visiting:

```bash
http://localhost:8000/health
```

Expected API response:

```json
{"status":"ok","model_version":"v1"}
```

---

## 🚗 Using the Vehicle Trust Platform

### Analyze a Vehicle Listing

1. Open the **Streamlit dashboard** at `http://localhost:8501`
2. Go to the **Analyze Listing** page
3. Enter listing details such as:

   * Make
   * Model
   * Year
   * Mileage / km driven
   * Listed price
   * Transmission
   * Condition
   * Sale month
   * Seller description
4. Optionally upload a vehicle image
5. Submit the listing for analysis

The platform will generate:

* **Predicted Fair Price**
* **Trust Score**
* **Risk Tier**
* **Price Anomaly Risk**
* **NLP Fraud Risk**
* **Image Quality Risk**
* **Explanatory Trust Report**

---

## 🧩 Running Without Docker (Optional)

If you want to run the backend and dashboard manually instead of using Docker, use two terminals.

### Terminal 1 — Start FastAPI Backend

```bash
uvicorn src.api.main:app --reload
```

### Terminal 2 — Start Streamlit Dashboard

```bash
streamlit run app/main.py
```

Then open:

* Dashboard → `http://localhost:8501`
* API health check → `http://localhost:8000/health`

---

## 🛑 Stopping the Application

To stop the Docker containers:

```bash
docker compose down
```

If you started the backend and dashboard manually, stop them using `Ctrl + C` in each terminal.


## Project Scope

This project was built as an end-to-end applied machine learning and software engineering prototype for used vehicle listing analysis. Its purpose is to demonstrate model development, risk scoring, API design, dashboard integration, and Dockerized deployment in a realistic workflow.
The current version is constrained by the available training data, simplified fraud heuristics, and lightweight image-quality analysis. The trust score should therefore be interpreted as a decision-support signal within a learning prototype, not as a definitive real-world fraud or pricing judgment.
