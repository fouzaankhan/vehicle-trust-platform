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
