# Churn Prediction MLOps Pipeline

[![CI/CD](https://github.com/Nihad257/churn-mlops-pipeline/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Nihad257/churn-mlops-pipeline/actions/workflows/docker-build.yml)
[![Docker Image](https://img.shields.io/badge/ghcr.io-churn--mlops--pipeline-blue)](https://github.com/Nihad257/churn-mlops-pipeline/pkgs/container/churn-mlops-pipeline)

Production-grade customer churn prediction system built end-to-end — from EDA and classical ML to MLflow tracking, FastAPI serving, SHAP explainability, Docker, CI/CD, and drift monitoring.

## Architecture

data/                   ← Telco customer dataset  
01_eda.ipynb            ← Exploratory data analysis  
02_train_model.ipynb     ← Feature engineering + XGBoost baseline  
03_mlflow.ipynb          ← MLflow tracking, registry, LightGBM champion  
app.py                   ← FastAPI server + SHAP explainability + Risk Console  
drift_simple.py          ← Lightweight drift monitoring (KS + TV distance)  
Dockerfile               ← Container definition  
.github/workflows/       ← CI/CD (build + push to GHCR)

## Tech Stack

| Area | Tools |
|------|------|
| Languages | Python, SQL |
| Classical ML | XGBoost, LightGBM, Scikit-learn |
| Experiment Tracking | MLflow (tracking + model registry) |
| Serving | FastAPI, Uvicorn |
| Explainability | SHAP |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Drift Monitoring | Custom KS-test + TV distance |
| Visualization | Matplotlib, Plotly |

## Key Results

| Model | ROC-AUC | Churn Recall |
|-------|---------|--------------|
| XGBoost | 0.829 | 0.73 |
| LightGBM (champion) | **0.831** | **0.74** |

Champion model is registered in MLflow as `@champion`.

## Run Locally

git clone https://github.com/Nihad257/churn-mlops-pipeline.git  
cd churn-mlops-pipeline

python -m venv venv  
venv\Scripts\activate

pip install -r requirements.txt  
uvicorn app:app --reload --host 127.0.0.1 --port 8000

Open in browser:  
http://127.0.0.1:8000/dashboard

## Docker

docker pull ghcr.io/nihad257/churn-mlops-pipeline:latest  
docker run -p 8000:8000 ghcr.io/nihad257/churn-mlops-pipeline:latest

The image is automatically built and pushed by GitHub Actions on every push to `main`.

## Drift Monitoring

Run:

python drift_simple.py

This compares reference vs production data using:

- Kolmogorov–Smirnov test for numerical features
- Total variation distance for categorical features

Features with `p < 0.05` or `TV > 0.1` are flagged as drifted.

## Project Phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | EDA & baseline model | ✅ |
| 2 | Feature engineering + XGBoost | ✅ |
| 3 | MLflow tracking & registry | ✅ |
| 4 | FastAPI + Risk Console + SHAP | ✅ |
| 5 | Docker + GitHub Actions CI/CD | ✅ |
| 6 | Drift monitoring | ✅ |

