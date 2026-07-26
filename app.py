# app.py  —  Phase 4, Step B: Churn Risk API + Interactive Console + SHAP
import os, warnings, joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import shap

warnings.filterwarnings("ignore")

PROJECT = r"D:\projects\real 3 JD prjct\churn-mlops-pipeline"
MODEL_PATH = os.path.join(PROJECT, "models", "xgboost_churn_v1.joblib")
COLS_PATH  = os.path.join(PROJECT, "models", "feature_columns.joblib")

# Load once at startup
model = joblib.load(MODEL_PATH)
FEATURE_COLS = joblib.load(COLS_PATH)
print(f"✅ Loaded model | classes={model.classes_.tolist()} | features={len(FEATURE_COLS)}")

# Prepare SHAP explainer (use TreeExplainer for XGBoost)
explainer = shap.TreeExplainer(model)
print("✅ SHAP explainer ready")

app = FastAPI(title="Churn Risk API", version="0.2.0")

class Customer(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 1
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 85.0
    TotalCharges: float = 85.0

def transform(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost_churn_v1"}

@app.post("/predict")
def predict(customer: Customer):
    raw = customer.model_dump()
    X = transform(raw)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    risk = "HIGH RISK 🔴" if pred == 1 else "LOW RISK 🟢"
    return {
        "churn_probability": round(proba, 4),
        "prediction": pred,
        "risk_label": risk
    }

@app.post("/explain")
def explain(customer: Customer):
    """Return top SHAP feature contributions for this customer."""
    raw = customer.model_dump()
    X = transform(raw)
    shap_values = explainer.shap_values(X)
    # For binary classification with XGBoost, shap_values is a list of two arrays;
    # we take the positive class (index 1) contributions.
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]
    # Pair feature names with SHAP values
    contributions = sorted(
        zip(FEATURE_COLS, vals),
        key=lambda x: abs(x[1]), reverse=True
    )[:10]
    return {
        "features": [c[0] for c in contributions],
        "shap_values": [round(float(c[1]), 4) for c in contributions]
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Churn Risk Console</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 2rem; }
  .container { max-width: 800px; margin: 0 auto; }
  h1 { color: #1a1a2e; text-align: center; }
  .card { background: white; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
  .presets { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .preset-btn { flex: 1; padding: 0.8rem; border: none; border-radius: 8px; color: white; font-weight: bold; cursor: pointer; transition: transform 0.1s; }
  .preset-btn:hover { transform: scale(1.02); }
  .risky { background: #e74c3c; }
  .safe { background: #27ae60; }
  .result-box { text-align: center; margin: 1rem 0; }
  .gauge { width: 200px; height: 100px; margin: 0 auto; position: relative; }
  .risk-label { font-size: 2rem; font-weight: bold; margin: 0.5rem 0; }
  .probability { font-size: 3rem; font-weight: bold; color: #2c3e50; }
  .shap-container { height: 300px; }
  .footer { text-align: center; color: #7f8c8d; margin-top: 2rem; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="container">
  <h1>🔍 Churn Risk Console</h1>
  
  <div class="card">
    <div class="presets">
      <button class="preset-btn risky" onclick="setCustomer('risky')">🔴 High-Risk Customer</button>
      <button class="preset-btn safe" onclick="setCustomer('safe')">🟢 Low-Risk Customer</button>
    </div>
    
    <div class="result-box">
      <div class="gauge">
        <svg width="200" height="100" viewBox="0 0 200 100">
          <path d="M20,90 A80,80 0 0,1 180,90" fill="none" stroke="#ddd" stroke-width="20" stroke-linecap="round"/>
          <path id="gauge-fill" d="M20,90 A80,80 0 0,1 180,90" fill="none" stroke="#e74c3c" stroke-width="20" stroke-linecap="round" stroke-dasharray="251" stroke-dashoffset="251"/>
          <text id="gauge-text" x="100" y="70" text-anchor="middle" font-size="18" fill="#2c3e50">0%</text>
        </svg>
      </div>
      <div id="risk-label" class="risk-label">--</div>
      <div id="probability" class="probability">--</div>
    </div>
  </div>
  
  <div class="card">
    <h3>📊 Why this prediction? (Top SHAP features)</h3>
    <div class="shap-container">
      <canvas id="shapChart" width="750" height="300"></canvas>
    </div>
  </div>
  
  <div class="footer">
    Built with FastAPI + XGBoost + SHAP | churn-mlops-pipeline
  </div>
</div>

<script>
const riskyCustomer = {
  gender: "Female", SeniorCitizen: 1, Partner: "No", Dependents: "No",
  tenure: 1, PhoneService: "Yes", MultipleLines: "No",
  InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No",
  DeviceProtection: "No", TechSupport: "No", StreamingTV: "No",
  StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check", MonthlyCharges: 85, TotalCharges: 85
};

const safeCustomer = {
  gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes",
  tenure: 60, PhoneService: "Yes", MultipleLines: "No",
  InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes",
  DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No",
  StreamingMovies: "No", Contract: "Two year", PaperlessBilling: "No",
  PaymentMethod: "Bank transfer (automatic)", MonthlyCharges: 45, TotalCharges: 2700
};

async function predictCustomer(customer) {
  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(customer)
  });
  const result = await response.json();
  document.getElementById('risk-label').textContent = result.risk_label;
  document.getElementById('probability').textContent = (result.churn_probability * 100).toFixed(1) + '%';
  // Update gauge
  const gaugeFill = document.getElementById('gauge-fill');
  const gaugeText = document.getElementById('gauge-text');
  const dashOffset = 251 - (251 * result.churn_probability);
  gaugeFill.setAttribute('stroke-dashoffset', dashOffset);
  gaugeFill.setAttribute('stroke', result.prediction === 1 ? '#e74c3c' : '#27ae60');
  gaugeText.textContent = (result.churn_probability * 100).toFixed(0) + '%';
  
  // Fetch SHAP explanation
  const explainResp = await fetch('/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(customer)
  });
  const shapData = await explainResp.json();
  drawShapChart(shapData.features, shapData.shap_values);
}

function drawShapChart(features, values) {
  const canvas = document.getElementById('shapChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  const barHeight = 25;
  const startY = 20;
  const maxVal = Math.max(...values.map(Math.abs));
  
  features.forEach((feat, i) => {
    const x = 150;
    const y = startY + i * (barHeight + 5);
    const val = values[i];
    const width = Math.abs(val) / maxVal * 250;
    const color = val > 0 ? '#e74c3c' : '#27ae60';
    
    ctx.fillStyle = '#333';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(feat.slice(0, 20) + (feat.length > 20 ? '…' : ''), 145, y + 16);
    
    ctx.fillStyle = color;
    ctx.fillRect(x + (val < 0 ? 0 : 0), y, width, barHeight);
    ctx.fillStyle = '#333';
    ctx.textAlign = 'left';
    ctx.fillText(val.toFixed(3), x + width + 5, y + 16);
  });
}

function setCustomer(type) {
  const cust = type === 'risky' ? riskyCustomer : safeCustomer;
  predictCustomer(cust);
}

// Default: high-risk on load
window.onload = () => predictCustomer(riskyCustomer);
</script>
</body>
</html>
"""