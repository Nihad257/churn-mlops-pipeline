# app.py — Churn Risk Console (clean, self-contained, professional UI)
import os, warnings, joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import shap

warnings.filterwarnings("ignore")

PROJECT    = r"D:\projects\real 3 JD prjct\churn-mlops-pipeline"
MODEL_PATH = os.path.join(PROJECT, "models", "xgboost_churn_v1.joblib")
COLS_PATH  = os.path.join(PROJECT, "models", "feature_columns.joblib")

model        = joblib.load(MODEL_PATH)
FEATURE_COLS = joblib.load(COLS_PATH)

EXPLAINER = None
try:
    EXPLAINER = shap.TreeExplainer(model)
    print("✅ SHAP explainer ready")
except Exception as e:
    print("⚠️  SHAP explainer unavailable (console will hide the chart):", e)

print(f"✅ Loaded model | classes={model.classes_.tolist()} | features={len(FEATURE_COLS)}")

app = FastAPI(title="Churn Risk API", version="1.0.0")

# ---------- one customer, as a business form would send it ----------
class Customer(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
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
    MonthlyCharges: float = 75.0
    TotalCharges: float = 75.0

NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# ---------- reproduce EXACTLY the training-time cleaning ----------
def transform(raw: dict) -> pd.DataFrame:
    row = pd.DataFrame([raw])
    for c in NUM_COLS:
        row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0)
    row["SeniorCitizen"] = row["SeniorCitizen"].astype(int)
    cat_cols = row.select_dtypes(include="object").columns.tolist()
    row[cat_cols] = row[cat_cols].fillna("Unknown")
    row = pd.get_dummies(row, columns=cat_cols, drop_first=False)
    row[row.select_dtypes(include="bool").columns] = \
        row[row.select_dtypes(include="bool").columns].astype(int)
    row.columns = [c.replace(" ", "_").replace("-", "_") for c in row.columns]
    row = row.reindex(columns=FEATURE_COLS, fill_value=0).fillna(0)   # 🔑 align to training cols
    return row

def pretty(name: str) -> str:
    return name.replace("_", " ")

# ---------- top SHAP drivers, defensively normalised for any shap version ----------
def shap_top(X: pd.DataFrame, n: int = 8):
    if EXPLAINER is None:
        return []
    try:
        raw = EXPLAINER.shap_values(X)
        if isinstance(raw, list):
            arr = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        else:
            arr = np.asarray(raw)
        if arr.ndim == 3:
            arr = arr[0, 1, :] if arr.shape[1] == 2 else arr[0, :, 1]
        elif arr.ndim == 2:
            arr = arr[1, :] if (arr.shape[0] == 2 and arr.shape[1] == len(FEATURE_COLS)) else arr[0, :]
        vals = arr.reshape(-1)
        if vals.shape[0] != len(FEATURE_COLS):
            return []
        pairs = sorted(zip(FEATURE_COLS, vals), key=lambda kv: abs(kv[1]), reverse=True)[:n]
        return [{"feature": pretty(k), "value": round(float(v), 3)} for k, v in pairs]
    except Exception as e:
        print("shap error:", e)
        return []

def risk_of(p: float):
    if p >= 0.60: return "HIGH RISK",   "🔴", "#e63946"
    if p >= 0.35: return "MEDIUM RISK", "🟠", "#f4a261"
    return "LOW RISK", "🟢", "#2a9d8f"

# =====================  THE CONSOLE (HTML / CSS / JS)  =====================
DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Risk Console</title>
<style>
  :root{ --bg:#f1f5f9; --ink:#0f172a; --muted:#64748b; --card:#ffffff;
         --line:#e2e8f0; --indigo:#4f46e5; --shadow:0 10px 30px rgba(15,23,42,.08); }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:var(--bg);color:var(--ink);line-height:1.5}
  header{background:linear-gradient(135deg,#0f172a,#1e293b 60%,#312e81);color:#fff;
         padding:26px 20px;text-align:center}
  header h1{margin:0;font-size:24px;letter-spacing:.3px}
  header p{margin:6px 0 0;color:#cbd5e1;font-size:13px}
  .live{display:inline-block;margin-top:10px;font-size:11px;font-weight:700;
        letter-spacing:.6px;background:rgba(45,212,191,.15);color:#5eead4;
        padding:4px 12px;border-radius:999px;border:1px solid rgba(45,212,191,.4)}
  main{max-width:760px;margin:0 auto;padding:22px 16px 60px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:18px;
        box-shadow:var(--shadow);padding:24px;margin-bottom:20px}
  .card h2{margin:0 0 4px;font-size:16px}
  .hint{margin:0 0 16px;color:var(--muted);font-size:13px}
  .presets{display:flex;gap:12px;flex-wrap:wrap}
  .preset{flex:1;min-width:200px;cursor:pointer;border:none;border-radius:14px;
          padding:16px 18px;font-size:15px;font-weight:700;color:#fff;
          transition:transform .12s ease, box-shadow .2s ease, opacity .2s}
  .preset:disabled{opacity:.6;cursor:wait}
  .preset.risky{background:linear-gradient(135deg,#ef4444,#b91c1c)}
  .preset.safe{background:linear-gradient(135deg,#10b981,#047857)}
  .preset:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,0,0,.18)}
  .preset.active{outline:3px solid #0f172a;outline-offset:2px}
  .preset small{display:block;font-weight:500;opacity:.9;margin-top:3px;font-size:12px}
  #status{min-height:18px;margin:12px 0 0;color:var(--indigo);font-size:13px;font-weight:600}
  #err{display:none;background:#fef2f2;border:1px solid #fecaca;color:#b91c1c;
       padding:12px 14px;border-radius:12px;margin-top:14px;font-size:13px}
  .gauge-wrap{text-align:center}
  .gauge{width:300px;max-width:82%;height:auto;display:block;margin:4px auto 0}
  .gauge-track{fill:none;stroke:#e2e8f0;stroke-width:16;stroke-linecap:round}
  .gauge-value{fill:none;stroke:#e63946;stroke-width:16;stroke-linecap:round;
               stroke-dasharray:0 282.743;
               transition:stroke-dasharray .9s cubic-bezier(.22,1,.36,1), stroke .4s}
  .gauge-num{font-weight:800;font-size:30px;text-anchor:middle;fill:#0f172a}
  #pill{display:inline-block;margin:14px 0 2px;font-size:20px;font-weight:800;
        letter-spacing:.5px;padding:8px 20px;border-radius:999px;border:1.5px solid}
  #probLabel{display:block;color:var(--muted);font-size:12px;margin-top:8px;
             text-transform:uppercase;letter-spacing:1px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:18px}
  .chip{background:#f8fafc;border:1px solid var(--line);border-radius:999px;
        padding:5px 12px;font-size:12px;color:#334155}
  .chip b{color:var(--muted);font-weight:600}
  .shap-row{display:grid;grid-template-columns:150px 1fr 56px;align-items:center;
            gap:12px;margin:9px 0}
  .shap-label{font-size:12.5px;color:#334155;text-align:right;word-break:break-word}
  .shap-track{background:#f1f5f9;border-radius:8px;height:16px;overflow:hidden}
  .shap-bar{height:100%;border-radius:8px;width:0;transition:width .7s cubic-bezier(.22,1,.36,1)}
  .shap-val{font-size:12.5px;font-weight:700;text-align:right;font-variant-numeric:tabular-nums}
  .muted{color:var(--muted);font-size:13px}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:6px}
  @media(max-width:520px){ .shap-row{grid-template-columns:96px 1fr 48px;gap:8px} }
</style>
</head>
<body>
  <header>
    <h1>📊 Churn Risk Console</h1>
    <p>Predict whether a telecom customer will leave — and see exactly <b>why</b>.</p>
    <span class="live">● LIVE · XGBoost v1 · FastAPI</span>
  </header>

  <main>
    <div class="card">
      <h2>1 · Simulate a customer</h2>
      <p class="hint">Pick a profile. The model scores it live and explains the verdict.</p>
      <div class="presets">
        <button class="preset risky" data-key="risky" onclick="run('risky')">
          🔴 At‑Risk Customer<small>new · fibre · month‑to‑month</small></button>
        <button class="preset safe"  data-key="safe"  onclick="run('safe')">
          🟢 Loyal Customer<small>2‑yr contract · 6‑yr tenure</small></button>
      </div>
      <p id="status"></p>
      <div id="err"></div>
    </div>

    <div class="card gauge-wrap">
      <svg viewBox="0 0 200 120" class="gauge">
        <path d="M10 100 A90 90 0 0 1 190 100" class="gauge-track"></path>
        <path id="gaugeVal" d="M10 100 A90 90 0 0 1 190 100" class="gauge-value"></path>
        <text id="gaugeNum" x="100" y="92" class="gauge-num">0%</text>
      </svg>
      <div><span id="pill">— </span></div>
      <span id="probLabel">probability of churn</span>
      <div class="chips" id="chips"></div>
    </div>

    <div class="card">
      <h2>2 · Why this prediction? <span style="color:var(--muted);font-weight:500">(top SHAP drivers)</span></h2>
      <p class="hint">🔴 pushes <b>toward</b> churn · 🟢 pushes <b>away</b> from churn.</p>
      <div id="shapRows"></div>
    </div>

    <footer>Model: XGBoost v1 · 45 engineered features · served locally via Uvicorn</footer>
  </main>

<script>
  const ARC = 282.743;
  const PRESETS = {
    risky: {gender:"Male",SeniorCitizen:0,Partner:"No",Dependents:"No",tenure:1,
            PhoneService:"Yes",MultipleLines:"No",InternetService:"Fiber optic",
            OnlineSecurity:"No",OnlineBackup:"No",DeviceProtection:"No",TechSupport:"No",
            StreamingTV:"No",StreamingMovies:"No",Contract:"Month-to-month",
            PaperlessBilling:"Yes",PaymentMethod:"Electronic check",
            MonthlyCharges:75.0,TotalCharges:75.0},
    safe:  {gender:"Female",SeniorCitizen:0,Partner:"Yes",Dependents:"Yes",tenure:72,
            PhoneService:"Yes",MultipleLines:"Yes",InternetService:"DSL",
            OnlineSecurity:"Yes",OnlineBackup:"Yes",DeviceProtection:"Yes",TechSupport:"Yes",
            StreamingTV:"No",StreamingMovies:"Yes",Contract:"Two year",
            PaperlessBilling:"No",PaymentMethod:"Bank transfer (automatic)",
            MonthlyCharges:20.0,TotalCharges:1500.0}
  };

  function setLoading(b){
    document.querySelectorAll('.preset').forEach(x => x.disabled = b);
    document.getElementById('status').textContent = b ? 'Evaluating model…' : '';
  }
  function setErr(m){
    const e = document.getElementById('err');
    e.textContent = m; e.style.display = m ? 'block' : 'none';
  }
  function setChips(o){
    const m = [['Contract',o.Contract],['Internet',o.InternetService],
               ['Tenure',o.tenure+' mo'],['Monthly $',o.MonthlyCharges],
               ['Payment',o.PaymentMethod]];
    document.getElementById('chips').innerHTML =
      m.map(x => '<span class="chip"><b>'+x[0]+':</b> '+x[1]+'</span>').join('');
  }
  function render(d){
    const pct = d.churn_probability * 100, color = d.risk_color;
    const gv = document.getElementById('gaugeVal');
    gv.style.stroke = color;
    requestAnimationFrame(() => { gv.style.strokeDasharray = (pct/100*ARC)+' '+ARC; });
    const gn = document.getElementById('gaugeNum');
    gn.textContent = pct.toFixed(1)+'%'; gn.style.fill = color;
    const pill = document.getElementById('pill');
    pill.textContent = d.risk_label+' '+d.risk_emoji;
    pill.style.color = color; pill.style.background = color+'1a'; pill.style.borderColor = color+'66';
    const wrap = document.getElementById('shapRows'); wrap.innerHTML = '';
    if(!d.shap || !d.shap.length){
      wrap.innerHTML = '<p class="muted">Explanations unavailable for this build.</p>';
    } else {
      const max = Math.max(...d.shap.map(s => Math.abs(s.value))) || 1;
      d.shap.forEach(s => {
        const pos = s.value >= 0, w = (Math.abs(s.value)/max*100).toFixed(1);
        const c = pos ? '#e63946' : '#2a9d8f';
        const row = document.createElement('div'); row.className = 'shap-row';
        row.innerHTML =
          '<span class="shap-label">'+s.feature+'</span>'+
          '<div class="shap-track"><div class="shap-bar" style="width:'+w+'%;background:'+c+'"></div></div>'+
          '<span class="shap-val" style="color:'+c+'">'+(pos?'+':'')+s.value.toFixed(2)+'</span>';
        wrap.appendChild(row);
      });
    }
  }
  async function run(key){
    document.querySelectorAll('.preset').forEach(b =>
      b.classList.toggle('active', b.dataset.key === key));
    setChips(PRESETS[key]); setErr(''); setLoading(true);
    // reset gauge so the animation replays
    document.getElementById('gaugeVal').style.strokeDasharray = '0 '+ARC;
    try{
      const r = await fetch('/predict', {method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify(PRESETS[key])});
      if(!r.ok) throw new Error('HTTP '+r.status);
      render(await r.json());
    }catch(e){
      setErr('Could not reach the model: '+e.message+'. Make sure uvicorn is running.');
    }finally{ setLoading(false); }
  }
  window.addEventListener('DOMContentLoaded', () => run('risky'));
</script>
</body>
</html>
"""

# =====================  ROUTES  =====================
@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost_churn_v1", "features": len(FEATURE_COLS)}

@app.post("/predict")
def predict(c: Customer):
    X     = transform(c.model_dump())
    proba = float(model.predict_proba(X)[:, 1][0])
    pred  = int(model.predict(X)[0])
    label, emoji, color = risk_of(proba)
    return {
        "churn_probability": round(proba, 4),
        "prediction": pred,
        "risk_label": label,
        "risk_emoji": emoji,
        "risk_color": color,
        "shap": shap_top(X),
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML