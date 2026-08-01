# drift_simple.py  —  Library-free drift summary (KS test + total-variation distance)
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT = r"D:\projects\real 3 JD prjct\churn-mlops-pipeline"
os.makedirs(os.path.join(PROJECT, "reports"), exist_ok=True)

# Load raw data
df = pd.read_csv(os.path.join(PROJECT, "data", "telco_churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df = df.drop(columns=["customerID", "Churn"])   # features only

# Reference = first half ; Current = second half WITH injected drift
reference = df.sample(frac=0.5, random_state=42)
current   = df.drop(reference.index).copy()
current["MonthlyCharges"] = current["MonthlyCharges"] * 1.4          # prices jump
current["tenure"]         = (current["tenure"] * 0.6).astype(int)    # shorter tenure
current["Contract"]       = np.where(current["Contract"] == "Two year",
                                      "Month-to-month", current["Contract"])  # fewer loyal

rows = []
for col in reference.columns:
    if pd.api.types.is_numeric_dtype(reference[col]):
        # Kolmogorov–Smirnov test for numeric features
        stat, p = ks_2samp(reference[col].dropna(), current[col].dropna())
        rows.append({
            "feature": col,
            "drift_score": round(stat, 3),
            "p_value": round(p, 4),
            "drifted": "YES 🔴" if p < 0.05 else "no"
        })
    else:
        # Total-variation distance for categorical features
        a = reference[col].value_counts(normalize=True)
        b = current[col].value_counts(normalize=True)
        all_keys = set(a.index) | set(b.index)
        tv = 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in all_keys)
        rows.append({
            "feature": col,
            "drift_score": round(tv, 3),
            "p_value": None,
            "drifted": "YES 🔴" if tv > 0.1 else "no"
        })

summary = pd.DataFrame(rows).sort_values("drift_score", ascending=False)
print(summary.to_string(index=False))

# Save CSV
csv_path = os.path.join(PROJECT, "reports", "drift_summary.csv")
summary.to_csv(csv_path, index=False)
print(f"\n✅ Drift summary saved to {csv_path}")
print("Open that CSV to see the drift dashboard (or read it in Excel).")