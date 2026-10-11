"""
Week 1 Task: Python for Machine Learning & Data Preprocessing
Step 0: Generate a raw (messy) customer churn dataset.

The dataset is SYNTHETIC: it is created with a fixed random seed so anyone
can reproduce it exactly. Realistic data-quality problems are injected on
purpose (missing values, duplicates, wrong data types, inconsistent labels,
impossible values) so that every preprocessing step in the next script has a
real problem to solve.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200

# ---------- 1. Generate clean "ground truth" customers ----------
customer_id = [f"C{str(i).zfill(4)}" for i in range(1, N + 1)]
gender = rng.choice(["Male", "Female"], size=N)
senior = rng.binomial(1, 0.16, size=N)
tenure = np.clip(rng.gamma(shape=2.0, scale=14.0, size=N), 1, 72).astype(int)
contract = rng.choice(["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.25, 0.20])
internet = rng.choice(["DSL", "Fiber optic", "No"], size=N, p=[0.35, 0.45, 0.20])
payment = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=N, p=[0.35, 0.20, 0.225, 0.225],
)

base_price = np.select(
    [internet == "No", internet == "DSL", internet == "Fiber optic"], [20, 50, 80]
)
monthly = np.round(base_price + rng.normal(0, 8, size=N), 2)
monthly = np.clip(monthly, 15, None)
total = np.round(tenure * monthly * rng.normal(1.0, 0.03, size=N), 2)
support_calls = rng.poisson(1.3, size=N)

# Churn depends on several real relationships (so the analysis has signal)
logit = (
    -1.2
    + 1.3 * (contract == "Month-to-month")
    - 0.9 * (contract == "Two year")
    + 0.6 * (internet == "Fiber optic")
    - 0.03 * tenure
    + 0.35 * support_calls
    + 0.01 * (monthly - 60)
    + 0.4 * (payment == "Electronic check")
    + 0.2 * senior
)
p_churn = 1 / (1 + np.exp(-logit))
churn = np.where(rng.random(N) < p_churn, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior,
    "tenure_months": tenure.astype(float),
    "contract": contract,
    "internet_service": internet,
    "payment_method": payment,
    "monthly_charges": monthly,
    "total_charges": total.astype(object),
    "support_calls": support_calls.astype(float),
    "Churn": churn,
})

# ---------- 2. Inject data-quality problems ----------
def pick(frac):
    return rng.choice(N, size=int(N * frac), replace=False)

# Missing values (random)
df.loc[pick(0.05), "monthly_charges"] = np.nan
df.loc[pick(0.03), "tenure_months"] = np.nan
df.loc[pick(0.04), "internet_service"] = np.nan
df.loc[pick(0.03), "support_calls"] = np.nan

# total_charges stored as text, with blank strings for some customers
df["total_charges"] = df["total_charges"].astype(str)
df.loc[pick(0.01), "total_charges"] = " "

# Inconsistent category labels (casing / spacing / abbreviations)
idx = pick(0.06)
df.loc[idx, "contract"] = df.loc[idx, "contract"].str.lower()
idx = pick(0.04)
df.loc[idx, "contract"] = df.loc[idx, "contract"] + " "
idx = pick(0.05)
df.loc[idx, "internet_service"] = df.loc[idx, "internet_service"].str.upper()
idx = pick(0.04)
df.loc[idx, "gender"] = df.loc[idx, "gender"].str.lower()
idx = pick(0.03)
df.loc[idx, "gender"] = df.loc[idx, "gender"].map({"Male": "M", "Female": "F", "male": "M", "female": "F"}).fillna(df.loc[idx, "gender"])

# Impossible / outlier values (data-entry errors)
bad_monthly = pick(0.005)
df.loc[bad_monthly, "monthly_charges"] = df.loc[bad_monthly, "monthly_charges"].fillna(70) * 10
bad_tenure = rng.choice(N, size=4, replace=False)
df.loc[bad_tenure, "tenure_months"] = -1

# Duplicate rows (same customer recorded twice)
dups = df.sample(n=20, random_state=7)
df = pd.concat([df, dups], ignore_index=True)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("raw_customer_churn.csv", index=False)
print(f"Raw dataset saved: raw_customer_churn.csv  shape={df.shape}")
