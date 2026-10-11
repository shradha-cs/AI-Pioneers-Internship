"""
Week 1 Task: Python for Machine Learning & Data Preprocessing

Documents every step of cleaning and preprocessing a raw customer churn
dataset, with exploratory data analysis (EDA), visualizations, and a short
justification printed for each decision.

Run:  python generate_dataset.py   (creates raw_customer_churn.csv)
      python preprocessing_eda.py  (runs everything below)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
BLUE, RED, GREY = "#2E86C1", "#C0392B", "#5D6D7E"
R = {}  # results collected for the report


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =====================================================================
# STEP 1: LOAD AND INSPECT
# =====================================================================
section("STEP 1: DATA LOADING & FIRST INSPECTION")
raw = pd.read_csv("raw_customer_churn.csv")
df = raw.copy()
print(f"Shape: {df.shape}")
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics (numeric):\n", df.describe().round(2))
R["raw_shape"] = list(df.shape)
R["raw_dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

# =====================================================================
# STEP 2: DATA QUALITY AUDIT (before cleaning)
# =====================================================================
section("STEP 2: DATA QUALITY AUDIT")
n_dup = int(df.duplicated().sum())
missing = df.isnull().sum()
blank_total = int((df["total_charges"].astype(str).str.strip() == "").sum())
print(f"Exact duplicate rows      : {n_dup}")
print(f"Missing values per column :\n{missing[missing > 0]}")
print(f"Blank strings hidden in total_charges: {blank_total}")
print(f"Negative tenure values    : {int((df['tenure_months'] < 0).sum())}")
print(f"monthly_charges > 200     : {int((df['monthly_charges'] > 200).sum())}")
print("\nInconsistent labels found:")
print("  contract        :", sorted(df["contract"].unique().tolist()))
print("  gender          :", sorted(df["gender"].unique().tolist()))
print("  internet_service:", sorted(df["internet_service"].dropna().unique().tolist()))
R["audit"] = {
    "duplicates": n_dup,
    "missing": {k: int(v) for k, v in missing[missing > 0].items()},
    "blank_total_charges": blank_total,
    "negative_tenure": int((df["tenure_months"] < 0).sum()),
    "monthly_gt_200": int((df["monthly_charges"] > 200).sum()),
    "contract_labels": int(df["contract"].nunique()),
    "gender_labels": int(df["gender"].nunique()),
    "internet_labels": int(df["internet_service"].nunique()),
}

# Figure 1: missing values
miss_pct = (df.isnull().sum() / len(df) * 100)
miss_pct["total_charges"] = blank_total / len(df) * 100  # blanks hidden as text
miss_pct = miss_pct[miss_pct > 0].sort_values()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(miss_pct.index, miss_pct.values, color=BLUE)
for i, v in enumerate(miss_pct.values):
    ax.text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("% of rows missing")
ax.set_title("Figure 1: Missing values per column (before cleaning)")
plt.tight_layout()
plt.savefig("fig1_missing_values.png")
plt.close()

# =====================================================================
# STEP 3: REMOVE DUPLICATES
# =====================================================================
section("STEP 3: REMOVE DUPLICATE ROWS")
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows before: {before}, after: {len(df)}  (removed {before - len(df)})")
print("Why: duplicates double-count the same customer, which inflates counts,")
print("biases statistics, and (in modelling) leaks identical rows into both")
print("training and test sets, making accuracy look better than it really is.")
R["rows_after_dedup"] = len(df)

# =====================================================================
# STEP 4: FIX DATA TYPES
# =====================================================================
section("STEP 4: FIX DATA TYPES")
print("total_charges dtype before:", df["total_charges"].dtype)
df["total_charges"] = pd.to_numeric(df["total_charges"].astype(str).str.strip(), errors="coerce")
print("total_charges dtype after :", df["total_charges"].dtype)
print(f"Blank strings converted to NaN: {int(df['total_charges'].isnull().sum())}")
print("Why: a numeric column stored as text cannot be averaged, scaled, or")
print("correlated. errors='coerce' turns unparseable blanks into NaN so they")
print("can be handled explicitly with the other missing values.")

# =====================================================================
# STEP 5: STANDARDISE CATEGORICAL LABELS
# =====================================================================
section("STEP 5: STANDARDISE INCONSISTENT CATEGORY LABELS")
contract_map = {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}
df["contract"] = df["contract"].str.strip().str.lower().map(contract_map)
internet_map = {"dsl": "DSL", "fiber optic": "Fiber optic", "no": "No"}
internet_clean = df["internet_service"].str.strip().str.lower().map(internet_map)
df["internet_service"] = internet_clean
gender_map = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
df["gender"] = df["gender"].str.strip().str.lower().map(gender_map)
print("contract        :", sorted(df["contract"].unique().tolist()))
print("gender          :", sorted(df["gender"].unique().tolist()))
print("internet_service:", sorted(df["internet_service"].dropna().unique().tolist()))
print("Why: 'Month-to-month', 'month-to-month' and 'Month-to-month ' would be")
print("treated as three different categories, splitting one real group into")
print("several and distorting both counts and later one-hot encoding.")

# =====================================================================
# STEP 6: INVALID VALUES -> treated as missing
# =====================================================================
section("STEP 6: HANDLE IMPOSSIBLE VALUES (DATA-ENTRY ERRORS)")
# Domain rule: tenure cannot be negative; this company's plans never exceed ~$130/month
bad_t = df["tenure_months"] < 0
bad_m = df["monthly_charges"] > 200
print(f"Negative tenure rows        : {int(bad_t.sum())}")
print(f"monthly_charges > 200 rows  : {int(bad_m.sum())}  values={sorted(df.loc[bad_m, 'monthly_charges'].round(0).tolist())}")

# Figure 2: outliers before treatment
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].boxplot(df["monthly_charges"].dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor="#AED6F1"), flierprops=dict(markerfacecolor=RED, marker="o"))
axes[0].set_title("monthly_charges (before)")
axes[0].set_xlabel("$ per month")
axes[0].set_yticks([])
df.loc[bad_t, "tenure_months"] = np.nan
df.loc[bad_m, "monthly_charges"] = np.nan
axes[1].boxplot(df["monthly_charges"].dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor="#ABEBC6"), flierprops=dict(markerfacecolor=RED, marker="o"))
axes[1].set_title("monthly_charges (after removing errors)")
axes[1].set_xlabel("$ per month")
axes[1].set_yticks([])
fig.suptitle("Figure 2: Outlier detection with boxplots", fontsize=11)
plt.tight_layout()
plt.savefig("fig2_outliers.png")
plt.close()
R["invalid"] = {"negative_tenure": int(bad_t.sum()), "monthly_gt_200": int(bad_m.sum())}
print("Why: a -1 month tenure or a $500+ monthly bill (plans here cost $15-$105) are errors, not rare")
print("but real customers. Capping an error would keep a made-up number, so")
print("they are set to NaN and imputed together with the other missing values.")

# IQR check on what remains
q1, q3 = df["monthly_charges"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_iqr = int(((df["monthly_charges"] < lo) | (df["monthly_charges"] > hi)).sum())
print(f"\nIQR check on monthly_charges (fence {lo:.1f} to {hi:.1f}): {n_iqr} remaining outliers")
sc_q1, sc_q3 = df["support_calls"].quantile([0.25, 0.75])
sc_hi = sc_q3 + 1.5 * (sc_q3 - sc_q1)
n_sc = int((df["support_calls"] > sc_hi).sum())
print(f"support_calls above IQR fence ({sc_hi:.1f}): {n_sc} rows -> KEPT (genuine heavy callers, a real churn signal)")
R["iqr"] = {"monthly_remaining": n_iqr, "support_calls_flagged": n_sc, "support_fence": float(sc_hi)}

# =====================================================================
# STEP 7: HANDLE MISSING VALUES
# =====================================================================
section("STEP 7: HANDLE MISSING VALUES")
print("Missing values before imputation:\n", df.isnull().sum()[df.isnull().sum() > 0])
R["missing_before_impute"] = {k: int(v) for k, v in df.isnull().sum()[df.isnull().sum() > 0].items()}

# 7a. internet_service (categorical) -> explicit 'Unknown' category
df["internet_service"] = df["internet_service"].fillna("Unknown")
print("\n7a. internet_service -> filled with 'Unknown' category")
print("    Why: with a categorical column the mode ('Fiber optic') would silently")
print("    assign ~4% of customers to a plan they may not have. 'Unknown' keeps")
print("    the rows and lets the model learn whether missingness itself matters.")

# 7b. tenure & support_calls: skewed counts -> median
for col in ["tenure_months", "support_calls"]:
    skew = df[col].skew()
    mean_v, med_v = df[col].mean(), df[col].median()
    print(f"\n7b. {col}: skew={skew:.2f}, mean={mean_v:.2f}, median={med_v:.2f}")
    R.setdefault("skew", {})[col] = round(float(skew), 2)
    df[col] = df[col].fillna(med_v)
print("    Why median: both are right-skewed, so the mean is pulled up by a long")
print("    tail of long-tenure customers / heavy callers; the median is robust.")

# 7c. monthly_charges: group-wise median by internet_service
group_med = df.groupby("internet_service")["monthly_charges"].median().round(2)
print("\n7c. monthly_charges median by internet_service:\n", group_med)
R["group_median"] = {k: float(v) for k, v in group_med.items()}
overall_med = round(float(df["monthly_charges"].median()), 2)
R["overall_median_monthly"] = overall_med
df["monthly_charges"] = df["monthly_charges"].fillna(df.groupby("internet_service")["monthly_charges"].transform("median"))
df["monthly_charges"] = df["monthly_charges"].fillna(overall_med)
print(f"    Overall median would be ${overall_med}. Using it for everyone would")
print("    give a Fiber customer a DSL-sized bill (and vice versa), so the fill")
print("    value is taken from customers on the SAME plan instead.")

# 7d. total_charges: derive from tenure x monthly (domain logic)
n_tc = int(df["total_charges"].isnull().sum())
df["total_charges"] = df["total_charges"].fillna((df["tenure_months"] * df["monthly_charges"]).round(2))
print(f"\n7d. total_charges: {n_tc} missing values rebuilt as tenure x monthly_charges")
print("    Why: total charges is (approximately) tenure x monthly price, so a")
print("    calculated value is more accurate than any statistical average.")
print("\nMissing values after imputation:", int(df.isnull().sum().sum()))
R["missing_after"] = int(df.isnull().sum().sum())
df["tenure_months"] = df["tenure_months"].astype(int)
df["support_calls"] = df["support_calls"].astype(int)

clean = df.copy()
clean.to_csv("cleaned_customer_churn.csv", index=False)
R["clean_shape"] = list(clean.shape)

# =====================================================================
# STEP 8: EXPLORATORY DATA ANALYSIS (on cleaned data)
# =====================================================================
section("STEP 8: EXPLORATORY DATA ANALYSIS")
churn_rate = (clean["Churn"] == "Yes").mean()
print(f"Overall churn rate: {churn_rate:.1%}")
R["churn_rate"] = round(float(churn_rate) * 100, 1)

# Figure 3: distributions
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, col, color in zip(axes.ravel(),
                           ["tenure_months", "monthly_charges", "total_charges", "support_calls"],
                           [BLUE, RED, GREY, "#28B463"]):
    ax.hist(clean[col], bins=25 if col != "support_calls" else range(0, clean[col].max() + 2),
            color=color, edgecolor="white")
    ax.axvline(clean[col].mean(), color="black", ls="--", lw=1, label=f"mean {clean[col].mean():.1f}")
    ax.axvline(clean[col].median(), color="orange", ls="-", lw=1.5, label=f"median {clean[col].median():.1f}")
    ax.set_title(col)
    ax.legend(fontsize=8)
fig.suptitle("Figure 3: Distributions of numeric features (after cleaning)", fontsize=11)
plt.tight_layout()
plt.savefig("fig3_distributions.png")
plt.close()
R["numeric_desc"] = clean[["tenure_months", "monthly_charges", "total_charges", "support_calls"]].describe().round(2).to_dict()

# Churn rate by category
def churn_by(col):
    return clean.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).round(1)

by_contract = churn_by("contract")
by_internet = churn_by("internet_service")
by_payment = churn_by("payment_method")
by_calls = clean.assign(calls=clean["support_calls"].clip(upper=4)).groupby("calls")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).round(1)
by_gender = churn_by("gender")
print("\nChurn rate (%) by contract:\n", by_contract)
print("\nChurn rate (%) by internet service:\n", by_internet)
print("\nChurn rate (%) by payment method:\n", by_payment)
print("\nChurn rate (%) by support calls (4 = 4 or more):\n", by_calls)
print("\nChurn rate (%) by gender:\n", by_gender)
R["by_contract"] = by_contract.to_dict()
R["by_internet"] = by_internet.to_dict()
R["by_payment"] = by_payment.to_dict()
R["by_calls"] = {str(k): float(v) for k, v in by_calls.items()}
R["by_gender"] = by_gender.to_dict()

# Figure 4: churn by category
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, ser, title in zip(axes, [by_contract, by_internet, by_calls],
                          ["Contract type", "Internet service", "Support calls (4 = 4+)"]):
    bars = ax.bar([str(i) for i in ser.index], ser.values, color=[BLUE, RED, GREY, "#28B463", "#F5B041"][:len(ser)])
    ax.axhline(churn_rate * 100, color="black", ls="--", lw=1)
    for b, v in zip(bars, ser.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel("Churn rate (%)")
fig.suptitle(f"Figure 4: Churn rate by segment (dashed line = overall {churn_rate:.0%})", fontsize=11)
plt.tight_layout()
plt.savefig("fig4_churn_by_segment.png")
plt.close()

# Tenure bands
bands = pd.cut(clean["tenure_months"], [0, 12, 24, 48, 100], labels=["0-12", "13-24", "25-48", "49+"])
by_tenure = clean.groupby(bands, observed=True)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).round(1)
print("\nChurn rate (%) by tenure band (months):\n", by_tenure)
R["by_tenure"] = {str(k): float(v) for k, v in by_tenure.items()}

# =====================================================================
# STEP 9: FEATURE SELECTION
# =====================================================================
section("STEP 9: FEATURE SELECTION")
y_bin = (clean["Churn"] == "Yes").astype(int)
chi_results = {}
for col in ["gender", "contract", "internet_service", "payment_method", "SeniorCitizen"]:
    ct = pd.crosstab(clean[col], y_bin)
    chi2, p, _, _ = chi2_contingency(ct)
    chi_results[col] = round(float(p), 4)
    verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"Chi-square test, {col:17s} vs Churn: p = {p:.4f}  ({verdict})")
R["chi2_p"] = chi_results

corr_tt = clean["total_charges"].corr(clean["tenure_months"])
corr_tm = clean["total_charges"].corr(clean["monthly_charges"])
print(f"\nCorrelation total_charges vs tenure_months   : {corr_tt:.2f}")
print(f"Correlation total_charges vs monthly_charges : {corr_tm:.2f}")
corr_tp = clean["total_charges"].corr(clean["tenure_months"] * clean["monthly_charges"])
print(f"Correlation total_charges vs (tenure_months x monthly_charges): {corr_tp:.3f}")
R["corr_total_product"] = round(float(corr_tp), 3)
R["corr_total_tenure"] = round(float(corr_tt), 2)
R["corr_total_monthly"] = round(float(corr_tm), 2)

drop_cols = ["customerID", "gender", "total_charges"]
print(f"\nColumns dropped: {drop_cols}")
print("  customerID   -> unique identifier, no predictive meaning (a model could memorise it).")
print(f"  gender       -> not significant at the 5% level (p = {chi_results['gender']:.3f}) AND a protected attribute:")
print("                  dropping it avoids a model that treats customers differently by gender.")
print("  total_charges-> almost fully explained by tenure x monthly_charges (redundant,")
print("                  causes multicollinearity), so the two parent features are kept instead.")
print("\nColumns KEPT despite a non-significant chi-square result:")
print(f"  SeniorCitizen  (p = {chi_results['SeniorCitizen']:.3f}) and payment_method (p = {chi_results['payment_method']:.3f})")
print("  Why kept: with only ~1,200 rows a non-significant test is weak evidence of 'no effect',")
print("  and both are plausible business drivers (electronic-check customers often churn more).")
print("  Keeping them costs little: if they truly do not matter, the model gives them")
print("  coefficients near zero. They are re-checked through the model coefficients in Step 12.")
feat = clean.drop(columns=drop_cols)

# =====================================================================
# STEP 10: ENCODING CATEGORICAL VARIABLES
# =====================================================================
section("STEP 10: ENCODE CATEGORICAL VARIABLES")
feat["Churn"] = (feat["Churn"] == "Yes").astype(int)
print("Churn: Yes/No -> 1/0 (binary target)")
feat["contract"] = feat["contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})
print("contract: ORDINAL encoding 0/1/2 (commitment length is naturally ordered, and")
print("          the churn table above shows churn falls steadily along that order).")
feat = pd.get_dummies(feat, columns=["internet_service", "payment_method"], drop_first=True, dtype=int)
print("internet_service, payment_method: ONE-HOT encoding (nominal, no order). drop_first=True")
print("          removes one column per feature to avoid the dummy-variable trap.")
print("\nColumns after encoding:", list(feat.columns))
R["encoded_columns"] = list(feat.columns)

# =====================================================================
# STEP 11: NORMALISATION / SCALING (with NumPy check)
# =====================================================================
section("STEP 11: FEATURE SCALING")
num_cols = ["tenure_months", "monthly_charges", "support_calls"]
before_stats = feat[num_cols].agg(["min", "max", "mean", "std"]).round(2)
print("Before scaling:\n", before_stats)

# NumPy by hand, to show what a scaler actually does
x = feat["monthly_charges"].to_numpy()
manual_minmax = (x - x.min()) / (x.max() - x.min())
scaler = MinMaxScaler()
sk_minmax = scaler.fit_transform(feat[["monthly_charges"]]).ravel()
print("\nNumPy manual Min-Max equals scikit-learn MinMaxScaler:", bool(np.allclose(manual_minmax, sk_minmax)))
z_manual = (x - x.mean()) / x.std()
z_sk = StandardScaler().fit_transform(feat[["monthly_charges"]]).ravel()
print("NumPy manual z-score equals scikit-learn StandardScaler:", bool(np.allclose(z_manual, z_sk)))

final = feat.copy()
final[num_cols] = scaler.fit_transform(final[num_cols])
after_stats = final[num_cols].agg(["min", "max", "mean", "std"]).round(2)
print("\nAfter Min-Max scaling:\n", after_stats)
print("\nWhy Min-Max: tenure (1-72), charges (15-101) and support calls (0-6) sit on")
print("very different scales, which would let large-valued columns dominate any")
print("distance- or gradient-based model. Min-Max maps all to 0-1 while keeping the")
print("shape of each distribution, and it is safe here because the extreme errors")
print("were already removed in Step 6 (Min-Max is sensitive to outliers).")
R["scaling_before"] = before_stats.to_dict()
R["scaling_after"] = after_stats.to_dict()
R["minmax_numpy_match"] = bool(np.allclose(manual_minmax, sk_minmax))

# Figure 5: before/after scaling
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].boxplot([feat[c] for c in num_cols], tick_labels=["tenure", "monthly", "calls"], patch_artist=True,
                boxprops=dict(facecolor="#AED6F1"))
axes[0].set_title("Before scaling (different ranges)")
axes[1].boxplot([final[c] for c in num_cols], tick_labels=["tenure", "monthly", "calls"], patch_artist=True,
                boxprops=dict(facecolor="#ABEBC6"))
axes[1].set_title("After Min-Max scaling (all 0 to 1)")
fig.suptitle("Figure 5: Effect of feature scaling", fontsize=11)
plt.tight_layout()
plt.savefig("fig5_scaling.png")
plt.close()

final.to_csv("preprocessed_customer_churn.csv", index=False)
R["final_shape"] = list(final.shape)
print(f"\nFinal preprocessed dataset: {final.shape[0]} rows x {final.shape[1]} columns")

# =====================================================================
# STEP 12: CORRELATION HEATMAP + BASELINE MODEL
# =====================================================================
section("STEP 12: CORRELATION ANALYSIS & BASELINE ML MODEL")
corr = final.corr()
corr_with_churn = corr["Churn"].drop("Churn").sort_values()
print("Correlation of each feature with Churn:\n", corr_with_churn.round(3))
R["corr_with_churn"] = corr_with_churn.round(3).to_dict()

fig, ax = plt.subplots(figsize=(8.5, 6.8))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5)
plt.colorbar(im, fraction=0.04)
ax.grid(False)
ax.set_title("Figure 6: Correlation heatmap of the preprocessed dataset")
plt.tight_layout()
plt.savefig("fig6_correlation_heatmap.png")
plt.close()

# Baseline model: shows what preprocessing makes possible
X = final.drop(columns=["Churn"])
y = final["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)
acc = accuracy_score(y_te, model.predict(X_te))
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
majority = max(y_te.mean(), 1 - y_te.mean())
print(f"\nBaseline Logistic Regression: accuracy={acc:.3f}, ROC-AUC={auc:.3f}")
print(f"Majority-class baseline (always predict 'No'): accuracy={majority:.3f}")
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("\nModel coefficients (sign shows direction of effect on churn):\n", coefs.round(2))
R["baseline"] = {"accuracy": round(float(acc), 3), "roc_auc": round(float(auc), 3),
                 "majority_accuracy": round(float(majority), 3), "n_train": len(X_tr), "n_test": len(X_te)}
R["coefs"] = coefs.round(2).to_dict()

with open("results.json", "w") as f:
    json.dump(R, f, indent=2, default=str)
print("\nDone. Saved: cleaned_customer_churn.csv, preprocessed_customer_churn.csv, figures, results.json")
