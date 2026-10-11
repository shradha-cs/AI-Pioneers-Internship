"""
Week 2 Task: Supervised Machine Learning Models

Implements Linear Regression, Logistic Regression, Decision Trees, Random
Forest and K-Nearest Neighbors with Scikit-learn, evaluates them with
performance metrics, and runs extra experiments to explain WHY each model
performs the way it does.

  Part A  Regression      : Diabetes dataset (predict disease progression)
  Part B  Classification  : Breast Cancer dataset (malignant vs benign)
  Part C  Why did Logistic Regression do so well? (experiments)
  Part D  Contrast dataset: two moons (a non-linear problem)

Run:  python week2_supervised_models.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_diabetes, load_breast_cancer, make_moons
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold,
    StratifiedKFold, cross_validate, cross_val_predict, learning_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, make_scorer, ConfusionMatrixDisplay,
)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
SEED = 42
COLORS = {"Logistic Regression": "#2E86C1", "Linear Regression": "#2E86C1",
          "Decision Tree": "#C0392B", "Random Forest": "#28B463", "KNN": "#F39C12"}
R = {}


def section(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


# =====================================================================
# PART A: REGRESSION (Diabetes)
# =====================================================================
section("PART A: REGRESSION - Diabetes dataset")
Xr, yr = load_diabetes(return_X_y=True, as_frame=True)
print(f"Samples: {Xr.shape[0]}, features: {Xr.shape[1]} -> {list(Xr.columns)}")
print(f"Target (disease progression after 1 year): mean={yr.mean():.1f}, std={yr.std():.1f}, range={yr.min():.0f}-{yr.max():.0f}")
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)
print(f"Train: {len(Xr_tr)}, Test: {len(Xr_te)}")
R["reg_data"] = {"n": int(Xr.shape[0]), "features": int(Xr.shape[1]), "y_mean": round(float(yr.mean()), 1),
                 "y_std": round(float(yr.std()), 1), "n_train": len(Xr_tr), "n_test": len(Xr_te)}

# Hyperparameters chosen by 5-fold CV on the TRAINING set only (test set untouched)
gs_dt = GridSearchCV(DecisionTreeRegressor(random_state=SEED), {"max_depth": list(range(1, 11))}, cv=5, scoring="r2").fit(Xr_tr, yr_tr)
gs_knn = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                      {"kneighborsregressor__n_neighbors": list(range(1, 41))}, cv=5, scoring="r2").fit(Xr_tr, yr_tr)
best_depth_r = gs_dt.best_params_["max_depth"]
best_k_r = gs_knn.best_params_["kneighborsregressor__n_neighbors"]
print(f"\nCV-selected Decision Tree depth: {best_depth_r} (CV R2={gs_dt.best_score_:.3f})")
print(f"CV-selected KNN k              : {best_k_r} (CV R2={gs_knn.best_score_:.3f})")
R["reg_best"] = {"depth": int(best_depth_r), "k": int(best_k_r)}

reg_models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Decision Tree (default)": DecisionTreeRegressor(random_state=SEED),
    f"Decision Tree (depth={best_depth_r})": DecisionTreeRegressor(max_depth=best_depth_r, random_state=SEED),
    "Random Forest (100 trees)": RandomForestRegressor(n_estimators=100, random_state=SEED),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    f"KNN (k={best_k_r})": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best_k_r)),
    "Baseline (predict the mean)": DummyRegressor(strategy="mean"),
}
cv_r = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
reg_rows = []
reg_fitted = {}
for name, m in reg_models.items():
    m.fit(Xr_tr, yr_tr)
    reg_fitted[name] = m
    p_te, p_tr = m.predict(Xr_te), m.predict(Xr_tr)
    mse = mean_squared_error(yr_te, p_te)
    cv = cross_validate(m, Xr, yr, cv=cv_r, scoring="r2")
    reg_rows.append({
        "Model": name, "Train R2": round(r2_score(yr_tr, p_tr), 3), "Test R2": round(r2_score(yr_te, p_te), 3),
        "MSE": round(mse, 1), "RMSE": round(np.sqrt(mse), 2), "MAE": round(mean_absolute_error(yr_te, p_te), 2),
        "CV R2 mean": round(cv["test_score"].mean(), 3), "CV R2 std": round(cv["test_score"].std(), 3),
    })
reg_df = pd.DataFrame(reg_rows)
print("\nRegression results (test set + 25-fold repeated CV):")
print(reg_df.to_string(index=False))
R["reg_table"] = reg_rows

# Linear regression coefficients (features are standardised, so comparable)
lin = reg_fitted["Linear Regression"]
coef = pd.Series(lin[-1].coef_, index=Xr.columns).sort_values()
print("\nLinear Regression standardised coefficients:\n", coef.round(1))
R["reg_coef"] = coef.round(1).to_dict()
resid = yr_te - lin.predict(Xr_te)
print(f"\nLinear Regression residuals: mean={resid.mean():.2f}, std={resid.std():.2f}")
print(f"Residual std ({resid.std():.1f}) vs target std ({yr.std():.1f}): the model removes only part of the spread.")
R["resid_std"] = round(float(resid.std()), 1)

# Multicollinearity check: Variance Inflation Factor (VIF > 10 is the usual warning level)
Zr = (Xr - Xr.mean()) / Xr.std()
vif = pd.Series(np.diag(np.linalg.inv(np.corrcoef(Zr.values.T))), index=Xr.columns).round(1)
print("\nVariance Inflation Factors (multicollinearity check):\n", vif.sort_values(ascending=False).to_string())
print(f"Correlation between s1 and s2: {Xr['s1'].corr(Xr['s2']):.3f}")
R["vif"] = vif.to_dict()
R["corr_s1_s2"] = round(float(Xr["s1"].corr(Xr["s2"])), 3)

# Figure 1: predicted vs actual
show = ["Linear Regression", f"Decision Tree (depth={best_depth_r})", "Random Forest (100 trees)", f"KNN (k={best_k_r})"]
fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
for ax, name in zip(axes.ravel(), show):
    pred = reg_fitted[name].predict(Xr_te)
    ax.scatter(yr_te, pred, s=22, alpha=0.7, color="#2E86C1", edgecolor="white", linewidth=0.3)
    lim = [min(yr_te.min(), pred.min()) - 5, max(yr_te.max(), pred.max()) + 5]
    ax.plot(lim, lim, "k--", lw=1)
    r2 = r2_score(yr_te, pred)
    ax.set_title(f"{name}\nTest R2 = {r2:.3f}", fontsize=10)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
fig.suptitle("Figure 1: Predicted vs actual (dashed line = perfect prediction)", fontsize=11)
plt.tight_layout(); plt.savefig("fig1_regression_pred_vs_actual.png"); plt.close()

# Figure 2: residuals + coefficients
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(lin.predict(Xr_te), resid, s=22, alpha=0.7, color="#C0392B", edgecolor="white", linewidth=0.3)
axes[0].axhline(0, color="k", lw=1)
axes[0].set_xlabel("Predicted value"); axes[0].set_ylabel("Residual (actual - predicted)")
axes[0].set_title("Linear Regression residuals")
axes[1].barh(coef.index, coef.values, color=["#C0392B" if v < 0 else "#2E86C1" for v in coef.values])
axes[1].set_xlabel("Standardised coefficient"); axes[1].set_title("What drives the prediction")
fig.suptitle("Figure 2: Linear Regression diagnostics", fontsize=11)
plt.tight_layout(); plt.savefig("fig2_regression_diagnostics.png"); plt.close()

# =====================================================================
# PART B: CLASSIFICATION (Breast Cancer)
# =====================================================================
section("PART B: CLASSIFICATION - Breast Cancer dataset")
bc = load_breast_cancer(as_frame=True)
Xc, yc = bc.data, bc.target          # 0 = malignant, 1 = benign
print(f"Samples: {Xc.shape[0]}, features: {Xc.shape[1]}")
print(f"Class counts: malignant={int((yc == 0).sum())}, benign={int((yc == 1).sum())}")
print("Malignant is treated as the POSITIVE class in precision / recall / F1, because a")
print("missed malignant tumour (false negative) is the costly mistake.")
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=SEED, stratify=yc)
print(f"Train: {len(Xc_tr)}, Test: {len(Xc_te)}")
R["cls_data"] = {"n": int(Xc.shape[0]), "features": int(Xc.shape[1]), "malignant": int((yc == 0).sum()),
                 "benign": int((yc == 1).sum()), "n_train": len(Xc_tr), "n_test": len(Xc_te)}

skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
gs_dtc = GridSearchCV(DecisionTreeClassifier(random_state=SEED), {"max_depth": list(range(1, 11))}, cv=skf5, scoring="accuracy").fit(Xc_tr, yc_tr)
gs_knnc = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),
                       {"kneighborsclassifier__n_neighbors": list(range(1, 31))}, cv=skf5, scoring="accuracy").fit(Xc_tr, yc_tr)
best_depth_c = gs_dtc.best_params_["max_depth"]
best_k_c = gs_knnc.best_params_["kneighborsclassifier__n_neighbors"]
print(f"\nCV-selected Decision Tree depth: {best_depth_c} (CV accuracy={gs_dtc.best_score_:.3f})")
print(f"CV-selected KNN k              : {best_k_c} (CV accuracy={gs_knnc.best_score_:.3f})")
R["cls_best"] = {"depth": int(best_depth_c), "k": int(best_k_c)}

cls_models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree (default)": DecisionTreeClassifier(random_state=SEED),
    f"Decision Tree (depth={best_depth_c})": DecisionTreeClassifier(max_depth=best_depth_c, random_state=SEED),
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    f"KNN (k={best_k_c})": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k_c)),
    "Baseline (always benign)": DummyClassifier(strategy="most_frequent"),
}
sc = {
    "acc": "accuracy",
    "rec_m": make_scorer(recall_score, pos_label=0),
    "f1_m": make_scorer(f1_score, pos_label=0),
    "auc": "roc_auc",
}
cv_c = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
cls_rows, cls_fit, cls_cv = [], {}, {}
for name, m in cls_models.items():
    m.fit(Xc_tr, yc_tr)
    cls_fit[name] = m
    pred = m.predict(Xc_te)
    proba = m.predict_proba(Xc_te)[:, 1]
    cm = confusion_matrix(yc_te, pred, labels=[0, 1])      # rows: true malignant, benign
    fn_m, fp_m = int(cm[0, 1]), int(cm[1, 0])             # FN = malignant predicted benign
    cv = cross_validate(m, Xc, yc, cv=cv_c, scoring=sc)
    cls_cv[name] = cv
    cls_rows.append({
        "Model": name,
        "Test Acc": round(accuracy_score(yc_te, pred), 4),
        "Precision": round(precision_score(yc_te, pred, pos_label=0, zero_division=0), 4),
        "Recall": round(recall_score(yc_te, pred, pos_label=0), 4),
        "F1": round(f1_score(yc_te, pred, pos_label=0), 4),
        "ROC-AUC": round(roc_auc_score(yc_te, proba), 4),
        "FN": fn_m, "FP": fp_m,
        "Train Acc": round(accuracy_score(yc_tr, m.predict(Xc_tr)), 4),
        "CV Acc mean": round(cv["test_acc"].mean(), 4), "CV Acc std": round(cv["test_acc"].std(), 4),
        "CV Recall(M)": round(cv["test_rec_m"].mean(), 4), "CV AUC": round(cv["test_auc"].mean(), 4),
    })
cls_df = pd.DataFrame(cls_rows)
print("\nClassification results (positive class = malignant):")
print(cls_df[["Model", "Test Acc", "Precision", "Recall", "F1", "ROC-AUC", "FN", "FP"]].to_string(index=False))
print("\nOverfitting check and 50-fold repeated CV (5-fold x 10 repeats):")
print(cls_df[["Model", "Train Acc", "Test Acc", "CV Acc mean", "CV Acc std", "CV Recall(M)", "CV AUC"]].to_string(index=False))
R["cls_table"] = cls_rows

# Figure 3: confusion matrices
show_c = ["Logistic Regression", f"Decision Tree (depth={best_depth_c})", "Random Forest (100 trees)", f"KNN (k={best_k_c})"]
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, name in zip(axes.ravel(), show_c):
    cm = confusion_matrix(yc_te, cls_fit[name].predict(Xc_te), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAcc = {(cm[0,0]+cm[1,1])/cm.sum():.3f}", fontsize=10)
    ax.grid(False)
fig.suptitle("Figure 3: Confusion matrices on the test set", fontsize=11)
plt.tight_layout(); plt.savefig("fig3_confusion_matrices.png"); plt.close()

# Figure 4: ROC curves + CV accuracy spread
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
for name in show_c:
    fpr, tpr, _ = roc_curve(yc_te, cls_fit[name].predict_proba(Xc_te)[:, 1])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yc_te, cls_fit[name].predict_proba(Xc_te)[:,1]):.3f})", lw=1.8)
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves (test set)"); axes[0].legend(fontsize=7.5, loc="lower right")
data = [cls_cv[n]["test_acc"] for n in show_c]
bp = axes[1].boxplot(data, tick_labels=["LogReg", f"Tree\nd={best_depth_c}", "Forest", f"KNN\nk={best_k_c}"], patch_artist=True)
for patch, c in zip(bp["boxes"], ["#AED6F1", "#F5B7B1", "#ABEBC6", "#FAD7A0"]):
    patch.set_facecolor(c)
axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy across 50 CV folds")
fig.suptitle("Figure 4: ROC curves and cross-validated accuracy spread", fontsize=11)
plt.tight_layout(); plt.savefig("fig4_roc_and_cv.png"); plt.close()

# =====================================================================
# PART C: WHY DID LOGISTIC REGRESSION DO SO WELL?
# =====================================================================
section("PART C: WHY DOES LOGISTIC REGRESSION DO SO WELL? (experiments)")

# C1. Is the gap between LR and RF real, or just noise from one small test split?
lr_s = cls_cv["Logistic Regression"]["test_acc"]
rf_s = cls_cv["Random Forest (100 trees)"]["test_acc"]
diff = lr_s - rf_s
k = len(diff)
nb_var = diff.var(ddof=1) * (1 / k + 0.25)   # Nadeau-Bengio corrected variance (test/train = 1/4)
t_stat = diff.mean() / np.sqrt(nb_var)
p_val = 2 * stats.t.sf(abs(t_stat), df=k - 1)
wins, ties = int((diff > 0).sum()), int((diff == 0).sum())
print("C1. Is the LR vs RF gap real?")
print(f"  Single test split : LR={cls_df.loc[0,'Test Acc']:.4f}  RF={cls_df.loc[3,'Test Acc']:.4f}  (difference of {int(round((cls_df.loc[0,'Test Acc']-cls_df.loc[3,'Test Acc'])*len(yc_te)))} test samples out of {len(yc_te)})")
print(f"  50-fold repeated CV: LR={lr_s.mean():.4f} +/- {lr_s.std():.4f}, RF={rf_s.mean():.4f} +/- {rf_s.std():.4f}")
print(f"  Mean difference (LR - RF) = {diff.mean():+.4f}; LR better in {wins}/{k} folds, tied in {ties}, RF better in {k - wins - ties}")
print(f"  Corrected resampled t-test (Nadeau & Bengio): t = {t_stat:.2f}, p = {p_val:.3f}")
R["gap"] = {"lr_mean": round(float(lr_s.mean()), 4), "lr_std": round(float(lr_s.std()), 4),
            "rf_mean": round(float(rf_s.mean()), 4), "rf_std": round(float(rf_s.std()), 4),
            "diff": round(float(diff.mean()), 4), "wins": wins, "ties": ties, "folds": k,
            "t": round(float(t_stat), 2), "p": round(float(p_val), 3)}

# C2. Out-of-fold errors: do the models fail on the same patients?
oof = {}
for name in ["Logistic Regression", "Decision Tree (default)", "Random Forest (100 trees)", "KNN (k=5)"]:
    oof[name] = cross_val_predict(cls_models[name], Xc, yc, cv=skf5)
err = {n: (p != yc.values) for n, p in oof.items()}
print("\nC2. Out-of-fold errors over all 569 samples (5-fold):")
for n, e in err.items():
    print(f"  {n:28s}: {int(e.sum()):3d} errors ({e.mean():.1%})")
both = int((err["Logistic Regression"] & err["Random Forest (100 trees)"]).sum())
only_lr = int((err["Logistic Regression"] & ~err["Random Forest (100 trees)"]).sum())
only_rf = int((~err["Logistic Regression"] & err["Random Forest (100 trees)"]).sum())
print(f"  Wrong in BOTH LR and RF: {both}; only LR wrong: {only_lr}; only RF wrong: {only_rf}")
R["oof"] = {n: int(e.sum()) for n, e in err.items()}
R["oof_overlap"] = {"both": both, "only_lr": only_lr, "only_rf": only_rf}

# C3. Feature redundancy
corr = Xc.corr().abs().values
iu = np.triu_indices_from(corr, k=1)
n_pairs = len(iu[0]); n_high = int((corr[iu] > 0.9).sum())
print(f"\nC3. Feature redundancy: {n_high} of {n_pairs} feature pairs have |correlation| > 0.9")
size_words = ("radius", "perimeter", "area")
n_size_pairs = sum(1 for i, j in zip(*iu) if corr[i, j] > 0.9
                   and any(w in Xc.columns[i] for w in size_words) and any(w in Xc.columns[j] for w in size_words))
print(f"    of which {n_size_pairs} pairs are both size measurements (radius / perimeter / area)")
R["corr_pairs"] = {"high": n_high, "total": n_pairs, "size_pairs": n_size_pairs}

# C4. Learning curves: does the winner depend on how much data there is?
sizes = np.linspace(0.1, 1.0, 8)
lc = {}
for name, m in [("Logistic Regression", cls_models["Logistic Regression"]),
                ("Decision Tree", DecisionTreeClassifier(random_state=SEED)),
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED)),
                ("KNN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)))]:
    ts, tr_s, va_s = learning_curve(m, Xc, yc, train_sizes=sizes, cv=skf5, scoring="accuracy", random_state=SEED, shuffle=True)
    lc[name] = (ts, tr_s.mean(1), va_s.mean(1))
print("\nC4. Validation accuracy as training data grows (training samples -> accuracy):")
print("  " + "samples".ljust(8) + "".join(n[:14].rjust(16) for n in lc))
for i in range(len(sizes)):
    print("  " + str(lc["Logistic Regression"][0][i]).ljust(8) + "".join(f"{lc[n][2][i]:16.3f}" for n in lc))
R["lc"] = {n: {"sizes": [int(x) for x in v[0]], "val": [round(float(x), 4) for x in v[2]], "train": [round(float(x), 4) for x in v[1]]} for n, v in lc.items()}

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
for name, (ts, tr, va) in lc.items():
    c = COLORS[name]
    ax[0].plot(ts, va, marker="o", ms=4, color=c, label=name)
    ax[1].plot(ts, tr, marker="o", ms=4, color=c, label=name)
ax[0].set_title("Validation accuracy (unseen data)"); ax[1].set_title("Training accuracy (seen data)")
for a in ax:
    a.set_xlabel("Number of training samples"); a.legend(fontsize=8)
ax[0].set_ylabel("Accuracy")
fig.suptitle("Figure 5: Learning curves - how each model uses more data", fontsize=11)
plt.tight_layout(); plt.savefig("fig5_learning_curves.png"); plt.close()

# C5. Feature importance (RF) vs coefficient size (LR)
rf_full = cls_fit["Random Forest (100 trees)"]
imp = pd.Series(rf_full.feature_importances_, index=Xc.columns).sort_values(ascending=False)
lr_coef = pd.Series(np.abs(cls_fit["Logistic Regression"][-1].coef_[0]), index=Xc.columns).sort_values(ascending=False)
top_rf, top_lr = list(imp.index[:10]), list(lr_coef.index[:10])
overlap = len(set(top_rf) & set(top_lr))
print(f"\nC5. Top-10 features: Random Forest vs Logistic Regression overlap = {overlap}/10")
print("  RF top 5:", ", ".join(f"{n} ({v:.3f})" for n, v in imp.head(5).items()))
print("  LR top 5:", ", ".join(f"{n} ({v:.2f})" for n, v in lr_coef.head(5).items()))
print(f"  RF: top 5 features hold {imp.head(5).sum():.1%} of total importance")
rank_rf = {f: i + 1 for i, f in enumerate(imp.index)}
print(f"  RF rank of 'worst texture' = {rank_rf['worst texture']}, of 'radius error' = {rank_rf['radius error']} (out of 30)")
R["importance"] = {"rf_rank_worst_texture": rank_rf["worst texture"], "rf_rank_radius_error": rank_rf["radius error"], "rf_top5": {n: round(float(v), 3) for n, v in imp.head(5).items()},
                   "lr_top5": {n: round(float(v), 2) for n, v in lr_coef.head(5).items()},
                   "overlap10": overlap, "rf_top5_share": round(float(imp.head(5).sum()), 3)}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
imp.head(10)[::-1].plot.barh(ax=ax[0], color="#28B463"); ax[0].set_title("Random Forest importance (top 10)")
lr_coef.head(10)[::-1].plot.barh(ax=ax[1], color="#2E86C1"); ax[1].set_title("Logistic Regression |coefficient| (top 10)")
fig.suptitle(f"Figure 6: Which features each model relies on (top-10 overlap: {overlap}/10)", fontsize=11)
plt.tight_layout(); plt.savefig("fig6_feature_importance.png"); plt.close()

# C6. Decision tree: overfitting curve + illustration
depths = list(range(1, 16))
tr_acc, cv_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=SEED)
    cvd = cross_validate(m, Xc_tr, yc_tr, cv=skf5, scoring="accuracy", return_train_score=True)
    tr_acc.append(cvd["train_score"].mean()); cv_acc.append(cvd["test_score"].mean())
print("\nC6. Decision Tree depth vs accuracy (5-fold CV on the training set):")
for d, a, b in zip(depths[:8], tr_acc[:8], cv_acc[:8]):
    print(f"  depth {d:2d}: train={a:.3f}  validation={b:.3f}")
full_tree = DecisionTreeClassifier(random_state=SEED).fit(Xc_tr, yc_tr)
print(f"  Unrestricted tree: depth={full_tree.get_depth()}, leaves={full_tree.get_n_leaves()}, train accuracy={full_tree.score(Xc_tr, yc_tr):.3f}, test accuracy={full_tree.score(Xc_te, yc_te):.3f}")
R["tree"] = {"depths": depths, "train": [round(float(x), 4) for x in tr_acc], "val": [round(float(x), 4) for x in cv_acc],
             "full_depth": int(full_tree.get_depth()), "full_leaves": int(full_tree.get_n_leaves()),
             "full_train": round(float(full_tree.score(Xc_tr, yc_tr)), 3), "full_test": round(float(full_tree.score(Xc_te, yc_te)), 3)}

# C7. KNN: effect of k and of scaling
ks = list(range(1, 31))
k_scaled, k_raw = [], []
for kk in ks:
    k_scaled.append(cross_validate(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=kk)), Xc_tr, yc_tr, cv=skf5)["test_score"].mean())
    k_raw.append(cross_validate(KNeighborsClassifier(n_neighbors=kk), Xc_tr, yc_tr, cv=skf5)["test_score"].mean())
print("\nC7. KNN: validation accuracy with and without feature scaling")
print(f"  Best scaled   : k={ks[int(np.argmax(k_scaled))]}, accuracy={max(k_scaled):.3f}")
print(f"  Best unscaled : k={ks[int(np.argmax(k_raw))]}, accuracy={max(k_raw):.3f}")
print(f"  At k=5        : scaled={k_scaled[4]:.3f}, unscaled={k_raw[4]:.3f}")
rng_ = Xc.max() - Xc.min()
print(f"  Feature ranges differ by a factor of {rng_.max() / rng_.min():,.0f} ('{rng_.idxmax()}' vs '{rng_.idxmin()}')")
R["knn"] = {"best_scaled_k": ks[int(np.argmax(k_scaled))], "best_scaled": round(float(max(k_scaled)), 4),
            "best_raw_k": ks[int(np.argmax(k_raw))], "best_raw": round(float(max(k_raw)), 4),
            "k5_scaled": round(float(k_scaled[4]), 4), "k5_raw": round(float(k_raw[4]), 4),
            "range_ratio": round(float(rng_.max() / rng_.min())), "range_max": rng_.idxmax(), "range_min": rng_.idxmin(),
            "ks": ks, "scaled": [round(float(x), 4) for x in k_scaled], "raw": [round(float(x), 4) for x in k_raw]}

# regression tuning curves for the figure
dr, kr = list(range(1, 11)), list(range(1, 41))
dt_r2 = [cross_validate(DecisionTreeRegressor(max_depth=d, random_state=SEED), Xr_tr, yr_tr, cv=5, scoring="r2", return_train_score=True) for d in dr]
knn_r2 = [cross_validate(make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=kk)), Xr_tr, yr_tr, cv=5, scoring="r2") for kk in kr]
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
ax[0, 0].plot(depths, tr_acc, marker="o", ms=4, color="#C0392B", label="Training"); ax[0, 0].plot(depths, cv_acc, marker="o", ms=4, color="#2E86C1", label="Validation")
ax[0, 0].set_title("Classification: Decision Tree depth"); ax[0, 0].set_xlabel("max_depth"); ax[0, 0].set_ylabel("Accuracy"); ax[0, 0].legend()
ax[0, 1].plot(ks, k_scaled, marker="o", ms=3, color="#2E86C1", label="Scaled"); ax[0, 1].plot(ks, k_raw, marker="o", ms=3, color="#C0392B", label="Unscaled")
ax[0, 1].set_title("Classification: KNN, effect of k and scaling"); ax[0, 1].set_xlabel("k (neighbours)"); ax[0, 1].legend()
ax[1, 0].plot(dr, [np.mean(x["train_score"]) for x in dt_r2], marker="o", ms=4, color="#C0392B", label="Training")
ax[1, 0].plot(dr, [np.mean(x["test_score"]) for x in dt_r2], marker="o", ms=4, color="#2E86C1", label="Validation")
ax[1, 0].set_title("Regression: Decision Tree depth"); ax[1, 0].set_xlabel("max_depth"); ax[1, 0].set_ylabel("R2"); ax[1, 0].legend()
ax[1, 1].plot(kr, [np.mean(x["test_score"]) for x in knn_r2], marker="o", ms=3, color="#2E86C1")
ax[1, 1].set_title("Regression: KNN, effect of k (scaled)"); ax[1, 1].set_xlabel("k (neighbours)"); ax[1, 1].set_ylabel("R2")
fig.suptitle("Figure 7: Choosing hyperparameters (bias-variance trade-off)", fontsize=11)
plt.tight_layout(); plt.savefig("fig7_tuning_curves.png"); plt.close()

# Figure 8: small illustrative tree
tree3 = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xc_tr, yc_tr)
print(f"\nIllustration tree (depth 3): test accuracy = {tree3.score(Xc_te, yc_te):.3f}")
R["tree3_test"] = round(float(tree3.score(Xc_te, yc_te)), 3)
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(tree3, feature_names=list(Xc.columns), class_names=["malignant", "benign"], filled=True, rounded=True, fontsize=8, ax=ax, impurity=False)
ax.set_title("Figure 8: A depth-3 Decision Tree (each box is a yes/no question about one feature)", fontsize=11)
plt.tight_layout(); plt.savefig("fig8_decision_tree.png"); plt.close()

# =====================================================================
# PART D: CONTRAST DATASET - TWO MOONS (non-linear)
# =====================================================================
section("PART D: CONTRAST - a non-linear problem (two moons)")
Xm, ym = make_moons(n_samples=1000, noise=0.3, random_state=SEED)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.2, random_state=SEED, stratify=ym)
moon_models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
}
moon_acc = {}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
xx, yy = np.meshgrid(np.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 300), np.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 300))
for ax, (name, m) in zip(axes, moon_models.items()):
    m.fit(Xm_tr, ym_tr)
    acc = m.score(Xm_te, ym_te); moon_acc[name] = round(float(acc), 3)
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(Xm_te[:, 0], Xm_te[:, 1], c=ym_te, cmap="coolwarm", s=12, edgecolor="k", linewidth=0.2)
    ax.set_title(f"{name}\nTest accuracy {acc:.3f}", fontsize=10); ax.grid(False)
    print(f"  {name:20s}: test accuracy = {acc:.3f}")
fig.suptitle("Figure 9: Decision boundaries on a non-linear dataset (two moons)", fontsize=11)
plt.tight_layout(); plt.savefig("fig9_moons_boundaries.png"); plt.close()
R["moons"] = moon_acc

with open("results.json", "w") as f:
    json.dump(R, f, indent=2, default=str)
print("\nDone. Saved figures fig1 - fig9 and results.json")
