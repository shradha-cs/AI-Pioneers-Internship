"""
Week 4 Task: AI Project Deployment & Capstone
Step 1: Data Preprocessing, Model Training, Testing, and Serialization

This script trains a Random Forest classifier on the Breast Cancer Wisconsin
dataset and saves the trained model + scaler to disk using joblib, so they
can be loaded later by the Flask API (app.py) without retraining.
"""

import json
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

print("="*70)
print("STEP 1: DATA LOADING & PREPROCESSING")
print("="*70)

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape}")
print(f"Target classes: {list(data.target_names)} (0=malignant, 1=benign)")
print(f"Class balance: {dict(y.value_counts())}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*70)
print("STEP 2: MODEL TRAINING")
print("="*70)

# Using the best hyperparameters found via GridSearchCV in Week 3
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("Model trained: RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=5)")

print("\n" + "="*70)
print("STEP 3: MODEL TESTING & EVALUATION")
print("="*70)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

print("\n" + "="*70)
print("STEP 3b: THRESHOLD ANALYSIS (precision-recall trade-off)")
print("="*70)
print("Default classification uses a 0.5 probability threshold. Testing other")
print("thresholds shows the trade-off between catching more malignant cases")
print("(recall) versus fewer false alarms (precision).")

threshold_results = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_proba >= t).astype(int)
    p_t = precision_score(y_test, y_pred_t)
    r_t = recall_score(y_test, y_pred_t)
    f1_t = f1_score(y_test, y_pred_t)
    cm_t = confusion_matrix(y_test, y_pred_t)
    fn_t = cm_t[0][1]  # malignant misclassified as benign
    fp_t = cm_t[1][0]  # benign misclassified as malignant
    threshold_results.append({
        "threshold": t, "precision": round(p_t, 4), "recall": round(r_t, 4),
        "f1": round(f1_t, 4), "false_negatives": int(fn_t), "false_positives": int(fp_t)
    })
    print(f"  Threshold={t}: Precision={p_t:.4f}, Recall={r_t:.4f}, F1={f1_t:.4f}, "
          f"False Negatives={fn_t}, False Positives={fp_t}")

print("\n" + "="*70)
print("STEP 3c: FEATURE IMPORTANCE (model interpretability)")
print("="*70)
import pandas as pd
importances = pd.Series(model.feature_importances_, index=feature_names)
top5_features = importances.sort_values(ascending=False).head(5)
print("Top 5 most influential features in the model's decisions:")
print(top5_features)

print("\n" + "="*70)
print("STEP 4: MODEL SERIALIZATION (joblib)")
print("="*70)

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(feature_names, "feature_names.joblib")
print("Saved: model.joblib, scaler.joblib, feature_names.joblib")

# Save metrics for documentation/reporting
metrics = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "confusion_matrix": cm.tolist(),
    "n_features": len(feature_names),
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "threshold_analysis": threshold_results,
    "top5_features": top5_features.round(4).to_dict(),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved: metrics.json")

# Save a couple of sample test rows (as JSON) for API testing/demo purposes
sample_indices = [0, 1, 2]
samples = []
for idx in sample_indices:
    row = X_test.iloc[idx].to_dict()
    samples.append({
        "features": row,
        "true_label": data.target_names[y_test.iloc[idx]]
    })
with open("sample_requests.json", "w") as f:
    json.dump(samples, f, indent=2)
print("Saved: sample_requests.json (example API request payloads)")

print("\nDone.")
