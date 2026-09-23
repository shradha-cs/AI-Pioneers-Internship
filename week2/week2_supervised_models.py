"""
Week 2 Task: Supervised Machine Learning Models
- Regression: Linear Regression (Diabetes dataset)
- Classification: Logistic Regression vs Random Forest (Breast Cancer dataset)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.datasets import load_diabetes, load_breast_cancer

print("="*60)
print("PART 1: REGRESSION - Linear Regression")
print("="*60)

# Load data
housing = load_diabetes(as_frame=True)
X_reg = housing.data
y_reg = housing.target

print(f"Dataset shape: {X_reg.shape}")
print(f"Features: {list(X_reg.columns)}")

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale
scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

# Train
lin_reg = LinearRegression()
lin_reg.fit(X_train_r_scaled, y_train_r)
y_pred_r = lin_reg.predict(X_test_r_scaled)

# Evaluate
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\nLinear Regression Results:")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R2   : {r2:.4f}")

print("\n" + "="*60)
print("PART 2: CLASSIFICATION - Logistic Regression vs Random Forest")
print("="*60)

# Load data
cancer = load_breast_cancer(as_frame=True)
X_clf = cancer.data
y_clf = cancer.target

print(f"Dataset shape: {X_clf.shape}")
print(f"Classes: {cancer.target_names}")

# Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Scale
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_c_scaled, y_train_c)
y_pred_log = log_reg.predict(X_test_c_scaled)

acc_log = accuracy_score(y_test_c, y_pred_log)
prec_log = precision_score(y_test_c, y_pred_log)
rec_log = recall_score(y_test_c, y_pred_log)
f1_log = f1_score(y_test_c, y_pred_log)

print(f"\nLogistic Regression Results:")
print(f"  Accuracy  : {acc_log:.4f}")
print(f"  Precision : {prec_log:.4f}")
print(f"  Recall    : {rec_log:.4f}")
print(f"  F1 Score  : {f1_log:.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test_c, y_pred_log)}")

# Model 2: Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_c, y_train_c)  # tree-based models don't need scaling
y_pred_rf = rf_clf.predict(X_test_c)

acc_rf = accuracy_score(y_test_c, y_pred_rf)
prec_rf = precision_score(y_test_c, y_pred_rf)
rec_rf = recall_score(y_test_c, y_pred_rf)
f1_rf = f1_score(y_test_c, y_pred_rf)

print(f"\nRandom Forest Results:")
print(f"  Accuracy  : {acc_rf:.4f}")
print(f"  Precision : {prec_rf:.4f}")
print(f"  Recall    : {rec_rf:.4f}")
print(f"  F1 Score  : {f1_rf:.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test_c, y_pred_rf)}")

# Feature importance from RF (top 5)
importances = pd.Series(rf_clf.feature_importances_, index=X_clf.columns)
top5 = importances.sort_values(ascending=False).head(5)
print(f"\nTop 5 important features (Random Forest):\n{top5}")

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [acc_log, acc_rf],
    "Precision": [prec_log, prec_rf],
    "Recall": [rec_log, rec_rf],
    "F1 Score": [f1_log, f1_rf]
})
print(comparison.to_string(index=False))

# Save results to a text file for reference
with open("results.txt", "w") as f:
    f.write("REGRESSION (Linear Regression on Diabetes dataset)\n")
    f.write(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}\n\n")
    f.write("CLASSIFICATION (Breast Cancer dataset)\n")
    f.write(comparison.to_string(index=False))
    f.write(f"\n\nTop 5 features:\n{top5.to_string()}")

print("\nDone. Results saved to result,txt")
