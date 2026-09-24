"""
Week 3 Task: Unsupervised Learning & Model Evaluation
Part 1: K-Means and Hierarchical Clustering (with PCA) on the Wine dataset
Part 2: Cross-validation, Confusion Matrix, Precision/Recall/F1, ROC-AUC,
        and Hyperparameter Tuning on the Breast Cancer dataset
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, accuracy_score, ConfusionMatrixDisplay
)

print("="*70)
print("PART 1: CLUSTERING - K-Means & Hierarchical Clustering (Wine dataset)")
print("="*70)

wine = load_wine(as_frame=True)
X_wine = wine.data
y_wine_true = wine.target  # true labels, used ONLY for reference/comparison

print(f"Dataset shape: {X_wine.shape}")
print(f"True classes (for reference only, not used in clustering): {list(wine.target_names)}")

# Scale features - essential for clustering (distance-based)
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

# --- Dimensionality Reduction with PCA ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_wine_scaled)
print(f"\nPCA explained variance ratio (2 components): {pca.explained_variance_ratio_}")
print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.4f}")

# --- Elbow Method to choose K for K-Means ---
inertias = []
sil_scores_kmeans = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_wine_scaled)
    inertias.append(km.inertia_)
    sil_scores_kmeans.append(silhouette_score(X_wine_scaled, labels))

print("\nK-Means: Inertia and Silhouette Score per K")
for k, inertia, sil in zip(K_range, inertias, sil_scores_kmeans):
    print(f"  K={k}: Inertia={inertia:.2f}, Silhouette={sil:.4f}")

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_range), inertias, marker='o', color='#2E4053')
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method for Optimal K")
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_range), sil_scores_kmeans, marker='o', color='#C0392B')
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs K")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("week3_elbow_silhouette.png", dpi=150)
plt.close()

# Final K-Means with K=3 (matches known 3 wine cultivars, also elbow-supported)
best_k = 3
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_wine_scaled)
kmeans_sil = silhouette_score(X_wine_scaled, kmeans_labels)
print(f"\nFinal K-Means (K={best_k}) Silhouette Score: {kmeans_sil:.4f}")

# --- Hierarchical Clustering ---
hier = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
hier_labels = hier.fit_predict(X_wine_scaled)
hier_sil = silhouette_score(X_wine_scaled, hier_labels)
print(f"Hierarchical Clustering (K={best_k}) Silhouette Score: {hier_sil:.4f}")

# Plot dendrogram
plt.figure(figsize=(10, 5))
Z = linkage(X_wine_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=20)
plt.title("Hierarchical Clustering Dendrogram (Ward linkage, truncated)")
plt.xlabel("Sample clusters")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("week3_dendrogram.png", dpi=150)
plt.close()

# Plot clusters on PCA-reduced data: K-Means vs Hierarchical vs True labels
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
scatter_kwargs = dict(cmap='viridis', s=40, edgecolor='k', linewidth=0.3)

sc0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, **scatter_kwargs)
axes[0].set_title(f"K-Means Clusters (K={best_k})\nSilhouette={kmeans_sil:.3f}")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")

sc1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, **scatter_kwargs)
axes[1].set_title(f"Hierarchical Clusters (K={best_k})\nSilhouette={hier_sil:.3f}")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")

sc2 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_wine_true, **scatter_kwargs)
axes[2].set_title("True Wine Cultivar Labels\n(for reference only)")
axes[2].set_xlabel("PCA Component 1")
axes[2].set_ylabel("PCA Component 2")

plt.tight_layout()
plt.savefig("week3_clusters_pca.png", dpi=150)
plt.close()

print("\nSaved plots: week3_elbow_silhouette.png, week3_dendrogram.png, week3_clusters_pca.png")

print("\n" + "="*70)
print("PART 2: MODEL EVALUATION & HYPERPARAMETER TUNING (Breast Cancer dataset)")
print("="*70)

cancer = load_breast_cancer(as_frame=True)
X_clf = cancer.data
y_clf = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# --- Baseline model ---
baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# --- Cross-validation (5-fold) on training data ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(baseline_rf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Accuracy Scores: {np.round(cv_scores, 4)}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit baseline on full training set, evaluate on test set
baseline_rf.fit(X_train, y_train)
y_pred_base = baseline_rf.predict(X_test)
y_proba_base = baseline_rf.predict_proba(X_test)[:, 1]

acc_base = accuracy_score(y_test, y_pred_base)
prec_base = precision_score(y_test, y_pred_base)
rec_base = recall_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)
roc_auc_base = roc_auc_score(y_test, y_proba_base)
cm_base = confusion_matrix(y_test, y_pred_base)

print(f"\nBaseline Random Forest (default params) - Test Set Performance:")
print(f"  Accuracy  : {acc_base:.4f}")
print(f"  Precision : {prec_base:.4f}")
print(f"  Recall    : {rec_base:.4f}")
print(f"  F1 Score  : {f1_base:.4f}")
print(f"  ROC-AUC   : {roc_auc_base:.4f}")
print(f"  Confusion Matrix:\n{cm_base}")

# --- Hyperparameter Tuning with GridSearchCV ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest Hyperparameters found: {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
y_proba_tuned = best_rf.predict_proba(X_test)[:, 1]

acc_tuned = accuracy_score(y_test, y_pred_tuned)
prec_tuned = precision_score(y_test, y_pred_tuned)
rec_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
roc_auc_tuned = roc_auc_score(y_test, y_proba_tuned)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

print(f"\nTuned Random Forest - Test Set Performance:")
print(f"  Accuracy  : {acc_tuned:.4f}")
print(f"  Precision : {prec_tuned:.4f}")
print(f"  Recall    : {rec_tuned:.4f}")
print(f"  F1 Score  : {f1_tuned:.4f}")
print(f"  ROC-AUC   : {roc_auc_tuned:.4f}")
print(f"  Confusion Matrix:\n{cm_tuned}")

# --- Plot: Confusion matrices (baseline vs tuned) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(cm_base, display_labels=cancer.target_names).plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f"Baseline RF Confusion Matrix\nAccuracy={acc_base:.3f}")
ConfusionMatrixDisplay(cm_tuned, display_labels=cancer.target_names).plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title(f"Tuned RF Confusion Matrix\nAccuracy={acc_tuned:.3f}")
plt.tight_layout()
plt.savefig("week3_confusion_matrices.png", dpi=150)
plt.close()

# --- Plot: ROC Curves ---
fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_base)
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_proba_tuned)

plt.figure(figsize=(6, 5))
plt.plot(fpr_base, tpr_base, label=f"Baseline RF (AUC={roc_auc_base:.3f})", color='#2E4053')
plt.plot(fpr_tuned, tpr_tuned, label=f"Tuned RF (AUC={roc_auc_tuned:.3f})", color='#C0392B')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("week3_roc_curve.png", dpi=150)
plt.close()

print("\nSaved plots: week3_confusion_matrices.png, week3_roc_curve.png")

# --- Save all results to text file ---
with open("results.txt", "w") as f:
    f.write("PART 1: CLUSTERING (Wine dataset)\n")
    f.write(f"PCA variance captured (2 components): {pca.explained_variance_ratio_.sum():.4f}\n")
    f.write(f"K-Means (K=3) Silhouette Score: {kmeans_sil:.4f}\n")
    f.write(f"Hierarchical Clustering (K=3) Silhouette Score: {hier_sil:.4f}\n\n")
    f.write("PART 2: MODEL EVALUATION (Breast Cancer dataset)\n")
    f.write(f"5-Fold CV Mean Accuracy (baseline): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n\n")
    f.write("Baseline Random Forest (test set):\n")
    f.write(f"  Accuracy: {acc_base:.4f}, Precision: {prec_base:.4f}, Recall: {rec_base:.4f}, F1: {f1_base:.4f}, ROC-AUC: {roc_auc_base:.4f}\n\n")
    f.write(f"Best Hyperparameters: {grid_search.best_params_}\n")
    f.write("Tuned Random Forest (test set):\n")
    f.write(f"  Accuracy: {acc_tuned:.4f}, Precision: {prec_tuned:.4f}, Recall: {rec_tuned:.4f}, F1: {f1_tuned:.4f}, ROC-AUC: {roc_auc_tuned:.4f}\n")

print("\nDone. Results saved to results.txt")
