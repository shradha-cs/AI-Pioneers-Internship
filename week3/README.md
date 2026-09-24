# Week 3 Task – Unsupervised Learning & Model Evaluation

Internship: AI Pioneers Internship (Machine Learning) — YuvaIntern

## Overview
This project covers two parts:
1. **Clustering & Dimensionality Reduction** — K-Means and Hierarchical Clustering on the Wine dataset, with PCA used to visualize clusters in 2D.
2. **Model Evaluation & Hyperparameter Tuning** — Cross-validation, confusion matrix, precision/recall/F1/ROC-AUC, and GridSearchCV tuning on the Breast Cancer dataset.

## Files
- `week3_clustering_evaluation.py` — main script
- `results.txt` — saved output of the run
- `week3_elbow_silhouette.png` — elbow method + silhouette score plots (choosing K)
- `week3_clusters_pca.png` — K-Means vs Hierarchical vs true labels, in PCA space
- `week3_dendrogram.png` — hierarchical clustering dendrogram
- `week3_confusion_matrices.png` — baseline vs tuned model confusion matrices
- `week3_roc_curve.png` — ROC curve comparison

## How to run
```bash
pip install scikit-learn pandas numpy matplotlib scipy
python week3_clustering_evaluation.py
```

## Results Summary

### Clustering (Wine dataset)
| Method | Silhouette Score |
|---|---|
| K-Means (K=3) | 0.2849 |
| Hierarchical Clustering (K=3) | 0.2774 |

### Model Evaluation (Breast Cancer dataset)
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline RF | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9937 |
| Tuned RF (GridSearchCV) | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9927 |

Best hyperparameters found: `n_estimators=200, max_depth=5, min_samples_split=5` (best CV F1 = 0.9702)

## Key Takeaway
Both clustering methods recovered groupings that closely matched the true wine cultivars,
without ever seeing the labels. For classification, cross-validated scores were more
informative than a single test split when judging whether hyperparameter tuning actually helped.
