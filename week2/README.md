# Week 2 Task – Supervised Machine Learning Models

Internship: AI Pioneers Internship (Machine Learning) — YuvaIntern

## Overview
All five required algorithms are implemented with Scikit-learn — **Linear Regression, Logistic Regression,
Decision Tree, Random Forest and K-Nearest Neighbors** — on two problems:

- **Regression:** Diabetes dataset (442 patients, 10 features) → predict disease progression
- **Classification:** Breast Cancer Wisconsin dataset (569 tumours, 30 features) → malignant vs benign

Beyond reporting scores, the project runs experiments to explain *why* each model behaves as it does
(cross-validation, a significance test, learning curves, out-of-fold error comparison, a contrast dataset).

## Files
| File | Description |
|---|---|
| `week2_supervised_models.py` | Trains and evaluates all models and runs the analysis experiments |
| `results.txt` | Full printed output of the run |
| `fig1` … `fig9` (`.png`) | Predicted vs actual, residuals, confusion matrices, ROC/CV spread, learning curves, feature importance, tuning curves, decision tree, two-moons decision boundaries |

## How to run
```bash
pip install scikit-learn pandas numpy matplotlib scipy
python week2_supervised_models.py
```

## How the models are compared
- 80/20 train/test split **plus** repeated cross-validation (regression 5x5, classification 5x10), because the test sets are small (89 and 114 rows)
- A no-skill **baseline** for each problem
- Tree depth and KNN `k` chosen by cross-validation on the **training set only**
- Scaling is done inside a `Pipeline` so the scaler never sees test data
- Malignant is the **positive class** for precision / recall / F1 (a missed cancer is the costly error)

## Results

### Regression (Diabetes)
| Model | Test R² | CV R² |
|---|---|---|
| Linear Regression | 0.453 | 0.480 |
| Decision Tree (default) | 0.061 | -0.141 |
| Decision Tree (depth=2) | 0.295 | 0.330 |
| Random Forest (100 trees) | 0.443 | 0.415 |
| KNN (k=5) | 0.425 | 0.375 |
| KNN (k=18) | 0.423 | 0.440 |
| Baseline (predict the mean) | -0.012 | -0.013 |

### Classification (Breast Cancer)
| Model | Test accuracy | CV accuracy |
|---|---|---|
| Logistic Regression | 0.9825 | 0.9789 |
| Decision Tree (default) | 0.9123 | 0.9299 |
| Decision Tree (depth=2) | 0.8947 | 0.9191 |
| Random Forest (100 trees) | 0.9561 | 0.9576 |
| KNN (k=5) | 0.9561 | 0.9657 |
| KNN (k=10) | 0.9649 | 0.9659 |
| Baseline (always benign) | 0.6316 | 0.6274 |

## Why did Logistic Regression outperform Random Forest?
1. **The gap is real but not conclusive.** LR won 41 of 50 cross-validation folds (mean +2.1 points), but the corrected resampled t-test gives p = 0.082.
2. **Random Forest memorises.** It scores 100% on training data but 95.6% on unseen data (a 4.4-point gap) versus 1.4 points for Logistic Regression.
3. **Simpler models help most with limited data.** LR leads at every training size; the lead is 3.7 points at 45 samples and 1.8 points at 455.
4. **Correlated features.** 21 of 435 feature pairs have |r| > 0.9; the two models rely on quite different features (only 5 of their top 10 overlap).
5. **It is dataset-specific.** On a non-linear dataset (two moons) Random Forest and KNN overtake Logistic Regression.

## Other findings
- An unrestricted Decision Tree scores 100% on training data but has a cross-validated R² **below the baseline** on the regression task
- KNN needs scaled features: best scaled accuracy 0.969 vs 0.941 unscaled (feature ranges differ by a factor of ~140,000)
- Linear Regression coefficients `s1` and `s2` have opposite signs because the two features are highly correlated (r = 0.897, VIF 59 and 39)
