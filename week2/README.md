# Week 2 Task – Supervised Machine Learning Models

Internship: AI Pioneers Internship (Machine Learning) — YuvaIntern

## Overview
This project implements and compares supervised machine learning models using Scikit-learn:
- **Regression**: Linear Regression on the Diabetes dataset
- **Classification**: Logistic Regression vs Random Forest on the Breast Cancer Wisconsin dataset

## Files
- `week2_supervised_models.py` — main script, trains all models and prints evaluation metrics
- `results.txt` — saved output of the run

## How to run
```bash
pip install scikit-learn pandas numpy
python week2_supervised_models.py
```

## Results Summary

### Regression (Linear Regression)
| Metric | Value |
|---|---|
| MSE | 2900.19 |
| RMSE | 53.85 |
| MAE | 42.79 |
| R² | 0.4526 |

### Classification (Logistic Regression vs Random Forest)
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 |

## Key Takeaway
Logistic Regression slightly outperformed Random Forest on this dataset, showing that
simpler models can outperform more complex ones on smaller, cleanly-separable data.
