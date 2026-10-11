# Week 1 Task – Python for Machine Learning & Data Preprocessing

Internship: AI Pioneers Internship (Machine Learning) — YuvaIntern

## Overview
This project cleans and preprocesses a raw customer churn dataset using Python,
NumPy and Pandas, and documents every step with the reason behind it.

The dataset is **synthetic**: `generate_dataset.py` creates it with a fixed random
seed (so it is exactly reproducible) and deliberately injects realistic data-quality
problems — missing values, duplicate rows, a numeric column stored as text,
inconsistent category spellings and impossible values.

## Files
| File | Description |
|---|---|
| `generate_dataset.py` | Creates the raw, messy dataset |
| `preprocessing_eda.py` | Cleaning, EDA, feature selection, encoding, scaling, baseline model |
| `raw_customer_churn.csv` | Raw data (1,220 rows, 11 columns) |
| `cleaned_customer_churn.csv` | After cleaning, before encoding/scaling (1,200 rows) |
| `preprocessed_customer_churn.csv` | Final model-ready data (1,200 rows, 12 numeric columns) |
| `results.txt` | Full printed output of the run, including the reasoning for each step |
| `fig1` … `fig6` (`.png`) | Missing values, outliers, distributions, churn by segment, scaling, correlation heatmap |

## How to run
```bash
pip install pandas numpy matplotlib scipy scikit-learn
python generate_dataset.py
python preprocessing_eda.py
```

## Preprocessing steps (with the reason for each)
1. **Audit** the raw data first — found 20 duplicates, missing values in 4 columns, 12 blanks hidden inside a text column, 10 spellings of 3 contract types, and 10 impossible values.
2. **Remove duplicates** — they double-count customers and leak identical rows into train and test sets.
3. **Fix data types** — `total_charges` was text; blanks became NaN.
4. **Standardise labels** — spelling variants mapped to one canonical value.
5. **Impossible values → missing** — negative tenure and bills above $200 are errors, not outliers; extreme-but-real `support_calls` were kept.
6. **Missing values, method chosen per column** — `Unknown` category (categorical), median (skewed numbers), median *within the same internet plan* (monthly charges), tenure × monthly (total charges).
7. **EDA** — churn rates by contract, plan, support calls and tenure.
8. **Feature selection** — chi-square tests and correlations; dropped `customerID`, `total_charges` (redundant) and `gender`.
9. **Encoding** — ordinal for `contract`, one-hot for `internet_service` and `payment_method`.
10. **Scaling** — Min-Max to 0–1, verified against a hand-written NumPy version.
11. **Baseline model** — Logistic Regression to confirm the data can be learned from.

## Key findings
- Overall churn rate: **40.2%**
- Month-to-month contracts churn at **55.0%** vs **11.2%** for two-year contracts
- Fiber optic customers churn at **50.5%** vs **34.5%** for DSL
- Customers with 4+ support calls churn at **62.2%** vs **31.2%** with none
- Baseline Logistic Regression: accuracy **0.704** (majority-class baseline 0.600), ROC-AUC **0.774**

## Note
Because the data is synthetic, the relationships found were built into the generator.
The findings show the methods work; they are not facts about real customers.
