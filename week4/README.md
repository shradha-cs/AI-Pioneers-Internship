# Week 4 Capstone – Breast Cancer Prediction API

Internship: AI Pioneers Internship (Machine Learning) — YuvaIntern

## Overview
An end-to-end machine learning project: data preprocessing → model training →
testing → serialization → deployment as a REST API using Flask.

The model predicts whether a breast tumor is **malignant** or **benign** based
on 30 numeric features computed from a digitized image of a cell nucleus
(Breast Cancer Wisconsin dataset).

## Project Structure
week4_project/
├── train_model.py # Data preprocessing, training, evaluation, serialization
├── app.py # Flask REST API that serves predictions
├── test_api.py # Tests the API endpoints end-to-end
├── model.joblib # Serialized trained model
├── scaler.joblib # Serialized StandardScaler (must match training)
├── feature_names.joblib # Ordered list of the 30 expected feature names
├── metrics.json # Saved evaluation metrics from training
├── sample_requests.json # Example API request payloads for testing/demo
└── requirements.txt


## Setup & Running Locally

```bash
pip install -r requirements.txt

# Step 1: Train the model (creates model.joblib, scaler.joblib, etc.)
python train_model.py

# Step 2: Run the API server
python app.py
# Server starts at http://127.0.0.1:5000

# Step 3 (optional): run the automated test suite against the API
python test_api.py
```

## API Endpoints

### `GET /`
Health check. Returns a welcome message and lists available endpoints.

### `GET /features`
Returns the 30 feature names the model expects, in the required order.

### `POST /predict`
Accepts a JSON body with a `features` object containing all 30 required
feature values, and returns a prediction.

**Example request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "mean radius": 14.2, "mean texture": 20.1, "mean perimeter": 91.5,
      "... (all 30 features required, see sample_requests.json for a full example)"
    }
  }'
```

**Example response:**
```json
{
  "prediction": "malignant",
  "prediction_code": 0,
  "probability_malignant": 0.9995,
  "probability_benign": 0.0005
}
```

`sample_requests.json` contains 3 complete, ready-to-use example payloads
(pulled from real test-set rows) for quick testing without needing to type
out all 30 feature values by hand.

## Model Performance (on held-out test set)

| Metric | Value |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9589 |
| Recall | 0.9722 |
| F1 Score | 0.9655 |
| ROC-AUC | 0.9927 |

Model: Random Forest Classifier (`n_estimators=200, max_depth=5, min_samples_split=5`)
— hyperparameters carried over from the GridSearchCV tuning done in Week 3.

## Deployment Notes
This project runs locally with Flask's built-in development server. For a
production deployment, the same `app.py` can be hosted on a platform such as
Render, Railway, PythonAnywhere, or Heroku, typically by:
1. Adding a `Procfile` (e.g. `web: gunicorn app:app`) and using `gunicorn`
   instead of Flask's dev server.
2. Committing `model.joblib`, `scaler.joblib`, and `feature_names.joblib`
   alongside the code (or regenerating them via `train_model.py` on the server).
3. Setting the platform's start command to run the Flask app.

## Tools & Libraries Used
Python 3, Flask, Scikit-learn, Joblib, Pandas, NumPy
