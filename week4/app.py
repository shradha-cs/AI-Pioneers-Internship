"""
Week 4 Task: AI Project Deployment & Capstone
Step 5: Prediction API using Flask

Loads the serialized model (model.joblib) and scaler (scaler.joblib),
and exposes a simple REST API for making breast cancer predictions.

Endpoints:
    GET  /            -> health check / welcome message
    GET  /features     -> list of expected input feature names
    POST /predict       -> takes feature values as JSON, returns prediction

Run locally with:
    python app.py
Then it will be available at http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model artifacts once, at startup (not on every request)
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Breast Cancer Prediction API is running.",
        "endpoints": {
            "GET /features": "List the input features the model expects",
            "POST /predict": "Send feature values as JSON to get a prediction"
        }
    })


@app.route("/features", methods=["GET"])
def features():
    return jsonify({
        "n_features": len(feature_names),
        "feature_names": feature_names
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if data is None or "features" not in data:
        return jsonify({
            "error": "Request body must be JSON with a 'features' object, "
                     "e.g. {'features': {'mean radius': 14.2, ...}}"
        }), 400

    input_features = data["features"]

    # Validate all required features are present
    missing = [f for f in feature_names if f not in input_features]
    if missing:
        return jsonify({
            "error": f"Missing {len(missing)} required feature(s).",
            "missing_features": missing
        }), 400

    # Build the feature vector in the correct order (as a DataFrame so the
    # scaler sees the same column names it was fitted with)
    try:
        import pandas as pd
        x = pd.DataFrame([[float(input_features[f]) for f in feature_names]], columns=feature_names)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature value: {e}"}), 400

    # Scale and predict
    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)[0]
    probability = model.predict_proba(x_scaled)[0]

    label = "benign" if prediction == 1 else "malignant"

    return jsonify({
        "prediction": label,
        "prediction_code": int(prediction),
        "probability_malignant": round(float(probability[0]), 4),
        "probability_benign": round(float(probability[1]), 4)
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
