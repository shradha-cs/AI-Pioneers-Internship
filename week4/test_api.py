"""
Test script for the Flask Prediction API.
Uses Flask's built-in test client (simulates real HTTP requests
without needing a separately running server) to verify all endpoints work.
"""

import json
from app import app

client = app.test_client()

print("="*70)
print("TEST 1: GET / (health check)")
print("="*70)
resp = client.get("/")
print(f"Status code: {resp.status_code}")
print(json.dumps(resp.get_json(), indent=2))

print("\n" + "="*70)
print("TEST 2: GET /features")
print("="*70)
resp = client.get("/features")
print(f"Status code: {resp.status_code}")
data = resp.get_json()
print(f"n_features: {data['n_features']}")
print(f"First 5 feature names: {data['feature_names'][:5]}")

print("\n" + "="*70)
print("TEST 3: POST /predict (with real sample data)")
print("="*70)
with open("sample_requests.json") as f:
    samples = json.load(f)

for i, sample in enumerate(samples):
    resp = client.post("/predict", json={"features": sample["features"]})
    result = resp.get_json()
    print(f"\nSample {i+1} (true label: {sample['true_label']}):")
    print(f"  Status code: {resp.status_code}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  P(malignant): {result['probability_malignant']}, P(benign): {result['probability_benign']}")
    match = "MATCH" if result["prediction"] == sample["true_label"] else "MISMATCH"
    print(f"  Result: {match}")

print("\n" + "="*70)
print("TEST 4: POST /predict with MISSING features (error handling check)")
print("="*70)
resp = client.post("/predict", json={"features": {"mean radius": 14.2}})
print(f"Status code: {resp.status_code}")
result = resp.get_json()
print(f"Error message: {result['error']}")
print(f"Number of missing features reported: {len(result['missing_features'])}")

print("\n" + "="*70)
print("TEST 5: POST /predict with NO body (error handling check)")
print("="*70)
resp = client.post("/predict", json={})
print(f"Status code: {resp.status_code}")
print(json.dumps(resp.get_json(), indent=2))

print("\nAll API tests completed.")
