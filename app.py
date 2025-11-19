# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("cvd_model.joblib")

# Helper: normalize incoming JSON into dataframe expected by the pipeline
def rows_to_df(rows):
    # rows: list of dicts or single dict
    if isinstance(rows, dict):
        rows = [rows]
    df = pd.DataFrame(rows)
    # Keep same column names/order as training
    # Convert yes/no, sex similar to training script
    if 'is_smoking' in df.columns:
        df['is_smoking'] = df['is_smoking'].map({'YES':1, 'NO':0}).fillna(df['is_smoking'])
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'M':1, 'F':0}).fillna(df['sex'])
    # Ensure numeric types where possible
    for c in ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error":"invalid json"}), 400
    df = rows_to_df(data.get('rows', data))
    if df.shape[0] == 0:
        return jsonify({"error":"no rows provided"}), 400
    # Use model pipeline to predict
    preds = model.predict(df)
    probs = None
    try:
        probs = model.predict_proba(df)[:,1].tolist()
    except Exception:
        # model doesn't have predict_proba
        probs = [None]*len(preds)
    results = []
    for i in range(len(preds)):
        results.append({
            "index": i,
            "prediction": int(preds[i]),
            "probability": float(probs[i]) if probs[i] is not None else None
        })
    return jsonify({"results": results}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
