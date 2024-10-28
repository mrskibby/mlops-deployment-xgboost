from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load('xgboost_model.pkl')  # Assuming the model is saved as .pkl

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON request data
    data = request.get_json()
    input_data = pd.DataFrame([data])

    # Make prediction using the model
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
