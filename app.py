import xgboost as xgb
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the saved XGBoost model (use the correct model format)
model = xgb.Booster()
model.load_model('xgboost_model.json')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data from the request
        input_data = request.get_json()

        # Convert the input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert DataFrame to DMatrix (XGBoost’s data format)
        dmatrix = xgb.DMatrix(input_df)

        # Perform prediction
        prediction = model.predict(dmatrix)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
