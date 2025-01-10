import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import torch
import joblib
import json
import os

# Check if GPU is available and proceed accordingly
if torch.cuda.is_available():
    print(torch.cuda.current_device())  # Shows which GPU is being used
    # Create a tensor and move it to GPU
    device = torch.device("cuda")
    print(torch.rand(5, 3).to(device))
else:
    print("No GPU found, using CPU.")
    device = torch.device("cpu")
    print(torch.rand(5, 3).to(device))

# Path to store model metadata (in local file or GCS)
metadata_file = 'best_model_metadata.json'

# Save the model using joblib for Random Forest or XGBoost
def save_model(model, mse, r2, model_name, is_xgboost=False):
    if is_xgboost:
        model.get_booster().save_model(f'{model_name}.json')
    else:
        joblib.dump(model, f'{model_name}.pkl')

    metadata = {
        'best_model_name': model_name,
        'best_mse': mse,
        'best_r2': r2
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

def load_current_best():
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    else:
        return {'best_mse': float('inf'), 'best_r2': float('-inf')}

current_best = load_current_best()

# Load dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT", "PRICE"
]
df = pd.DataFrame(np.hstack([data, target.reshape(-1, 1)]), columns=columns)

X = df.drop("PRICE", axis=1)
y = df["PRICE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest MSE: {mse_rf}")
print(f"Random Forest R²: {r2_rf}")

# XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb}")
print(f"XGBoost R²: {r2_xgb}")

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': np.arange(3, 10, 1),
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0]
}

random_search_xgb = RandomizedSearchCV(
    xgb_model, param_distributions=param_grid_xgb, n_iter=50,
    scoring='neg_mean_squared_error', cv=5, verbose=1, random_state=42, n_jobs=-1
)
random_search_xgb.fit(X_train, y_train)
print(f"Best Parameters for XGBoost: {random_search_xgb.best_params_}")

best_xgb_model = random_search_xgb.best_estimator_
y_pred_best_xgb = best_xgb_model.predict(X_test)
best_mse_xgb = mean_squared_error(y_test, y_pred_best_xgb)
best_r2_xgb = r2_score(y_test, y_pred_best_xgb)
print(f"Best XGBoost MSE: {best_mse_xgb}")
print(f"Best XGBoost R²: {best_r2_xgb}")

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': np.arange(3, 15, 1),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf = RandomizedSearchCV(
    rf_model, param_distributions=param_grid_rf, n_iter=50,
    scoring='neg_mean_squared_error', cv=5, verbose=1, random_state=42, n_jobs=-1
)
random_search_rf.fit(X_train, y_train)
print(f"Best Parameters for Random Forest: {random_search_rf.best_params_}")

best_rf_model = random_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)
best_mse_rf = mean_squared_error(y_test, y_pred_best_rf)
best_r2_rf = r2_score(y_test, y_pred_best_rf)
print(f"Best Random Forest MSE: {best_mse_rf}")
print(f"Best Random Forest R²: {best_r2_rf}")

# Save the better model
if best_mse_xgb < current_best['best_mse']:
    print(f"XGBoost is better with MSE: {best_mse_xgb}")
    save_model(best_xgb_model, best_mse_xgb, best_r2_xgb, 'xgboost_model', is_xgboost=True)
elif best_mse_rf < current_best['best_mse']:
    print(f"Random Forest is better with MSE: {best_mse_rf}")
    save_model(best_rf_model, best_mse_rf, best_r2_rf, 'random_forest_model')
else:
    print("No improvement over the current best model.")



# this is for testing the code.