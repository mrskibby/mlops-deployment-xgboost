from google.cloud import aiplatform
from google.cloud import storage  # Importing Google Cloud Storage
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import json
import os

# Initialize Vertex AI
aiplatform.init(
    project="mlops-flask-app-440106", 
    location="us-central1", 
    staging_bucket="gs://mlops-flask-bucket"
)

# Define the hyperparameter tuning job
job = aiplatform.CustomJob.from_local_script(
    display_name="xgboost-rf-hpt-job",
    script_path="tymestack.py",
    container_uri="gcr.io/cloud-aiplatform/training/scikit-learn-cpu.0-23:latest",  # Generic sklearn container for both models
    requirements=["xgboost", "scikit-learn", "torch", "flask", "pandas", "numpy"],
    replica_count=1,
    machine_type="n1-standard-4"
)

# Define hyperparameter tuning
hp_job = aiplatform.HyperparameterTuningJob(
    display_name="xgboost-rf-hpt",
    custom_job=job,
    metric_spec={"neg_mean_squared_error": "minimize"},  # Minimize MSE
    parameter_spec={
        "xgboost__n_estimators": hpt.IntegerParameterSpec(min=50, max=500, scale="linear"),
        "xgboost__learning_rate": hpt.DoubleParameterSpec(min=0.01, max=0.3, scale="log"),
        "xgboost__max_depth": hpt.IntegerParameterSpec(min=3, max=10, scale="linear"),
        "xgboost__subsample": hpt.DoubleParameterSpec(min=0.6, max=1.0, scale="linear"),
        "rf__n_estimators": hpt.IntegerParameterSpec(min=50, max=500, scale="linear"),
        "rf__max_depth": hpt.IntegerParameterSpec(min=3, max=15, scale="linear"),
        "rf__min_samples_split": hpt.IntegerParameterSpec(min=2, max=10, scale="linear"),
        "rf__min_samples_leaf": hpt.IntegerParameterSpec(min=1, max=4, scale="linear")
    },
    max_trial_count=20,
    parallel_trial_count=4
)

# Run the tuning job
hp_job.run()

# Get the best trial after the tuning job finishes
best_trial = hp_job.best_trial
print(f"Best Trial ID: {best_trial.id}")
print(f"Best Hyperparameters: {best_trial.parameters}")
print(f"Best MSE: {best_trial.final_measurement.metrics['neg_mean_squared_error']}")

# After identifying the best model, deploy it
best_model_image = "gcr.io/your-project-id/xgboost-flask-app"

# Deploy to Cloud Run
aiplatform.Model.upload(
    display_name="Best XGBoost Model",
    serving_container_image_uri=best_model_image
)

# Store the best trial information and results in a dictionary
best_trial_info = {
    "trial_id": best_trial.id,
    "best_hyperparameters": best_trial.parameters,
    "best_mse": best_trial.final_measurement.metrics['neg_mean_squared_error']
}

# Store the best trial results in Google Cloud Storage
def store_best_trial_results(results, bucket_name, file_name):
    """Store the best trial results in Google Cloud Storage"""
    # Initialize the storage client
    client = storage.Client()
    
    # Get the bucket
    bucket = client.bucket(bucket_name)
    
    # Create a blob (file in GCS)
    blob = bucket.blob(file_name)
    
    # Upload the results as a JSON file
    blob.upload_from_string(json.dumps(results), content_type='application/json')
    print(f"Best trial results saved to {bucket_name}/{file_name}")

# Save the best trial to the bucket
store_best_trial_results(best_trial_info, "mlops-flask-bucket", "best_trial_results.json")
