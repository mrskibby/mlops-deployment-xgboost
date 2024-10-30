MLOps Flask App: XGBoost & Random Forest Model Pipeline

Project Overview

This project implements an end-to-end MLOps pipeline for training, hyperparameter tuning, and deploying machine learning models (XGBoost and Random Forest). The pipeline uses various Google Cloud services such as Vertex AI for hyperparameter tuning and Google Cloud Run for model deployment. GitHub Actions is employed for Continuous Integration/Continuous Deployment (CI/CD).

Architecture Overview

The following components make up the architecture:

1. Data Preparation: The pipeline uses the Boston Housing Dataset for model training.
2. Model Training: Both XGBoost and Random Forest models are trained. Hyperparameter tuning is performed to find the best-performing model.
3. Hyperparameter Tuning: Vertex AI is used to tune the models' hyperparameters using Bayesian optimization.
4. Containerization: The models are packaged in a Docker container for deployment.
5. CI/CD with GitHub Actions: Automatically builds, tests, and deploys the model whenever changes are pushed to GitHub.
6. Deployment on Cloud Run: The best-performing model is deployed as a REST API on Google Cloud Run, making it available for inference.

Tools and Services Used

1. Vertex AI: Used for hyperparameter tuning with parallel trial execution and resource autoscaling. It runs optimization jobs to find the best model configuration.
2. Google Cloud Run: Deployed the trained model as a REST API using Cloud Run, allowing for scalable serverless deployments.
3. Google Cloud Storage: Used to store trial artifacts and best model metadata.
4. GitHub Actions: Automated CI/CD pipeline to build, test, and deploy the model on Google Cloud.
5. Docker: The model code and dependencies are containerized for portability and consistent deployment.
6. Python Libraries:
- XGBoost: Used for building the primary regression model.
- Scikit-learn: Used for Random Forest and evaluation metrics.
- Pandas and NumPy: For data manipulation and numerical operations.

Key Features

Hyperparameter Tuning:

- Parallelized trials using Vertex AI Hyperparameter Tuning to automatically find the best parameters for both XGBoost and Random Forest models.
- Metrics like Mean Squared Error (MSE) and R² are used to evaluate the model performance.

Continuous Integration & Continuous Deployment (CI/CD):

- The pipeline automatically builds, tests, and deploys the model every time code is pushed to GitHub.
- Docker containers are built and pushed to Google Container Registry, followed by deployment on Cloud Run.

Automatic Deployment:

- The best model found by Vertex AI is automatically deployed using Google Cloud Run, and an API endpoint is generated for making predictions.

How to Reproduce the Pipeline

Clone the GitHub Repository:

git clone https://github.com/your-username/mlops-flask-app.git

Configure Secrets:

- Add the following secrets to GitHub for authentication and project access:
-- GCP_PROJECT_ID: Google Cloud Project ID.
-- GCLOUD_SERVICE_KEY: Base64-encoded service account key for authentication.

Push Changes to GitHub:

- Once the CI/CD pipeline is set up, any changes pushed to the repository will trigger the pipeline.

Hyperparameter Tuning:

- The pipeline runs hyperparameter tuning jobs in Vertex AI, optimizing both XGBoost and Random Forest models.
- The best-performing model is selected and deployed.

Cloud Run Deployment:

- Once the model is deployed to Cloud Run, a REST API is available for inference. You can access it via the provided URL (e.g., https://xgboost-flask-app-xxxxxx.us-central1.run.app).

API URL and Body

URL: https://xgboost-flask-app-634631521378.us-central1.run.app
request: POST
Body: JSON (raw)
{
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0.0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.0900,
    "RAD": 1.0,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98
}
response:
{
    "prediction": [
        27.69729995727539
    ]
}

Challenges Encountered

Authentication in CI/CD Pipeline:

- One challenge was properly authenticating GitHub Actions to interact with Google Cloud services. This was resolved by correctly configuring the GCLOUD_SERVICE_KEY secret for service account authentication.

Parallel Hyperparameter Tuning:

- Tuning multiple models (XGBoost and Random Forest) required careful configuration to ensure the correct parameter spaces were explored, and the resources were efficiently used during parallel trials.

Docker Configuration:

- Ensuring that the Docker image built locally is consistently deployed on Cloud Run involved multiple iterations of Dockerfile optimizations.

Vertex AI Tuning Job Complexity:

- Managing the complexity of hyperparameter tuning jobs across multiple models required attention to the parallelism and resource management features of Vertex AI.

Trade-offs Made

- XGBoost vs Random Forest: While both models were trained and tuned, XGBoost showed superior performance in this scenario, and hence it was selected for deployment.
- Cloud Run for Deployment: Cloud Run was chosen for its simplicity and serverless architecture. However, Kubernetes could offer more control over scaling and deployments if the project scales up.

Conclusion

This project demonstrates the full MLOps lifecycle using Google Cloud services, from data preparation and hyperparameter tuning to CI/CD automation and scalable deployment. It provides an efficient and scalable pipeline for deploying machine learning models in production.