from google.cloud import aiplatform

aiplatform.init(project="mlops-flask-app-440106", location="us-central1")

job = aiplatform.HyperparameterTuningJob(
    display_name="xgboost-hyperparam-tuning",
    custom_job=aiplatform.CustomJob.from_local_script(
        display_name="xgboost-training",
        script_path="train.py",
        container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-3:latest",
        requirements=["xgboost"],
    ),
    metric_spec={"accuracy": "maximize"},
    parameter_spec={
        "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(0.001, 0.1, scale="log"),
        "max_depth": aiplatform.hyperparameter_tuning.IntegerParameterSpec(3, 10, scale="linear"),
    },
    max_trial_count=10,
    parallel_trial_count=3,
)

job.run()
