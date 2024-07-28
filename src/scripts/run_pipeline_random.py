import joblib
from core.config import DIR_PATH, settings
from pipelines.pipeline_random import evaluate_model, preprocess, train_model
from utils.pipeline_logging import configure_logging

import mlflow

logger = configure_logging()


# Set the tracking URI to the local MLflow server
tracking_uri = settings.tracking_uri
mlflow.set_tracking_uri(tracking_uri)

# Setting of the pipeline
model_name = "RandomRecommender_model"
model_output_fname = f"{model_name}.pkl"
model_filename = DIR_PATH / f"mlflow/artifacts/{model_output_fname}"

# Logging the experiment details
logger.info(f"MLflow tracking uri: {settings.tracking_uri}")
logger.info(f"artifact root: {settings.artifact_location}")
logger.info(f"experiment id: {settings.experiment_id}")
logger.info(f"experiment name: {settings.experiment_name}")


with mlflow.start_run(experiment_id=settings.experiment_id) as run:
    # Preprocess
    user_num = 1000
    input_data = preprocess(user_num)
    mlflow.log_param("user_num", user_num)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("dataset", "ml-10m")

    # Train the model, saving the trained model locally, registering the artifact
    algo = train_model(input_data)
    joblib.dump(algo, model_filename)
    mlflow.log_artifact(model_filename, artifact_path="models")
    logger.info(f"Model saved to {model_filename}")

    # Predict & Evaluate
    metrics = evaluate_model(algo)
    mlflow.log_metric("rmse", metrics.rmse)
    mlflow.log_metric("recall_at_k", metrics.recall_at_k)
    mlflow.log_metric("precision_at_k", metrics.precision_at_k)
    logger.info(metrics)
