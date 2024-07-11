import logging
import os

import joblib
from utils.pipeline_module import evaluation_model, preprocess, train_model

import mlflow

# Settings of logging output
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s][%(message)s]",
)
logger = logging.getLogger(__name__)

# TODO: integrate to core.config
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
experiment_name = "recommendation_system_2"

# Set the tracking URI to the local MLflow server
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
    )
else:
    experiment_id = experiment.experiment_id


# Get the experiment details for logging
experiment = mlflow.get_experiment(experiment_id)
artifact_location = experiment.artifact_location
tracking_uri = mlflow.get_tracking_uri()

# Logging the experiment details
logger.info(f"MLFLOW_TRACKING_URI: {tracking_uri}")
logger.info(f"ARTIFACT_ROOT: {artifact_location}")
logger.info(f"experiment_id: {experiment_id}")


# ================================================
#  Pipeline
# ================================================

with mlflow.start_run(experiment_id=experiment_id) as run:
    # Preprocess
    trainset, testset = preprocess()
    mlflow.log_param("test_size", 0.25)

    # model training
    algo = train_model(trainset)

    # Save the model locally
    model_filename = "mlflow/artifacts/svd_model.pkl"
    joblib.dump(algo, model_filename)
    print(f"Model saved to {model_filename}")
    mlflow.log_artifact(model_filename, artifact_path="models")

    # predict
    rmse = evaluation_model(algo, testset)
    mlflow.log_metric("rmse", rmse)
    print(f"RMSE: {rmse}")
