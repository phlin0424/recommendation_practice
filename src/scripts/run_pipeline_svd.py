import logging

import joblib
from core.config import DIR_PATH, settings
from pipelines.pipeline_svd.evaluate import evaluation_model
from pipelines.pipeline_svd.preprocess import preprocess
from pipelines.pipeline_svd.train import train_model

import mlflow

# Settings of logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Logging the experiment details
logger.info(f"MLFLOW_TRACKING_URI: {settings.tracking_uri}")
logger.info(f"ARTIFACT_ROOT: {settings.artifact_location}")
logger.info(f"experiment_id: {settings.experiment_id}")

# Load the tracking uri and the experiment name
tracking_uri = settings.tracking_uri
experiment_name = settings.experiment_name

# Set the tracking URI to the local MLflow server
mlflow.set_tracking_uri(tracking_uri)

# Set up the experiment
# experiment = mlflow.get_experiment_by_name(experiment_name)


# ================================================
#  Pipeline
# ================================================

with mlflow.start_run(experiment_id=settings.experiment_id) as run:
    # Preprocess
    test_size = 0.25
    trainset, testset = preprocess(test_size)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("model_name", "svd")
    mlflow.log_param("dataset", "ml-lm")

    # model training
    algo = train_model(trainset)

    # Save the model locally
    model_filename = DIR_PATH / "mlflow/artifacts/svd_model.pkl"
    joblib.dump(algo, model_filename)
    print(f"Model saved to {model_filename}")
    mlflow.log_artifact(model_filename, artifact_path="models")

    # predict
    rmse = evaluation_model(algo, testset)
    mlflow.log_metric("rmse", rmse)
    print(f"RMSE: {rmse}")
