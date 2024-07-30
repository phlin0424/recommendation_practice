import asyncio

from datareader.ml_10m_data import IntegratedDatas
from utils.evaluation_metrics import Metrics
from utils.models import RandomRecommender
from utils.pipeline_logging import configure_logging
import joblib
from core.config import DIR_PATH, settings
import os

import mlflow

logger = configure_logging()


def preprocess(user_num) -> IntegratedDatas:
    integrated_datas = asyncio.run(IntegratedDatas.from_db(user_num=user_num))
    return integrated_datas


def train_model(integrated_datas: IntegratedDatas) -> RandomRecommender:
    random_recommender = RandomRecommender(integrated_datas)
    random_recommender.train()
    return random_recommender


def evaluate_model(random_recommender: RandomRecommender) -> Metrics:
    random_recommender.predict()
    metrics = random_recommender.evaluate()
    return metrics


def run_pipeline():
    # Load the necessary parameters
    user_num = int(os.getenv("USER_NUM", "1000"))

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
        input_data = preprocess(user_num)
        mlflow.log_param("user_num", user_num)
        mlflow.log_param("model", model_name)
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


if __name__ == "__main__":
    # input_data = preprocess(user_num=1000)
    # recommender = train_model(input_data)
    # metrics = evaluate_model(recommender)
    # logger.info(metrics)
    run_pipeline()
