import asyncio

import mlflow
from core.config import DIR_PATH, settings
from datareader.ml_10m_data import PopularityDatas
from pipelines.popularity.model import PopularityRecommender
from pydantic_settings import BaseSettings, SettingsConfigDict
from utils.evaluation_metrics import Metrics
from utils.pipeline_logging import configure_logging
import joblib


logger = configure_logging()


# pipeline environment variables controller:
class PipelineSettings(BaseSettings):
    user_num: int = 1000
    threshold: int = 10
    model_name: str = "Popularity"

    model_config = SettingsConfigDict(
        env_file="pipeline_params.env",
        env_file_encoding="utf-8",
    )


def preprocess(pipeline_settings: PipelineSettings) -> PopularityRecommender:
    popularity_datas = asyncio.run(
        PopularityDatas.from_db(user_num=pipeline_settings.user_num, threshold=1)
    )

    # Initialize the recommender
    popularity_recommender = PopularityRecommender(popularity_datas)

    return popularity_recommender


def train_model(recommender: PopularityRecommender) -> PopularityRecommender:
    recommender.train()
    return recommender


def evaluate_model(
    popularity_recommender: PopularityRecommender,
    pipeline_settings: PipelineSettings,
) -> Metrics:
    popularity_recommender.predict(threshold=pipeline_settings.threshold)
    metrics = popularity_recommender.evaluate()
    return metrics


def run_pipeline(pipeline_settings: PipelineSettings):
    # Set the tracking URI to the local MLflow server
    tracking_uri = settings.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # Logging the experiment details
    logger.info(f"MLflow tracking uri: {settings.tracking_uri}")
    logger.info(f"artifact root: {settings.artifact_location}")
    logger.info(f"experiment id: {settings.experiment_id}")
    logger.info(f"experiment name: {settings.experiment_name}")

    with mlflow.start_run(
        experiment_id=settings.experiment_id,
        run_name=pipeline_settings.model_name,
    ) as run:
        mlflow.log_params(pipeline_settings.model_dump())

        # ++++++++++++++++++++++++++
        # Preprocess
        # ++++++++++++++++++++++++++
        algo = preprocess(pipeline_settings)
        # joblib.dump(algo, preprocessed_model_filename)
        # logger.info(f"Preprocessed Model saved to {model_filename}")

        # ++++++++++++++++++++++++++
        # Train
        # ++++++++++++++++++++++++++
        algo = train_model(algo)
        # Setting of artifact (trained model)
        model_name = pipeline_settings.model_name
        model_output_fname = f"{model_name}.pkl"
        model_filename = DIR_PATH / f"mlflow/artifacts/{model_output_fname}"
        joblib.dump(algo, model_filename)
        mlflow.log_artifact(model_filename, artifact_path="models")
        # logger.info(f"Model saved to {pipeline_settings.model_name}")

        # ++++++++++++++++++++++++++
        # Predict & Evaluate
        # ++++++++++++++++++++++++++
        # Predict & Evaluate
        metrics = evaluate_model(algo, pipeline_settings)
        # Log the evaluation results
        mlflow.log_metrics(metrics.model_dump())
        logger.info(metrics)


if __name__ == "__main__":
    # input_data = preprocess(1000)
    # recommender = train_model(input_data)
    # metrics = evaluate_model(recommender, 10)
    # logger.info(metrics)
    pipeline_settings = PipelineSettings()

    logger.info(pipeline_settings)

    run_pipeline(pipeline_settings)
