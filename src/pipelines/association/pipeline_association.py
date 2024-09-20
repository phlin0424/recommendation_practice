import asyncio

import mlflow
from core.config import settings
from datareader.ml_10m_data import IntegratedDatas
from pipelines.association.model import AssociationRecommender
from pydantic_settings import BaseSettings, SettingsConfigDict
from utils.evaluation_metrics import Metrics
from utils.pipeline_logging import configure_logging

logger = configure_logging()


class PipelineSettings(BaseSettings):
    user_num: int = 100
    min_support: float = 0.2
    model_name: str = "Association"

    model_config = SettingsConfigDict(
        env_file="pipeline_params.env",
        env_file_encoding="utf-8",
    )


def preprocess(pipeline_settings: PipelineSettings) -> AssociationRecommender:
    # Load the data
    intergrated_datas = asyncio.run(
        IntegratedDatas.from_db(user_num=pipeline_settings.user_num)
    )

    # Initialize the recommender
    association_recommender = AssociationRecommender(
        intergrated_datas,
        pipeline_settings.min_support,
    )

    return association_recommender


def train_model(
    association_recommender: AssociationRecommender,
) -> AssociationRecommender:
    association_recommender.train()
    return association_recommender


def evaluate_model(association_recommender: AssociationRecommender) -> Metrics:
    association_recommender.predict()
    metrics = association_recommender.evaluate()
    return metrics


def run_pipeline(pipeline_settings: PipelineSettings):
    # Set the tracking URI to the local MLflow server
    tracking_uri = settings.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # Setting of the pipeline
    # model_name = "MFRecommender_model"
    # model_output_fname = f"{model_name}.pkl"
    # model_filename = DIR_PATH / f"mlflow/artifacts/{model_output_fname}"

    # Logging the experiment details
    logger.info(f"MLflow tracking uri: {settings.tracking_uri}")
    logger.info(f"artifact root: {settings.artifact_location}")
    logger.info(f"experiment id: {settings.experiment_id}")
    logger.info(f"experiment name: {settings.experiment_name}")

    with mlflow.start_run(
        experiment_id=settings.experiment_id, run_name=pipeline_settings.model_name
    ) as run:
        # Log all the pipeline parameters
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
        # joblib.dump(algo, model_filename)
        # mlflow.log_artifact(model_filename, artifact_path="models")
        # logger.info(f"Model saved to {model_filename}")

        # ++++++++++++++++++++++++++
        # Predict & Evaluate
        # ++++++++++++++++++++++++++
        metrics = evaluate_model(algo)

        # Log the evaluation results
        mlflow.log_metrics(metrics.model_dump())
        logger.info(metrics)


if __name__ == "__main__":
    # input_data = preprocess(user_num=1000)
    # recommender = train_model(input_data)
    # metrics = evaluate_model(recommender)
    # logger.info(metrics)
    pipeline_settings = PipelineSettings()

    logger.info(pipeline_settings)

    run_pipeline(pipeline_settings)

    # print(pipeline_settings)
