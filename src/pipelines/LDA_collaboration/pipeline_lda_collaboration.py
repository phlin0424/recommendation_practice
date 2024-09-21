import asyncio

import mlflow
from core.config import DIR_PATH, settings
from datareader.ml_10m_data import IntegratedTagsDatas
from pipelines.LDA_collaboration.model import LDACollaborativeRecommender
from pydantic_settings import BaseSettings, SettingsConfigDict
from utils.evaluation_metrics import Metrics
from utils.pipeline_logging import configure_logging
import joblib

logger = configure_logging()


# pipeline environment variables controller:
class PipelineSettings(BaseSettings):
    user_num: int = 1000
    factors: int = 50
    n_epochs: int = 30
    model_name: str = "LDA_collaboration"

    model_config = SettingsConfigDict(
        env_file="pipeline_params.env",
        env_file_encoding="utf-8",
    )


def preprocess(pipeline_settings: PipelineSettings) -> LDACollaborativeRecommender:
    # Load the tag& genres data (movie contents)
    integrated_datas = asyncio.run(
        IntegratedTagsDatas.from_db(user_num=pipeline_settings.user_num)
    )

    # Initialize the recommender
    recommender = LDACollaborativeRecommender(
        integrated_datas,
    )

    # Apply preprocess
    recommender.preprocess()
    return recommender


def train_model(
    recommender: LDACollaborativeRecommender, pipeline_settings: PipelineSettings
) -> LDACollaborativeRecommender:
    # Train the recommender
    recommender.train(
        factors=pipeline_settings.factors,
        n_epochs=pipeline_settings.n_epochs,
    )
    return recommender


def evaluate_model(recommender: LDACollaborativeRecommender) -> Metrics:
    recommender.predict()
    metrics = recommender.evaluate()
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
        algo = train_model(algo, pipeline_settings)
        # Setting of artifact (trained model)
        model_name = pipeline_settings.model_name
        model_output_fname = f"{model_name}.pkl"
        model_filename = DIR_PATH / f"mlflow/artifacts/{model_output_fname}"
        joblib.dump(algo, model_filename)
        mlflow.log_artifact(model_filename, artifact_path="models")
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
