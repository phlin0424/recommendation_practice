import asyncio

import pandas as pd
from datareader.ml_1m_data import Ratings
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split


async def get_data_from_db():
    # fetch the data from db
    ratings = await Ratings.from_db()
    return ratings.data


def preprocess():
    # Load the training data from db
    ratings = asyncio.run(get_data_from_db())

    # Convert the training data to pd.Dataframe
    ratings_df = pd.DataFrame([rating.model_dump() for rating in ratings])

    # Define a Reader object
    reader = Reader(rating_scale=(1, 5))

    # Load the data into a Surprise dataset
    data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)

    # Split the dataset into train and test sets
    trainset, testset = train_test_split(data, test_size=0.25)

    return trainset, testset


def train_model(trainset):
    algo = SVD()
    algo.fit(trainset)
    return algo


def evaluation_model(algo, testset):
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    return rmse


if __name__ == "__main__":
    trainset, testset = preprocess()
    print(testset)
    # MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    # EXPERIMENT_NAME = "recommendation_system"
    # ARTIFACT_ROOT = "sqlite:/mlflow/mlflow.db"
    # data_dir = Path(__file__).resolve().parent / "ml-1m"
    # filepath = data_dir / "ratings.dat"
    # # Settings of the mlflow server
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # # Create a new experiment if not exists
    # experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    # if experiment is None:
    #     experiment_id = mlflow.create_experiment(
    #         name=EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT
    #     )
    # else:
    #     experiment_id = experiment.experiment_id

    # # Get the experiment details
    # experiment = mlflow.get_experiment(experiment_id)
    # artifact_location = experiment.artifact_location
    # tracking_uri = mlflow.get_tracking_uri()

    # with mlflow.start_run(experiment_id=experiment_id) as run:
    #     run_id = run.info.run_id
    #     # Preprocess
    #     trainset, testset = preprocess(filepath)
    #     mlflow.log_param("test_size", 0.25)

    #     # model training
    #     algo = train_model(trainset)

    #     # Save the model locally
    #     model_filename = "svd_model.pkl"
    #     joblib.dump(algo, model_filename)
    #     print(f"Model saved to {model_filename}")
    #     mlflow.log_artifact(model_filename)

    #     # predict
    #     rmse = evaluation_model(algo, testset)
    #     mlflow.log_metric("rmse", rmse)
    #     print(f"RMSE: {rmse}")
