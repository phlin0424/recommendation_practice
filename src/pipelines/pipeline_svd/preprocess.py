from datareader.ml_1m_data import Ratings
import asyncio
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


async def get_data_from_db():
    # fetch the data from db
    ratings = await Ratings.from_db()
    return ratings.data


def preprocess(test_size):
    # Load the training data from db
    ratings = asyncio.run(get_data_from_db())

    # Convert the training data to pd.Dataframe
    ratings_df = pd.DataFrame([rating.model_dump() for rating in ratings])

    # Define a Reader object
    reader = Reader(rating_scale=(1, 5))

    # Load the data into a Surprise dataset
    data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)

    # Split the dataset into train and test sets
    trainset, testset = train_test_split(data, test_size=test_size)

    return trainset, testset
