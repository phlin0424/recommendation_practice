import asyncio
from datareader.ml_10m_data import Ratings, Rating


async def get_data_from_db():
    # fetch the data from db
    ratings = await Ratings.from_db()
    return ratings


def preprocess(test_size) -> tuple[list[Rating], list[Rating]]:
    # Load the training data from db
    ratings = asyncio.run(get_data_from_db())

    # Split the dataset into train and test sets
    # train_data, test_data = ratings.split_data()
    traindata, testdata = ratings.split_data(test_size=test_size)

    return traindata, testdata
