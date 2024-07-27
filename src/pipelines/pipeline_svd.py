from surprise import accuracy
import asyncio
from datareader.ml_10m_data import IntegratedData, IntegratedDatas
from surprise import Dataset, Reader, SVD
import pandas as pd


async def get_data_from_db(user_num=1000):
    # fetch the data from db
    # If wanna get the ratings from all the users, use user_num=71567
    ratings = await IntegratedDatas.from_db(user_num=user_num)
    return ratings


def preprocess(user_num=1000) -> tuple[list[IntegratedData], list[IntegratedData]]:
    # Load the training data from db
    ratings = asyncio.run(get_data_from_db(user_num=user_num))

    # Split the dataset into train and test sets
    # train_data, test_data = ratings.split_data()
    traindata, testdata = ratings.split_data()

    # Convert to pandas
    traindata_df = pd.DataFrame(
        {
            "userId": [row.user_id for row in traindata],
            "movieId": [row.movie_id for row in traindata],
            "rating": [row.rating for row in traindata],
        }
    )

    testdata_df = pd.DataFrame(
        {
            "userId": [row.user_id for row in testdata],
            "movieId": [row.movie_id for row in testdata],
            "rating": [row.rating for row in testdata],
        }
    )

    # Convert to surprise dataset
    # Prepare the Reader:
    reader = Reader(rating_scale=(1, 5))

    # Load the data into Surprise datasets:
    train_data = Dataset.load_from_df(traindata_df, reader)

    # Build the trainset and testset:
    trainset = train_data.build_full_trainset()
    testset = list(testdata_df.itertuples(index=False, name=None))

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
    user_num = 1000
    trainset, testset = preprocess(user_num)
    algo = train_model(trainset)
    rmse = evaluation_model(algo, testset)
    print(rmse)
