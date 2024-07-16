import asyncio
from datareader.ml_10m_data import IntegratedData, IntegratedDatas


async def get_data_from_db(user_num=1000):
    # fetch the data from db
    # If wanna get the ratings from all the users, use user_num=71567
    ratings = await IntegratedDatas.from_db(user_num=user_num)
    return ratings


def preprocess(
    test_size=0.25, user_num=1000
) -> tuple[list[IntegratedData], list[IntegratedData]]:
    # Load the training data from db
    ratings = asyncio.run(get_data_from_db(user_num=user_num))

    # Split the dataset into train and test sets
    # train_data, test_data = ratings.split_data()
    traindata, testdata = ratings.split_data(test_size=test_size)

    return traindata, testdata


if __name__ == "__main__":
    traindata, testdata = preprocess(test_size=0.25, user_num=1000)
    print(traindata[0])
    print(traindata[1])
    print(traindata[2])
