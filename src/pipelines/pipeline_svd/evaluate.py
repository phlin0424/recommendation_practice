from surprise import accuracy


def evaluation_model(algo, testset):
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    return rmse
