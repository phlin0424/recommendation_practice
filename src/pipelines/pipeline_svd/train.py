from surprise import SVD


def train_model(trainset):
    algo = SVD()
    algo.fit(trainset)
    return algo
