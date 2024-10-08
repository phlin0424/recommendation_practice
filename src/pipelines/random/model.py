from collections import defaultdict

import numpy as np
from datareader.ml_10m_data import IntegratedDatas
from utils.evaluation_metrics import Metrics
from utils.helper import indices_mapper
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging

logger = configure_logging()


class RandomRecommender(BaseRecommender):
    def __init__(self, random_datas: IntegratedDatas):
        super().__init__(input_data=random_datas)

        # Initialize the prediction matrix
        self.pred_matrix = None

    def train(self):
        logger.info("RandomRecommender: Training")
        # Fetch an unique list of the user id & movie id.
        unique_user_ids = sorted(set([row.user_id for row in self.train_data]))
        unique_movie_ids = sorted(set([row.movie_id for row in self.train_data]))

        # Create a predicted ratings matrix using random numbers
        # This matrix would be served as the training result
        pred_matrix = np.random.uniform(
            0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids))
        )

        # Save the prediction result
        self.pred_matrix = pred_matrix

        # Create a user_id to dict and a movie_id to ind  for the predictor to refer to:
        self.user_id_indices, _ = indices_mapper(
            input_data=self.train_data, id_col_name="user_id", reverse=False
        )
        self.movie_id_indices, self.indices_movie_id = indices_mapper(
            input_data=self.train_data, id_col_name="movie_id", reverse=True
        )

    def _predict_rating(
        self, user_id: int, movie_id: int | None = None
    ) -> float | list[float]:
        """Make a prediction for a given user_id and a movie_id

        Args:
            user_id (int): _description_
            movie_id (int | None, optional): _description_. Defaults to None.

        Raises:
            RuntimeError: _description_

        Returns:
            float | list[float]: _description_
        """

        if self.pred_matrix is None:
            raise RuntimeError("Train the model fist")

        user_id_index = self.user_id_indices[user_id]
        if movie_id is not None:
            movie_id_index = self.movie_id_indices[movie_id]
            return self.pred_matrix[user_id_index, movie_id_index]
        else:
            return self.pred_matrix[user_id_index, :]

    def predict(self) -> None:
        """Predict the ratings& recommended movies to the specific user."""
        logger.info("RandomRecommender: Predicting")
        # Extract the test data
        test_data = self.test_data

        # Derive a list of predicted rating values according to the list of user_id and movie_id
        predict_ratings = []
        # Predict the ratings using random method
        for row in test_data:
            # If the specified movie_id-user_id pair doesn't exist in the training data,
            # fill in with a random number.
            if row.movie_id not in self.movie_id_indices.keys():
                predict_rating = np.random.uniform(0.5, 5)
            else:
                predict_rating = self._predict_rating(row.user_id, row.movie_id)

            # Collect the predicted & true ratings
            predict_ratings.append(predict_rating)

        # Derive a list of dict of a list of movie_id being recommended to the user
        unique_user_ids_test = sorted(set([row.user_id for row in self.train_data]))
        pred_user2items = defaultdict(list)

        for user_id in unique_user_ids_test:
            # Collect the movie id rated by each user
            all_movie_id_rating = self._predict_rating(user_id)

            # Collect the recommended movie_ids of specific user
            movie_indexes_sorted = np.argsort(-all_movie_id_rating)
            for movie_index in movie_indexes_sorted:
                movie_id = self.indices_movie_id[movie_index]
                if movie_id not in pred_user2items[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        # Save the prediction result
        self._pred_ratings = predict_ratings
        self._pred_user2items = pred_user2items

    def evaluate(self) -> Metrics:
        """Evaluate the predicted recommendation result.

        Returns:
            Metrics: Evaluation results of the prediction.
        """
        logger.info("RandomRecommender: Calculating metrics")
        metrics = Metrics.from_ml_10m_data(
            true_ratings=self._true_ratings,
            pred_ratings=self._pred_ratings,
            true_user2items=self._true_user2items,
            pred_user2items=self._pred_user2items,
        )
        return metrics


if __name__ == "__main__":
    import asyncio

    input_data = asyncio.run(IntegratedDatas.from_db())
    # print(input_data.ave_ratings[0:10])
    recommender = RandomRecommender(input_data)
    recommender.train()
    recommender.predict()
    metrics = recommender.evaluate()

    print(recommender.pred_user2items[8])

    logger.info(
        f"""
        model: RandomRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
