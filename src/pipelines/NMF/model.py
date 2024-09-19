from collections import defaultdict

import numpy as np
from datareader.ml_10m_data import IntegratedDatas
from sklearn.decomposition import NMF
from utils.evaluation_metrics import Metrics
from utils.helper import indices_mapper
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging

logger = configure_logging()


# Non-negative Matrix Factorization
class NMFRecommender(BaseRecommender):
    def __init__(self, input_data: IntegratedDatas):
        super().__init__(input_data=input_data)

    def _get_unique_ids(self):
        # Get the unique user id list
        self.unique_user_ids = sorted(set([row.user_id for row in self.train_data]))

        # Get the unique movie id list
        self.unique_movie_ids = sorted(set([row.movie_id for row in self.train_data]))

    def _get_indices(self):
        self.user_id_indices, _ = indices_mapper(
            input_data=self.train_data, id_col_name="user_id", reverse=False
        )
        self.movie_id_indices, self.indices_movie_id = indices_mapper(
            input_data=self.train_data, id_col_name="movie_id", reverse=True
        )

    def _create_rating_matrix(self):
        train_rating_matrix = np.full(
            (len(self.unique_user_ids), len(self.unique_movie_ids)), np.nan
        )

        for row in self.train_data:
            train_rating_matrix[
                self.user_id_indices[row.user_id], self.movie_id_indices[row.movie_id]
            ] = row.rating

        self.train_rating_matrix = train_rating_matrix

    def train(self, fill_with_zero=False):
        logger.info("NMFRecommender: Training")

        self._get_unique_ids()
        self._get_indices()
        self._create_rating_matrix()

        # Fill nan with specified number
        train_rating_mean = np.mean([row.rating for row in self.train_data])
        self.rating_mean = train_rating_mean
        m = self.train_rating_matrix.copy()
        if not fill_with_zero:
            m[np.isnan(m)] = train_rating_mean
        else:
            m[np.isnan(m)] = 0
        # Apply NMF
        nmf = NMF(n_components=5)
        nmf.fit(m)
        P = nmf.fit_transform(m)
        Q = nmf.components_

        self.pred_matrix = np.dot(P, Q)

    def _predict_single_user(self, user_id, movie_id):
        if user_id not in self.user_id_indices or movie_id not in self.movie_id_indices:
            return self.rating_mean

        user_index = self.user_id_indices[user_id]
        movie_index = self.movie_id_indices[movie_id]
        return self.pred_matrix[user_index, movie_index]

    def _predict_all_movie_rating(
        self, user_id: int, movie_id: int | None = None
    ) -> float | list[float]:
        if self.pred_matrix is None:
            raise RuntimeError("Train the model fist")

        user_id_index = self.user_id_indices[user_id]
        if movie_id is not None:
            movie_id_index = self.movie_id_indices[movie_id]
            return self.pred_matrix[user_id_index, movie_id_index]
        else:
            return self.pred_matrix[user_id_index, :]

    def predict(self):
        logger.info("NMFRecommender: Predicting")

        # predict: all ratings among all the users
        predict_ratings = []
        for row in self.test_data:
            predict_ratings.append(self._predict_single_user(row.user_id, row.movie_id))

        # Prepare the unique user list which we want to predict:
        unique_user_ids_test = sorted(set([row.user_id for row in self.train_data]))

        # predict: the most liked movies for each user based on the inferred rating predictions
        pred_user2items = defaultdict(list)

        for user_id in unique_user_ids_test:
            # Collect the movie id rated by each user
            all_movie_id_rating = self._predict_all_movie_rating(user_id)

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
        logger.info("NMFRecommender: Calculating metrics")
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
    recommender = NMFRecommender(input_data)
    recommender.train()
    recommender.predict()
    metrics = recommender.evaluate()

    print(recommender.pred_user2items[8])

    logger.info(
        f"""
        model: NMFRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )

    # 2024-09-01 12:41:59,888 model.utils.pipeline_logging [INFO] NMFRecommender: Training
    # 2024-09-01 12:42:06,266 model.utils.pipeline_logging [INFO] NMFRecommender: Predicting
    # 2024-09-01 12:42:06,720 model.utils.pipeline_logging [INFO] NMFRecommender: Calculating metrics
    # [2571, 296, 1210, 2959, 260, 1196, 110, 318, 2858, 593]
    # 2024-09-01 12:42:06,722 model.utils.pipeline_logging [INFO]
    #         model: NMFRecommender
    #         rmse: 1.062924470118066,
    #         recall_at_k: 0.032114735658042744,
    #         precision_at_k:0.010911136107986504
