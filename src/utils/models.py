from abc import ABC, abstractmethod
from datareader.ml_data_base import AbstractDatas
from datareader.ml_10m_data import (
    IntegratedDatas,
    PopularityDatas,
    BaseData,
    IntegratedData,
    PopularityAveRating,
)
import numpy as np
from utils.evaluation_metrics import Metrics
from collections import defaultdict
import logging

# Set up log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(module)s: %(message)s",
)
logger = logging.getLogger(__name__)


np.random.seed(0)


class BaseRecommender(ABC):
    def __init__(self, input_data: AbstractDatas):
        # Split the data into train& test
        train_data, test_data = input_data.split_data()
        self.train_data = train_data
        self.test_data = test_data

        # Get the ground truth data
        self._true_ratings = self.get_true_ratings(test_data)
        self._true_user2items = self.get_true_user2items(test_data)

        # Initialize the predictions
        self._pred_ratings = None
        self._pred_user2items = None

    @abstractmethod
    def train():
        """Training process"""
        pass

    @abstractmethod
    def evaluate() -> Metrics:
        """Evaluate the training result"""
        pass

    @abstractmethod
    def predict():
        """Recommend items or predict ratings based on a given user"""
        pass

    @staticmethod
    def get_true_ratings(test_data: list[BaseData]) -> list[int | float]:
        """Get the ground truth: true_ratings

        Args:
            test_data (list[BaseData]): _description_

        Returns:
            list[int | float]: _description_
        """
        true_rating = []
        for row in test_data:
            true_rating.append(row.rating)
        return true_rating

    @staticmethod
    def get_true_user2items(test_data: list[BaseData]) -> dict[int, list[int]]:
        """Get the ground truth: true user2items

        Args:
            test_data (list[BaseData]): _description_

        Returns:
            dict[int, list[int]]: _description_
        """
        true_user2items = defaultdict(list)
        for row in test_data:
            if row.rating >= 4:
                true_user2items[row.user_id].append(row.movie_id)
        return true_user2items

    @property
    def true_ratings(self):
        return self._true_ratings

    @property
    def pred_ratings(self):
        return self._pred_ratings

    @property
    def true_user2items(self):
        return self._true_user2items

    @property
    def pred_user2items(self):
        return self._pred_user2items


class RandomRecommender(BaseRecommender):
    def __init__(self, random_datas: IntegratedDatas):
        super().__init__(input_data=random_datas)

        # Initialize the prediction matrix
        self.pred_matrix = None

    def train(self):
        logging.info("RandomRecommender: Training")
        # Fetch an unique list of the user id & movie id.
        unique_user_ids = sorted(set([row.user_id for row in self.train_data]))
        unique_movie_ids = sorted(set([row.movie_id for row in self.train_data]))

        self.unique_user_ids = unique_user_ids

        # Create a predicted ratings matrix using random numbers
        # This matrix would be served as the training result
        pred_matrix = np.random.uniform(
            0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids))
        )

        # Save the prediction result
        self.pred_matrix = pred_matrix

        # Create a user_id - index dict for the predictor to refer to:
        # Usage: index (to the matrix)= user_id_indices[user_id]
        user_id_indices = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        self.user_id_indices = user_id_indices

        # Create a movie_id - index dict for the predictor to refer to:
        # Usage: index (to the matrix)= movie_id_indices[movie_id]
        movie_id_indices = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))
        self.movie_id_indices = movie_id_indices

    def _predict_rating(
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

    def predict(self) -> None:
        """Predict the ratings& recommended movies to the specific user."""
        logging.info("RandomRecommender: Predicting")
        # Extract the test data
        test_data = self.test_data

        # Derive a list of predicted rating values according to the list of user_id and movie_id
        predict_ratings = []

        # Derive a list of dict of a list of movie_id being recommended to the user
        pred_user2items = defaultdict(list)

        # Create a reverse dictionary to referring movie_id from the index
        index_to_movie_id = {
            index: movie_id for movie_id, index in self.movie_id_indices.items()
        }

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

        for user_id in self.unique_user_ids:
            # Collect the movie id rated by each user
            all_movie_id_rating = self._predict_rating(user_id)

            # Collect the recommended movie_ids of specific user
            movie_indexes_sorted = np.argsort(-all_movie_id_rating)
            for movie_index in movie_indexes_sorted:
                movie_id = index_to_movie_id[movie_index]
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
        logging.info("RandomRecommender: Calculating metrics")
        metrics = Metrics.from_ml_10m_data(
            true_ratings=self._true_ratings,
            pred_ratings=self._pred_ratings,
            true_user2items=self._true_user2items,
            pred_user2items=self._pred_user2items,
        )
        return metrics


class PopularityRecommender(BaseRecommender):
    def __init__(self, popularity_data: PopularityDatas):
        # Load the necessary data from the input model base
        # Load the average rating of all the movies
        self.ave_ratings = popularity_data.ave_ratings

        super().__init__(input_data=popularity_data)

    def train(self):
        pass

    def _get_watched_list(self) -> dict[int, list[int]]:
        # Get the watched list for each user
        user_watched_movies = defaultdict(list)
        for row in self.train_data:
            user_watched_movies[row.user_id].append(row.movie_id)
        self.user_watched_movies = user_watched_movies

    @staticmethod
    def _predict_user2items(
        test_data: list[IntegratedData],
        ave_ratings: list[PopularityAveRating],
        user_watched_movies: dict[int, list[int]],
        threshold: int = 10,
    ) -> dict[int, list[float]]:
        # Drive the sorted movie_id list, which is sorted by rating
        movie_ids_sorted_by_ave = [
            item.movie_id for item in ave_ratings if item.rated_movies_count > threshold
        ]

        # Get the user_id list
        unique_user_ids = sorted(set([row.user_id for row in test_data]), reverse=True)

        # Predict the recommended movies:
        pred_user2items = defaultdict(list)
        for user_id in unique_user_ids:
            for movie_id in movie_ids_sorted_by_ave:
                if movie_id not in user_watched_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        return pred_user2items

    @staticmethod
    def _predict_ratings(
        test_data: list[IntegratedData], ave_ratings: list[PopularityAveRating]
    ) -> list[float]:
        movie_id_to_ave_rating = {
            item.movie_id: item.ave_rating for item in ave_ratings
        }
        pred_ratings = []
        for row in test_data:
            pred_ratings.append(movie_id_to_ave_rating.get(row.movie_id, 0))
        return pred_ratings

    def predict(self, threshold: int = 10):
        logging.info("PopularityRecommender: Predicting...")
        ave_ratings = self.ave_ratings
        test_data = self.test_data

        # Prediction-1: predict item recommendations for a specific user
        self._get_watched_list()
        pred_user2items = self._predict_user2items(
            test_data, ave_ratings, self.user_watched_movies, threshold
        )

        # Prediction-2: predicted all the ratings
        predict_ratings = self._predict_ratings(test_data, ave_ratings)

        # Save the results
        self._pred_ratings = predict_ratings
        self._pred_user2items = pred_user2items

    def evaluate(self) -> Metrics:
        logging.info("PopularityRecommender: Calculating metrics")
        metrics = Metrics.from_ml_10m_data(
            true_ratings=self.true_ratings,
            pred_ratings=self.pred_ratings,
            true_user2items=self.true_user2items,
            pred_user2items=self.pred_user2items,
        )
        return metrics


if __name__ == "__main__":
    import asyncio

    # input_data = asyncio.run(IntegratedDatas.from_db())
    # recommender = RandomRecommender(input_data)
    # recommender.train()
    # recommender.predict()
    # metrics = recommender.evaluate()
    # logging.info(
    #     f"""
    #     model: RandomRecommender
    #     rmse: {metrics.rmse},
    #     recall_at_k: {metrics.recall_at_k},
    #     precision_at_k:{metrics.precision_at_k}
    #     """
    # )
    # # rmse: 1.866140822115631,
    # # recall_at_k: 0.001639344262295082,
    # # precision_at_k:0.00032786885245901645

    input_data = asyncio.run(PopularityDatas.from_db())
    print(input_data.ave_ratings[0:10])
    recommender = PopularityRecommender(input_data)
    recommender.train()
    recommender.predict(threshold=100)
    metrics = recommender.evaluate()

    print(recommender.pred_user2items[8])

    logging.info(
        f"""
        model: PopularityRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
