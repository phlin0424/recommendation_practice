from abc import ABC, abstractmethod
from datareader.ml_10m_data import (
    IntegratedDatas,
    PopularityDatas,
)
import numpy as np
from utils.evaluation_metrics import Metrics
from collections import defaultdict
import logging

# Set up log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


np.random.seed(0)


class BaseRecommender(ABC):
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


class RandomRecommender(BaseRecommender):
    def __init__(self, random_datas: IntegratedDatas):
        train_data, test_data = random_datas.split_data()
        self.train_data = train_data
        self.test_data = test_data
        self.pred_matrix = None

    def train(self):
        logging.info("RandomRecommender: Training")
        # Fetch an unique list of the user id & movie id.
        unique_user_ids = sorted(set([row.user_id for row in self.train_data]))
        unique_movie_ids = sorted(set([row.movie_id for row in self.train_data]))

        # Create a predicted ratings matrix using random numbers
        # This matrix would be served as the training result
        pred_matrix = np.random.uniform(
            0.5, 5, (len(unique_user_ids), len(unique_movie_ids))
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

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating according to the given user_id and movie_id

        Args:
            user_id (int): _description_
            movie_id (int): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            float: The predicted rating of a item given by the specified user.
        """

        if self.pred_matrix is None:
            raise RuntimeError("Train the model fist")

        user_id_index = self.user_id_indices[user_id]
        movie_id_index = self.movie_id_indices[movie_id]
        return self.pred_matrix[user_id_index, movie_id_index]

    @staticmethod
    def _sort_true_by_preds(
        rating_of_user: dict[int, float],
    ) -> tuple[list[float], list[int]]:
        """Giving a user's rating data (both true and predicted values),
            sort the true ratings by the predicted value.

        Args:
            rating_of_user (dict[int, float]): A dict that contains true-pred rating pairs.

        Returns:
            tuple[list[float], list[int]]: the sorted "pred" value & the sorted "true" value
        """
        ratings_items = list(rating_of_user.items())
        sorted_ratings = sorted(
            ratings_items,
            key=lambda item: item[1][0],
            reverse=True,
        )
        sorted_pred_ratings = [rating[1][0] for rating in sorted_ratings]
        sorted_true_ratings = [rating[1][1] for rating in sorted_ratings]

        return sorted_pred_ratings, sorted_true_ratings

    @property
    def true_ratings(self):
        return self.__true_ratings

    @property
    def pred_ratings(self):
        return self.__pred_ratings

    @property
    def true_user2items(self):
        return self.__true_user2items

    @property
    def pred_user2items(self):
        return self.__pred_user2items

    def evaluate(self) -> Metrics:
        """Evaluate the predicted recommendation result.

        Returns:
            Metrics: Evaluation results of the prediction.
        """
        logging.info("RandomRecommender: Evaluating")
        # Extract the test data
        test_data = self.test_data

        # Derive a list of predicted rating values according to the list of user_id and movie_id
        # Derive true_ratings & pred_ratings to calculate rmse
        predict_ratings = []
        true_ratings = []

        # Create a nested dictionary to store the grouped data
        grouped_ratings = defaultdict(lambda: defaultdict())

        # Predict the ratings using random method
        for row in test_data:
            # If the specified movie_id-user_id pair doesn't exist, fill in with a random number.
            if row.movie_id not in self.movie_id_indices.keys():
                predict_rating = np.random.uniform(0.5, 5)
            else:
                predict_rating = self.predict(row.user_id, row.movie_id)

            # Collect the predicted & true ratings
            predict_ratings.append(predict_rating)
            true_ratings.append(row.rating)

            # Collect the movie id rated by each user
            grouped_ratings[row.user_id][row.movie_id] = [predict_rating, row.rating]

        # Create true_user2items and pred_user2items to calculate precision_at_k and recall_at_k
        true_user2items = {}
        pred_user2items = {}
        for user_id in grouped_ratings.keys():
            # Sort the movie-id ratings by users
            ratings_items = grouped_ratings[user_id]
            sorted_ratings_pred, sorted_ratings_true = self._sort_true_by_preds(
                ratings_items
            )
            pred_user2items[user_id], true_user2items[user_id] = (
                sorted_ratings_pred,
                sorted_ratings_true,
            )

        print("true_user2items", true_user2items[1], true_user2items[2])
        print("pred_user2items", pred_user2items[1], pred_user2items[2])
        # Save the data
        self.__true_ratings = true_ratings
        self.__pred_ratings = predict_ratings
        self.__true_user2items = true_user2items
        self.__pred_user2items = pred_user2items

        logging.info("RandomRecommender: Calculating metrics")
        metrics = Metrics.from_ml_10m_data(
            true_ratings=true_ratings,
            pred_ratings=predict_ratings,
            true_user2items=true_user2items,
            pred_user2items=pred_user2items,
        )
        return metrics


class PopularityRecommender:
    def __init__(self, popularity_data: PopularityDatas):
        # Load the necessary data from the input model base
        self.ave_ratings = popularity_data.ave_ratings
        self.all_data = popularity_data.data

        # Split the data
        train_data, test_data = popularity_data.split_data()
        self.train_data = train_data
        self.test_data = test_data

    def train(self):
        pass

    @property
    def true_ratings(self):
        return self.__true_ratings

    @property
    def pred_ratings(self):
        return self.__pred_ratings

    @property
    def true_user2items(self):
        return self.__true_user2items

    @property
    def pred_user2items(self):
        return self.__pred_user2items

    def _get_true_user2items(self):
        # List of the user id in test data
        test_user_id = [item.user_id for item in self.test_data]

        # Derive the real ranking of the movie rating for each user
        true_user2items_all = defaultdict(list)
        for row in self.all_data:
            true_user2items_all[row.user_id].append(row.movie_id)

        for user_id in true_user2items_all:
            true_user2items_all[user_id].sort()

        true_user2items = {
            user_id: true_user2items_all[user_id] for user_id in test_user_id
        }

        self.__true_user2items = true_user2items

    def predict(self):
        ave_ratings = self.ave_ratings
        test_data = self.test_data

        # Drive the sorted movie ranking list
        # ranking from the most liked to the less
        sorted_movie_ids = [item.movie_id for item in ave_ratings]

        # Initialize a defaultdict to store lists of movie_ids for each user
        user_movies = defaultdict(list)

        # Iterate through the test_data and populate the defaultdict
        for row in self.test_data:
            user_movies[row.user_id].append(row.movie_id)

        # Prediction-1: predict the items
        # Iterate the movie_id over all the user-movie-id list (rated)
        # if the movie_id hasn't been rated by the user yet, append it based on the
        # training result (which is, the move popularity ranking)
        pred_user2items = defaultdict(list)
        for user_id in user_movies.keys():
            for movie_id in sorted_movie_ids:
                if movie_id not in user_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        # Prediction-2: predicted ratings of each movie from every user
        # The movies which is not included in the test data
        # Predict the ratings using random method
        # Create a lookup dictionary for ave_ratings by movie_id
        movie_id_to_ave_rating = {
            item.movie_id: item.ave_rating for item in ave_ratings
        }

        predict_ratings = []
        true_ratings = []
        for row in test_data:
            # If the specified movie_id-user_id pair doesn't exist, fill in 0.
            predict_rating = movie_id_to_ave_rating.get(row.movie_id, 0)

            # Collect the predicted & true ratings
            predict_ratings.append(predict_rating)
            true_ratings.append(row.rating)

        # Save the results
        self.__true_ratings = true_ratings
        self.__pred_ratings = predict_ratings
        self.__pred_user2items = pred_user2items

    def evaluate(self) -> Metrics:
        self._get_true_user2items()
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

    async def _read_data():
        movies = await IntegratedDatas.from_db()
        return movies

    input_data = asyncio.run(_read_data())
    recommender = RandomRecommender(input_data)

    recommender.train()
    metrics = recommender.evaluate()

    print(recommender.true_user2items[10])
    print(recommender.pred_user2items[10])

    # 1.911800406357343 0.0 0.0
    print(
        metrics.rmse,
        metrics.recall_at_k,
        metrics.precision_at_k,
    )
