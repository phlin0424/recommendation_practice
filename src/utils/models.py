from abc import ABC, abstractmethod
from datareader.ml_10m_data import IntegratedData
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
    def __init__(
        self, train_data: list[IntegratedData], test_data: list[IntegratedData]
    ):
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
            key=lambda item: item[1],
            reverse=True,
        )
        sorted_pred_ratings = list(dict(sorted_ratings).values())
        sorted_true_ratings = list(dict(sorted_ratings).keys())

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
            grouped_ratings[row.user_id][row.rating] = predict_rating

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
    def __init__(
        self, train_data: list[IntegratedData], test_data: list[IntegratedData]
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.pred_matrix = None

    def train():
        pass

    def predict():
        pass

    def evaluate():
        pass


if __name__ == "__main__":
    from pipelines.pipeline_random.preprocess import preprocess

    logging.info("Loading data")
    traindata, testdata = preprocess(user_num=1000)
    recommender = RandomRecommender(train_data=traindata, test_data=testdata)
    recommender.train()
    metrics = recommender.evaluate()
    # 1.911800406357343 0.0 0.0
    print(
        metrics.rmse,
        metrics.recall_at_k,
        metrics.precision_at_k,
    )
