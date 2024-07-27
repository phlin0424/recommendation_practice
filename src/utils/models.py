from abc import ABC, abstractmethod
from datareader.ml_data_base import AbstractDatas
from datareader.ml_10m_data import IntegratedDatas, PopularityDatas, BaseData
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
        self.ave_ratings = popularity_data.ave_ratings
        self.all_data = popularity_data.data

        # Split the data
        train_data, test_data = popularity_data.split_data()
        self.train_data = train_data
        self.test_data = test_data

        self.__true_ratings = self.get_true_ratings(test_data)
        self.__true_user2items = self.get_true_user2items(test_data)

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
    recommender.predict()
    metrics = recommender.evaluate()

    test_user_id = 8

    logging.info(f"test_user_id: {test_user_id}")
    print(
        np.sort(recommender.true_user2items[test_user_id]),
        len(recommender.true_user2items[test_user_id]),
    )

    print(
        recommender.pred_user2items[test_user_id],
        len(recommender.pred_user2items[test_user_id]),
    )

    correct_recommendation = len(
        set(recommender.true_user2items[test_user_id])
        & set(recommender.pred_user2items[test_user_id])
    )
    logging.info(f"correct recommendation: {correct_recommendation}")

    # 1.9092755485816106 0.0063848062833906346 0.04536842105263157
    logging.info(
        f"""
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
