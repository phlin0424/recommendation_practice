from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from datareader.ml_10m_data import BaseData
from datareader.ml_data_base import AbstractDatas
from utils.evaluation_metrics import Metrics
from utils.pipeline_logging import configure_logging

# Set up log
logger = configure_logging()

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
