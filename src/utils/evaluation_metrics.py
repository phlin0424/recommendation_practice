from sklearn.metrics import mean_squared_error
import numpy as np
from pydantic import BaseModel


class Metrics(BaseModel):
    rmse: float
    recall_at_k: float
    precision_at_k: float

    @classmethod
    def from_ml_10m_data(
        cls,
        true_ratings: list[float] | np.ndarray,
        pred_ratings: list[float] | np.ndarray,
        true_user2items: dict[int, list[int]] | dict[int, np.ndarray],
        pred_user2items: dict[int, list[int]] | dict[int, np.ndarray],
        k: int = 10,
    ) -> "Metrics":
        rms = cls.cal_rmse(true_ratings, pred_ratings)
        recall_at_k = cls.calc_recall_at_k(true_user2items, pred_user2items, k)
        precision_at_k = cls.calc_precision_at_k(true_user2items, pred_user2items, k)
        return cls(
            rmse=rms,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
        )

    @staticmethod
    def cal_rmse(true_ratings: list[float], pred_ratings: list[float]) -> float:
        """Calculate Root mean squared error

        Args:
            true_rating (list[float]): The true rating data.
            pred_rating (list[float]): The predicted rating data.

        Returns:
            RMSE (float): the rmse value. the smaller the metric value is the better the prediction is.
        """
        return np.sqrt(mean_squared_error(true_ratings, pred_ratings))

    @staticmethod
    def _recall_at_k(true_items: list[int], pred_items: list[int], k: int) -> float:
        """Calculate the recall at K (for given K recommended items, how many items are correctly predicted) for each user.

        Args:
            true_items (list[int]): A list of the item id that user really likes.
            pred_items (list[int]): A list of the item id that got recommended.
            k (int): _description_

        Returns:
            float: recall at K value.
        """
        if len(true_items) == 0 or k == 0:
            return 0.0

        r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)

        return r_at_k

    @staticmethod
    def calc_recall_at_k(
        true_user2items: dict[int, list[int]],
        pred_user2items: dict[int, list[int]],
        k: int,
    ) -> float:
        """Calculate the recall at k value at each user, averaging over all the users.

        Args:
            true_user2items (dict[int, list[int]]): A dict of the user& item_id list. value of the dict: A list of a item id that a user like.
            pred_user2items (dict[int, list[int]]): recommended items of every users.
            k (int): _description_

        Returns:
            float: recall at k value over all the users.
        """

        scores = []
        for user_id in true_user2items.keys():
            r_at_k = Metrics._recall_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(r_at_k)

        return np.mean(scores)

    @staticmethod
    def _precision_at_k(true_items: list[int], pred_items: list[int], k: int) -> float:
        """Calculate the precision at k value of a certain user.

        Args:
            true_items (list[int]): item id list that the user really likes.
            pred_items (list[int]): items id that the user got recommended.
            k (int): _description_

        Returns:
            float: precision at k value of the specific user.
        """
        if k == 0:
            return 0.0
        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
        return p_at_k

    @staticmethod
    def calc_precision_at_k(
        true_user2items: dict[int, list[int]],
        pred_user2items: dict[int, list[int]],
        k: int,
    ) -> float:
        """Calculate the precision at k value over all the users and get the averaged values.

        Args:
            true_user2items (dict[int, list[int]]): _description_
            pred_user2items (dict[int, list[int]]): _description_
            k (int): _description_

        Returns:
            float: precision at k value averaging over all the users.
        """

        scores = []
        for user_id in true_user2items.keys():
            p_at_k = Metrics._precision_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(p_at_k)
        return np.mean(scores)


if __name__ == "__main__":
    true_ratings = [1, 4, 5, 5, 5]
    pred_ratings = [2, 4, 5, 5, 5]
    true_items = [1, 3, 5, 6, 10, 11]
    pred_items = [3, 5, 6, 13, 11, 12]
    true_items2 = [5, 3, 5, 6, 10, 11]
    pred_items2 = [3, 5, 6, 20, 11, 12]
    true_user2items = {1: true_items, 2: true_items2}
    pred_user2items = {1: pred_items, 2: pred_items2}
    metrics = Metrics.from_ml_10m_data(
        true_ratings=true_ratings,
        pred_ratings=pred_ratings,
        true_user2items=true_user2items,
        pred_user2items=pred_user2items,
    )
    print(metrics.rmse)
    print(metrics.recall_at_k)
    print(metrics.precision_at_k)
