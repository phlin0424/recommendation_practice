from utils.models import BaseRecommender
from datareader.ml_10m_data import IntegratedDatas, IntegratedData
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict
import numpy as np
from collections import Counter
from utils.evaluation_metrics import Metrics
from utils.pipeline_logging import configure_logging


logger = configure_logging()


class AssociationRecommender(BaseRecommender):
    def __init__(
        self,
        random_datas: IntegratedDatas,
        min_support: float = 0.1,
    ):
        super().__init__(input_data=random_datas)

        # Initialize the prediction matrix
        self.rating_matrix = None

        # Save the arguments into the instance variables
        self.min_support = min_support

        # Convert train data to pandas dataframe
        self.train_df = pd.DataFrame(
            [rating.model_dump() for rating in self.train_data]
        )

    def _create_rating_matrix(self):
        matrix = self.train_df.pivot(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        self.rating_matrix = matrix

    def _binary_mask_matrix(self, threshold: int = 4):
        matrix = self.rating_matrix
        matrix[matrix.isnull()] = 0
        matrix[matrix < threshold] = 0
        matrix[matrix >= threshold] = 1

        self.rating_matrix = matrix

    def train(self):
        logger.info("AssociationRecommender: Training")

        # Create tje user_id-movie_id matrix
        self._create_rating_matrix()

        # Mask the items with rating <4
        self._binary_mask_matrix()

        # Calculate support to movie_id
        freq_movies = apriori(
            self.rating_matrix, min_support=self.min_support, use_colnames=True
        )

        # Calculate antecedents and consequents:
        rules = association_rules(freq_movies, metric="lift", min_threshold=1)

        # Use the gained antecedents and consequents as the training result
        self.rules = rules

    def _get_watched_list(self) -> dict[int, list[int]]:
        # Get the watched list for each user
        user_watched_movies = defaultdict(list)
        for row in self.train_data:
            user_watched_movies[row.user_id].append(row.movie_id)
        self.user_watched_movies = user_watched_movies

    def _create_high_ratted_movie_list(self):
        high_rated_list = []
        for row in self.train_data:
            if row.rating >= 4:
                high_rated_list.append(row)

        self.high_rated_list = high_rated_list

    @staticmethod
    def group_by_user_id(
        integarated_data: list[IntegratedData],
    ) -> dict[int, list]:
        """Collect the items group by each user.

        Args:
            integarated_data (list[IntegratedData]): _description_

        Returns:
            dict[int, list]: _description_
        """

        user2items = defaultdict(list)
        for row in integarated_data:
            user2items[row.user_id].append([row.movie_id, row.timestamp])

        return user2items

    @staticmethod
    def sort_movie_id(user2items: dict[int, list[list]]) -> dict[int, list[int]]:
        """Fetch the last 5 rated items per users.

        Args:
            user2items (dict[int, list[list]]): _description_

        Returns:
            dict[int, list[int]]: last 5 items for each user.
        """
        for user_id, movie_ids in user2items.items():
            # Sort the list of movie_id by the timestamp (second element of each sublist)
            movie_ids.sort(key=lambda x: x[1], reverse=True)
            sorted_movie_ids = user2items[user_id] = [item[0] for item in movie_ids]
            user2items[user_id] = sorted_movie_ids[-5:]

        return user2items

    def predict(self):
        logger.info("AssociationRecommender: Predicting")

        self._create_high_ratted_movie_list()

        # Get the list of rated movies for each users
        self._get_watched_list()

        # Get the last 5 items for each user:
        user2items = self.group_by_user_id(self.train_data)
        user2items_last5 = self.sort_movie_id(user2items)

        # Get the antecedents values
        antecedents_rule_list = self.rules["antecedents"].values
        consequents_rule_list = self.rules["consequents"].values
        lift_rule_list = self.rules["lift"].values

        # Initialize pred_user2items
        pred_user2items = defaultdict(list)

        for user_id, movie_ids in user2items_last5.items():
            # Fetch the last 5 rated items for the user
            input_data = movie_ids

            # In the rule table, find the row that has at least one item matches antecedents
            matchted_flg = [
                True if len(set(input_data) & item) >= 1 else False
                for item in antecedents_rule_list
            ]

            # Find the related consequents and sort it by the lift value
            consequent_movies = consequents_rule_list[matchted_flg]
            lift_movies = lift_rule_list[matchted_flg]
            sort_ind = np.argsort(lift_movies)
            consequent_movies_sorted = consequent_movies[sort_ind]

            # Find the movies that appears at `consequent_movies_sorted` list most frequently
            consequent_counter = Counter(consequent_movies_sorted)
            for movie_id, movie_cnt in consequent_counter.most_common():
                if movie_id not in self.user_watched_movies[user_id]:
                    pred_user2items[user_id].extend(movie_id)

                # Only recommend the first 10 movies
                if pred_user2items[user_id].extend(movie_id):
                    break

        self._pred_user2items = pred_user2items
        # we dont evaluate rmse in this recommender
        self._pred_ratings = np.zeros((len(self._true_ratings), 1))

    def evaluate(self) -> Metrics:
        logger.info("AssociationRecommender: Calculating metrics")
        metrics = Metrics.from_ml_10m_data(
            true_ratings=self.true_ratings,
            pred_ratings=self.pred_ratings,
            true_user2items=self.true_user2items,
            pred_user2items=self.pred_user2items,
        )
        return metrics


if __name__ == "__main__":
    import asyncio

    input_data = asyncio.run(IntegratedDatas.from_db())
    # print(input_data.ave_ratings[0:10])
    recommender = AssociationRecommender(input_data)
    recommender.train()
    recommender.predict()
    metrics = recommender.evaluate()

    print(recommender.pred_user2items[8])

    logger.info(
        f"""
        model: AssociationRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
