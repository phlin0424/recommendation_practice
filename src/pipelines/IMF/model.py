from collections import defaultdict

import implicit
from datareader.ml_10m_data import IntegratedData, IntegratedDatas
from scipy.sparse import coo_matrix
from utils.evaluation_metrics import Metrics
from utils.helper import indices_mapper
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging

logger = configure_logging()


class IMFRecommender(BaseRecommender):
    def __init__(self, input_data: IntegratedDatas):
        super().__init__(input_data)

    @staticmethod
    def _filter_data(
        train_data: list[IntegratedData], minimum_num_rating=1
    ) -> list[IntegratedData]:
        """filter the data by the following two criteria:
            (1): Only select the movie id that have been rated more than once
            (2): Only select the data with ratings higher than the threshold

        Args:
            train_data (list[IntegratedData]): _description_
            minimum_num_rating (int, optional): _description_. Defaults to 1.

        Returns:
            list[IntegratedData]: _description_
        """
        group_by_movies = defaultdict(list)
        for item in train_data:
            group_by_movies[item.movie_id].append(item)

        filtered_data = [
            item
            for items in group_by_movies.values()
            if len(items) >= 2
            for item in items
        ]

        movielens_train_high_rating = [
            item for item in filtered_data if item.rating >= minimum_num_rating
        ]

        return movielens_train_high_rating

    def preprocess(
        self,
        minimum_num_rating=1,
    ):
        # Filter the original data with the following two criteria:
        # (1) movies id with more than two rating data
        # (2) only select the data with rating > minimum_num_rating
        filtered_train_data = self._filter_data(
            self.train_data, minimum_num_rating=minimum_num_rating
        )
        self.filtered_train_data = filtered_train_data

    @staticmethod
    def _extract_col(
        input_data: list[IntegratedData], col_name: str
    ) -> list[int | float]:
        return [item.get(col_name) for item in input_data]

    def _get_interaction_matrix(self):
        """create an interaction matrix"""
        self._get_indices(input_data=self.filtered_train_data)

        interaction_list = self._extract_col(self.filtered_train_data, "rating")
        user_id_list = self._extract_col(self.filtered_train_data, "user_id")
        movie_id_list = self._extract_col(self.filtered_train_data, "movie_id")

        user_index_list = [self.user_id_indices[user_id] for user_id in user_id_list]
        movie_index_list = [
            self.movie_id_indices[movie_id] for movie_id in movie_id_list
        ]

        self.interaction_matrix = coo_matrix(
            (interaction_list, (user_index_list, movie_index_list))
        )

    def _get_indices(self, input_data: list[IntegratedData]):
        # Create a user_id to dict and a movie_id to ind  for the predictor to refer to:
        self.user_id_indices, _ = indices_mapper(
            input_data=input_data, id_col_name="user_id", reverse=False
        )
        self.movie_id_indices, self.indices_movie_id = indices_mapper(
            input_data=input_data, id_col_name="movie_id", reverse=True
        )

        # Get the unique user id list
        self.unique_user_ids = sorted(set([row.user_id for row in input_data]))

        # Get the unique movie id list
        self.unique_movie_ids = sorted(set([row.movie_id for row in input_data]))

    def train(self, factors=10, n_epochs=50, alpha=1.0):
        # Prepare the interaction matrix
        self._get_interaction_matrix()

        # Convert the matrix to CSR format
        ratings_csr = self.interaction_matrix.tocsr()

        # Initialize the ALS model
        als_model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=n_epochs,
            calculate_training_loss=True,
            random_state=1,
        )

        # Train the model using the interaction matrix we have created
        als_model.fit(ratings_csr)

        self.algo = als_model
        self.ratings_csr = ratings_csr

    def _predict_single_user_top10(self, user_id: int = 1, N: int = 10) -> list[int]:
        # Execute the recommendation
        user_index = self.user_id_indices[user_id]
        recommendations = self.algo.recommend(
            user_index, self.ratings_csr[user_index], N=N
        )

        # Map the recommendations back to movie IDs
        recommended_movie_ids = [
            self.indices_movie_id[movie_index] for movie_index in recommendations[0]
        ]

        return recommended_movie_ids

    def predict(self):
        # Predict the top 10 items for each user
        pred_user2items = defaultdict(list)
        # _pred_ratings: do not predict the ratings in this algorithm
        pred_ratings = [row.rating for row in self.test_data]

        test_user_ids = sorted(set([row.user_id for row in self.test_data]))

        # for row in self.test_data:
        for test_user_id in test_user_ids:
            pred_user2items[test_user_id] = self._predict_single_user_top10(
                test_user_id
            )
            # pred_ratings.append(0)  # Do not predict this value

        self._pred_user2items = pred_user2items
        self._pred_ratings = pred_ratings

    def evaluate(self) -> Metrics:
        metrics = Metrics.from_ml_10m_data(
            true_ratings=self._true_ratings,
            pred_ratings=self._pred_ratings,
            true_user2items=self._true_user2items,
            pred_user2items=self._pred_user2items,
        )
        return metrics


if __name__ == "__main__":
    import asyncio

    input_data = asyncio.run(IntegratedDatas.from_db(user_num=100))
    # print(input_data.ave_ratings[0:10])
    recommender = IMFRecommender(input_data)
    recommender.preprocess()
    recommender.train()
    print(len(recommender.indices_movie_id))
    # print(recommender.indices_movie_id.values())
    recommender.predict()
    # breakpoint()
    metrics = recommender.evaluate()

    # print(recommender.pred_user2items[8])

    logger.info(
        f"""
        model: IMFRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
