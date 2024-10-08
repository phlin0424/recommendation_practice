from collections import defaultdict

import implicit
from datareader.ml_10m_data import IntegratedData, IntegratedDatas
from scipy.sparse import coo_matrix
from utils.evaluation_metrics import Metrics
from utils.helper import indices_mapper
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging


class IMFRecommender(BaseRecommender):
    def __init__(self, input_data: IntegratedDatas):
        super().__init__(input_data)

    @staticmethod
    def _filter_data(
        train_data: list[IntegratedData],
        minimum_rating=1,
        minimum_num_rating=2,
    ) -> list[IntegratedData]:
        """filter the data by the following two criteria:
            (1): Only select the movie id that have been rated more than once
                --> 一回でも映画を見たら、ユーザーは映画に対し興味を示している　とう暗黙のルール
            (2): Only select the data with ratings higher than the threshold
                --> 映画へ対し、一点以上評価すると、ユーザーは映画に対し興味を示している　とう暗黙のルール

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
            if len(items) >= minimum_num_rating
            for item in items
        ]

        movielens_train_high_rating = [
            item for item in filtered_data if item.rating >= minimum_rating
        ]

        return movielens_train_high_rating

    def preprocess(
        self,
        minimum_rating=1,
        minimum_num_rating=2,
    ):
        """暗黙の評価値を出すためとある閾値以下の評価値を０にする

        Args:
            minimum_rating (int, optional): 興味があると判定する評価点数の閾値. Defaults to 1.
            minimum_num_rating (int, optional): 興味があると判定する映画あたり評価件数の閾値. Defaults to 2.
        """
        filtered_train_data = self._filter_data(
            self.train_data,
            minimum_num_rating=minimum_num_rating,
            minimum_rating=minimum_rating,
        )
        self.filtered_train_data = filtered_train_data

    @staticmethod
    def _extract_col(
        input_data: list[IntegratedData], col_name: str
    ) -> list[int | float]:
        return [item.get(col_name) for item in input_data]

    def _get_confidence_matrix(self, alpha=1.0):
        """create an interaction matrix"""
        self._get_indices(input_data=self.filtered_train_data)

        interaction_list = self._extract_col(self.filtered_train_data, "rating")
        # Apply the transformation C_ui = 1 + alpha * r_ui
        # 評価点数を基づいて信頼度を計算
        confidence_list = [item * alpha + 1 for item in interaction_list]

        user_id_list = self._extract_col(self.filtered_train_data, "user_id")
        movie_id_list = self._extract_col(self.filtered_train_data, "movie_id")

        user_index_list = [self.user_id_indices[user_id] for user_id in user_id_list]
        movie_index_list = [
            self.movie_id_indices[movie_id] for movie_id in movie_id_list
        ]

        self.confidence_matrix = coo_matrix(
            (confidence_list, (user_index_list, movie_index_list))
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
        """信頼度を計算し、学習を行う

        Args:
            factors (int, optional): factors. モデルのパラメーター。Defaults to 10.
            n_epochs (int, optional): epochs数. モデルのパラメーター。Defaults to 50.
            alpha (float, optional): 信頼度を計算する際に使用する重み. Defaults to 1.0.
        """
        # Prepare the interaction matrix
        self._get_confidence_matrix(alpha=alpha)

        # Convert the confidence matrix to CSR format
        ratings_csr = self.confidence_matrix.tocsr()

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

    logger = configure_logging()

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
