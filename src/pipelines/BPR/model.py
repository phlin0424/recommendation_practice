from datareader.ml_10m_data import IntegratedData, IntegratedDatas
from utils.evaluation_metrics import Metrics
from utils.helper import indices_mapper
from utils.models import BaseRecommender
from collections import defaultdict
from scipy.sparse import coo_matrix
import implicit
from utils.pipeline_logging import configure_logging


class BPRRecommender(BaseRecommender):
    def __init__(self, input_data: IntegratedDatas):
        super().__init__(input_data)

    @staticmethod
    def _filter_data(
        train_data: list[IntegratedData],
        minimum_num_rating=1,
    ) -> list[IntegratedData]:
        group_by_movies = defaultdict(list)
        for item in train_data:
            group_by_movies[item.movie_id].append(item)

        filtered_data = [
            item
            for items in group_by_movies.values()
            if len(items) >= minimum_num_rating
            for item in items
        ]

        return filtered_data

    def preprocess(
        self,
        minimum_num_rating=1,
    ):
        """暗黙の評価値を出すためとある閾値以下の評価値を０にする

        Args:
            minimum_num_rating (int, optional): 興味があると判定する映画あたり評価件数の閾値. Defaults to 2.
        """
        filtered_train_data = self._filter_data(
            self.train_data,
            minimum_num_rating=minimum_num_rating,
        )
        self.filtered_train_data = filtered_train_data

    def _get_indices(self, input_data: list[IntegratedData]):
        """index辞書を作成

        Args:
            input_data (list[IntegratedData]): _description_
        """
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

    @staticmethod
    def _extract_col(
        input_data: list[IntegratedData], col_name: str
    ) -> list[int | float]:
        """指定した列のみを抽出し、新たなリストにする

        Args:
            input_data (list[IntegratedData]): _description_
            col_name (str): _description_

        Returns:
            list[int | float]: _description_
        """
        return [item.get(col_name) for item in input_data]

    def _get_interaction_matrix(self):
        """create an interaction matrix"""
        self._get_indices(input_data=self.filtered_train_data)

        # User id list
        user_id_list = self._extract_col(self.filtered_train_data, "user_id")
        # Movie id list
        movie_id_list = self._extract_col(self.filtered_train_data, "movie_id")

        user_index_list = [self.user_id_indices[user_id] for user_id in user_id_list]
        movie_index_list = [
            self.movie_id_indices[movie_id] for movie_id in movie_id_list
        ]

        self.interaction_matrix = coo_matrix(
            ([1] * len(self.filtered_train_data), (user_index_list, movie_index_list))
        )

    def train(self, factors=10, n_epochs=50):
        # Prepare the interaction matrix
        self._get_interaction_matrix()

        # Convert the interaction matrix to CSR format
        interaction_csr = self.interaction_matrix.tocsr()

        # Initialize the ALS model
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=factors, iterations=n_epochs
        )

        # train the model
        model.fit(interaction_csr)

        # Save the model into instance model
        self.algo = model
        self.interaction_csr = interaction_csr

    def _predict_single_user_top10(self, user_id: int = 1, N: int = 10) -> list[int]:
        # Execute the recommendation
        user_index = self.user_id_indices[user_id]
        recommendations = self.algo.recommend(
            user_index, self.interaction_csr[user_index], N=N
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
    recommender = BPRRecommender(input_data)
    recommender.preprocess()
    recommender.train()
    print(len(recommender.indices_movie_id))
    # print(recommender.indices_movie_id.values())
    recommender.predict()
    # breakpoint()
    metrics = recommender.evaluate()

    logger.info(
        f"""
        model: BPRRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
