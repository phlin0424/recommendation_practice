from collections import defaultdict

import numpy as np
import pandas as pd
from datareader.ml_10m_data import IntegratedData, IntegratedDatas
from surprise import Dataset as SurpriseDataset
from surprise import KNNWithMeans, Reader
from utils.evaluation_metrics import Metrics
from utils.helper import indices_mapper
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging

logger = configure_logging()


class UMCFSurpriseRecommender(BaseRecommender):
    def __init__(self, random_datas: IntegratedDatas):
        super().__init__(input_data=random_datas)

        # Transform the dataset into dataframe
        self._get_df()

        # Create a dataset that will be used in SVD
        reader = Reader(rating_scale=(0.5, 5))
        self.data_train = SurpriseDataset.load_from_df(
            self.train_df[["user_id", "movie_id", "rating"]], reader
        ).build_full_trainset()

        # Create the test data using the opposite of the train data
        self.data_test = self.data_train.build_anti_testset(None)

    def _get_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the dataset into Data frame format

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: _description_
        """
        self.train_df = pd.DataFrame(
            [rating.model_dump() for rating in self.train_data]
        )
        self.test_df = pd.DataFrame(
            [rating.model_dump() for rating in self.test_data],
        )

    @staticmethod
    def _get_top_n(predictions, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        # iterate the predictions, sorting the items based on the rating
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = [d[0] for d in user_ratings[:n]]

        return top_n

    def train(self):
        # Use pearson to calculate the nearest users
        sim_options = {"name": "pearson", "user_based": True}
        knn = KNNWithMeans(k=30, min_k=1, sim_options=sim_options)
        knn.fit(self.data_train)

        self.algo = knn

    def __find_nearest_user(self, user_id: int):
        knn = self.algo
        user_inner_id = knn.trainset.to_inner_uid(user_id)
        neighbors = knn.get_neighbors(user_inner_id, k=10)
        neighbors = [knn.trainset.to_raw_uid(inner_id) for inner_id in neighbors]

    def predict(self):
        logger.info("UMCFRecommender: predicting (surprise)")

        knn = self.algo
        predictions = knn.test(self.data_test)
        self._pred_user2items = self._get_top_n(predictions, n=10)

        average_score = self.train_df["rating"].mean()
        pred_results = []
        for _, row in self.test_df.iterrows():
            user_id = row["user_id"]
            movie_id = row["movie_id"]

            if (
                user_id not in self.data_train._raw2inner_id_users
                or movie_id not in self.data_train._raw2inner_id_items
            ):
                pred_results.append(average_score)
                continue
            pred_score = knn.predict(uid=user_id, iid=movie_id).est
            pred_results.append(pred_score)
        self._pred_ratings = pred_results

    def evaluate(self) -> Metrics:
        """Evaluate the predicted recommendation result.

        Returns:
            Metrics: Evaluation results of the prediction.
        """
        logger.info("UMCFRecommender: Calculating metrics (surprise)")

        metrics = Metrics.from_ml_10m_data(
            true_ratings=self._true_ratings,
            pred_ratings=self._pred_ratings,
            true_user2items=self._true_user2items,
            pred_user2items=self._pred_user2items,
        )
        return metrics


class UMCFRecommender(BaseRecommender):
    def __init__(self, random_datas: IntegratedDatas):
        super().__init__(input_data=random_datas)

    @staticmethod
    def _pearson_coefficient(u: np.ndarray, v: np.ndarray) -> float:
        u_diff = u - np.mean(u)
        v_diff = v - np.mean(v)
        numerator = np.dot(u_diff, v_diff)
        denominator = np.sqrt(sum(u_diff**2)) * np.sqrt(sum(v_diff**2))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _find_common_no_nan(
        arr1: np.ndarray, arr2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Find indices where neither arr1 nor arr2 has NaN values
        valid_indices = ~np.isnan(arr1) & ~np.isnan(arr2)

        # Skip further processing if there are no valid indices
        if not valid_indices.any():
            return None, None  # or return an appropriate placeholder

        # Filter both arrays to remove NaN values
        filtered_arr1 = arr1[valid_indices]
        filtered_arr2 = arr2[valid_indices]

        return filtered_arr1, filtered_arr2

    @staticmethod
    def _get_test_user_rated_movie_ids(
        test_data: list[IntegratedData], test_user_id: int
    ):
        """Return a list of the movie_ids which the specified user has rated.

        Args:
            test_data (list[IntegratedData]): _description_
            test_user_id (int): _description_

        Returns:
            _type_: _description_
        """
        return [row.movie_id for row in test_data if row.user_id == test_user_id]

    def _find_test_users(self):
        unique_test_user_ids = []
        for row in self.test_data:
            unique_test_user_ids.append(row.user_id)

        self.unique_test_user_ids = unique_test_user_ids

    def _create_rating_matrix(self):
        train_rating_matrix = np.full(
            (len(self.unique_user_ids), len(self.unique_movie_ids)), np.nan
        )

        for row in self.train_data:
            train_rating_matrix[
                self.user_id_indices[row.user_id], self.movie_id_indices[row.movie_id]
            ] = row.rating

        self.train_rating_matrix = train_rating_matrix

    def _get_unique_ids(self):
        # Get the unique user id list
        self.unique_user_ids = sorted(set([row.user_id for row in self.train_data]))

        # Get the unique movie id list
        self.unique_movie_ids = sorted(set([row.movie_id for row in self.train_data]))

    def _get_indices(self):
        # Create a user_id to dict and a movie_id to ind  for the predictor to refer to:
        self.user_id_indices, _ = indices_mapper(
            input_data=self.train_data, id_col_name="user_id", reverse=False
        )
        self.movie_id_indices, _ = indices_mapper(
            input_data=self.train_data, id_col_name="movie_id", reverse=False
        )

    def train(self):
        # Create the necessary variables when predicting:
        self._get_unique_ids()
        self._get_indices()
        self._create_rating_matrix()
        self._find_test_users()

    def _predict_naive(self):
        # A list to store the predicted results
        prediction_results = []

        # Iterate all the test users to find the similarities for the target test user to other users
        for user_id1 in self.unique_test_user_ids:
            # A list that store the similar_users (when the estimated rho> 0)
            similar_users = []

            # A list that stores the similarities between the users (when the estimated rho>0 )
            similarities = []

            # A list that stores the averaged ratings of all the rated movies by the similar user (user_id2)
            avgs = []

            # Calculate the averaged ratings of movies rated by the test user:
            avg_1 = np.nanmean(
                self.train_rating_matrix[self.user_id_indices[user_id1], :]
            )

            # Get the movie ids that the test user (usr_id1) has already rated, which are the targets we wanna predict
            test_movies = self._get_test_user_rated_movie_ids(self.test_data, user_id1)

            for user_id2 in self.unique_user_ids:
                if user_id1 == user_id2:
                    # Do not calculate the similarity of the user to their selves
                    continue
                # u_1: all the ratings from the test user
                u_1 = self.train_rating_matrix[self.user_id_indices[user_id1], :]
                u_2 = self.train_rating_matrix[self.user_id_indices[user_id2], :]
                # print(user_id1, user_id2)
                # print(u_1, u_2)
                u_1, u_2 = self._find_common_no_nan(u_1, u_2)
                if u_1 is None and u_2 is None:
                    continue

                rho_12 = self._pearson_coefficient(u_1, u_2)

                if rho_12 > 0:
                    # the similar user list to user_id1 (test_user_id)
                    similar_users.append(user_id2)
                    # The pho value between two two users
                    similarities.append(rho_12)
                    # The averaged rating of the similar user
                    avgs.append(np.mean(u_2))

                # If at least more than 1 similar user got identified, predict the ratings based on the similarities
                if similar_users:
                    # Iterate the movies that we want to predict
                    for movie_id in test_movies:
                        # Confirm that if the movie we wanna predict has corresponding rating in the training data
                        if movie_id in self.unique_movie_ids:
                            # Find all the ratings rated from other similar users:
                            similar_users_indices = [
                                self.user_id_indices[uid] for uid in similar_users
                            ]
                            r_xy = self.train_rating_matrix[
                                similar_users_indices, self.movie_id_indices[movie_id]
                            ]

                            # Do not estimated the predictions when no ratings are found
                            rating_exists = ~np.isnan(r_xy)
                            if not rating_exists.any():
                                continue

                            # Estimate the predicted rating of the target movie,
                            # By using the information of the similar users
                            r_xy = r_xy[rating_exists]
                            rho_1x = np.array(similarities)[rating_exists]
                            avg_x = np.array(avgs)[rating_exists]
                            r_hat_1y = (
                                avg_1 + np.dot(rho_1x, (r_xy - avg_x)) / rho_1x.sum()
                            )

                            prediction_results.append(
                                {
                                    "user_id": user_id1,
                                    "movie_id": movie_id,
                                    "rating": r_hat_1y,
                                }
                            )

            # The prediction result goes here:
            self.prediction_results = prediction_results

            # It takes time when using this algorithm so we don't calculate this metric this time.
            self._pred_user2items = defaultdict(list)

    def _predict_non_naive(self):
        raise RuntimeError("Non-naive prediction is not supported currently.")

    def predict(self, naive=False):
        if naive:
            self._predict_naive()
        else:
            self._predict_non_naive()

        # Re-order the predicted ratings
        pred_ratings_dict = {
            (item["user_id"], item["movie_id"]): item["rating"]
            for item in self.prediction_results
        }

        pred_ratings = []
        for row in self.test_data:
            # Avg: when the rating wasn't predicted
            avg_1 = np.nanmean(
                self.train_rating_matrix[self.user_id_indices[row.user_id], :]
            )
            pred_rating = pred_ratings_dict.get((row.user_id, row.movie_id), avg_1)
            pred_ratings.append(pred_rating)

        self._pred_ratings = pred_ratings

        # self._pred_ratings = [
        #     pred_ratings_dict[(item.user_id, item.movie_id)] for item in self.test_data
        # ]

    def evaluate(self) -> Metrics:
        """Evaluate the predicted recommendation result.

        Returns:
            Metrics: Evaluation results of the prediction.
        """
        logger.info("UMCFRRecommender: Calculating metrics")

        metrics = Metrics.from_ml_10m_data(
            true_ratings=self._true_ratings,
            pred_ratings=self._pred_ratings,
            true_user2items=self._true_user2items,
            pred_user2items=self._pred_user2items,
        )
        return metrics


if __name__ == "__main__":
    import asyncio

    input_data = asyncio.run(IntegratedDatas.from_db(user_num=500))
    # print(input_data.ave_ratings[0:10])
    recommender = UMCFRecommender(input_data)
    recommender.train()
    recommender.predict(naive=True)
    # breakpoint()
    metrics = recommender.evaluate()

    print(recommender.pred_user2items[8])

    logger.info(
        f"""
        model: UMCFRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
