from utils.models import BaseRecommender
from datareader.ml_10m_data import (
    PopularityDatas,
    IntegratedData,
    PopularityAveRating,
)
from utils.evaluation_metrics import Metrics
from collections import defaultdict
from utils.pipeline_logging import configure_logging


# Set up log
logger = configure_logging()


class PopularityRecommender(BaseRecommender):
    def __init__(self, popularity_data: PopularityDatas):
        # Load the necessary data from the input model base
        # Load the average rating of all the movies
        self.ave_ratings = popularity_data.ave_ratings

        super().__init__(input_data=popularity_data)

    def train(self):
        pass

    def _get_watched_list(self) -> dict[int, list[int]]:
        # Get the watched list for each user
        user_watched_movies = defaultdict(list)
        for row in self.train_data:
            user_watched_movies[row.user_id].append(row.movie_id)
        self.user_watched_movies = user_watched_movies

    @staticmethod
    def _predict_user2items(
        test_data: list[IntegratedData],
        ave_ratings: list[PopularityAveRating],
        user_watched_movies: dict[int, list[int]],
        threshold: int = 10,
    ) -> dict[int, list[float]]:
        # Drive the sorted movie_id list, which is sorted by rating
        movie_ids_sorted_by_ave = [
            item.movie_id for item in ave_ratings if item.rated_movies_count > threshold
        ]

        # Get the user_id list
        unique_user_ids = sorted(set([row.user_id for row in test_data]), reverse=True)

        # Predict the recommended movies:
        pred_user2items = defaultdict(list)
        for user_id in unique_user_ids:
            for movie_id in movie_ids_sorted_by_ave:
                if movie_id not in user_watched_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        return pred_user2items

    @staticmethod
    def _predict_ratings(
        test_data: list[IntegratedData], ave_ratings: list[PopularityAveRating]
    ) -> list[float]:
        movie_id_to_ave_rating = {
            item.movie_id: item.ave_rating for item in ave_ratings
        }
        pred_ratings = []
        for row in test_data:
            pred_ratings.append(movie_id_to_ave_rating.get(row.movie_id, 0))
        return pred_ratings

    def predict(self, threshold: int = 10):
        logger.info("PopularityRecommender: Predicting...")
        ave_ratings = self.ave_ratings
        test_data = self.test_data

        # Prediction-1: predict item recommendations for a specific user
        self._get_watched_list()
        pred_user2items = self._predict_user2items(
            test_data, ave_ratings, self.user_watched_movies, threshold
        )

        # Prediction-2: predicted all the ratings
        predict_ratings = self._predict_ratings(test_data, ave_ratings)

        # Save the results
        self._pred_ratings = predict_ratings
        self._pred_user2items = pred_user2items

    def evaluate(self) -> Metrics:
        logger.info("PopularityRecommender: Calculating metrics")
        metrics = Metrics.from_ml_10m_data(
            true_ratings=self.true_ratings,
            pred_ratings=self.pred_ratings,
            true_user2items=self.true_user2items,
            pred_user2items=self.pred_user2items,
        )
        return metrics


if __name__ == "__main__":
    import asyncio

    input_data = asyncio.run(PopularityDatas.from_db())
    # print(input_data.ave_ratings[0:10])
    recommender = PopularityRecommender(input_data)
    recommender.train()
    recommender.predict(threshold=100)
    metrics = recommender.evaluate()

    print(recommender.pred_user2items[8])

    logger.info(
        f"""
        model: PopularityRecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
