import pandas as pd
from collections import defaultdict
from surprise import SVD, Reader, Dataset, Prediction
from datareader.ml_10m_data import IntegratedDatas
from utils.evaluation_metrics import Metrics
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging

logger = configure_logging()


class MFRecommender(BaseRecommender):
    def __init__(self, input_data: IntegratedDatas):
        super().__init__(input=input_data)

        # Transform the dataset into dataframe
        self._get_df()

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
    def _filter_data(df: pd.DataFrame, minimum_num_rating=100):
        """Filter the data; only select the movies that have at least 100 rating data.
        Purpose:
            Preprocess

        Args:
            data (list[IntegratedData]): _description_
            minimum_num_rating (int, optional): _description_. Defaults to 100.
        """
        df.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )

        return df

    def preprocess(self, minimum_num_rating: int = 100):
        filtered_movielens_train = self._filter_data(
            self.train_df, minimum_num_rating=minimum_num_rating
        )

        # Create dataset for surprise
        reader = Reader(rating_scale=(0.5, 5))
        data_train = Dataset.load_from_df(
            filtered_movielens_train[["user_id", "movie_id", "rating"]], reader
        ).build_full_trainset()

        self.data_train = data_train

    def train(
        self,
        factors: int = 5,
        use_biase=False,
        lr_all=0.005,
        n_epochs=50,
    ):
        # Applying MF to the preprocessed dataset
        matrix_factorization = SVD(
            n_factors=factors, n_epochs=n_epochs, lr_all=lr_all, biased=use_biase
        )
        matrix_factorization.fit(self.data_train)

        # The matrix we would use in the prediction
        self.matrix_factorization = matrix_factorization

    @staticmethod
    def _get_top_n(predictions: list[Prediction], n=10) -> dict[int, list[float]]:
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        # iterate the predictions, sorting the items based on the rating
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = [d[0] for d in user_ratings[:n]]

        return top_n

    @staticmethod
    def _reform_surprise_prediction_df(predictions: list[Prediction]) -> pd.DataFrame:
        """Reform the predictions retrieved from surprise module into a dataframe.

        Args:
            predictions (list[Prediction]): _description_

        Returns:
            pd.DataFrame: _description_

        Example of the return:
                user_id	movie_id	rating_pred
            0	1	    260	        5.000000
            1	1	    733	        5.000000
            2	1	    786	        4.500792
        """

        predictions_list_reformed = [
            {"user_id": p.uid, "movie_id": p.iid, "rating_pred": p.est}
            for p in predictions
        ]
        return pd.DataFrame.from_dict(predictions_list_reformed)

    def predict(self):
        data_test = self.data_train.build_anti_testset(None)
        predictions = self.matrix_factorization.test(data_test)

        # Create pred_ussr2items
        pred_user2items = self._get_top_n(predictions, n=10)
        self._pred_user2items = pred_user2items

        # Create pred_ratings
        pred_df = self._reform_surprise_prediction_df(predictions)
        movie_rating_predict = self.test_df.merge(
            pred_df, on=["user_id", "movie_id"], how="left"
        )

        # Fill the ratings that were not predicted:
        average_score = self.train_df["rating"].mean()
        movie_rating_predict.rating_pred.fillna(average_score, inplace=True)
        pred_results = movie_rating_predict.rating_pred.to_list()

        self._pred_ratings = pred_results

    def evaluate(self) -> Metrics:
        """Evaluate the predicted recommendation result.

        Returns:
            Metrics: Evaluation results of the prediction.
        """
        logger.info("MFRecommender: Calculating metrics")

        metrics = Metrics.from_ml_10m_data(
            true_ratings=self._true_ratings,
            pred_ratings=self._pred_ratings,
            true_user2items=self._true_user2items,
            pred_user2items=self._pred_user2items,
        )
        return metrics
