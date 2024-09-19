from collections import defaultdict

import gensim
from datareader.ml_10m_data import IntegratedTagsDatas
from gensim.corpora.dictionary import Dictionary
from utils.evaluation_metrics import Metrics
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging


class LDACollaborativeRecommender(BaseRecommender):
    def __init__(self, input_data: IntegratedTagsDatas):
        super().__init__(input_data)

        # Load the tag&genres data
        # The data length of this data would be the number of movie list
        self.movie_contents = input_data.movie_contents

    def get_movies_by_user(self):
        # Find the movie collections per user
        movies_by_user = defaultdict(list)

        # Initialize a list which will be used in prediction
        hight_rating_train_list = []

        for row in self.train_data:
            # Treat rating>4 as highly rated items
            if row.rating >= 4:
                # Need to convert the movie id into str
                movies_by_user[row.user_id].append(str(row.movie_id))
                hight_rating_train_list.append(row)

        self.movies_by_user = movies_by_user
        self.hight_rating_train_list = hight_rating_train_list

    def preprocess(self):
        # Find the movie collections per user, only counting hight ratings
        self.get_movies_by_user()

        # Prepare data for creating lda dictionary
        lda_data = list(self.movies_by_user.values())

        # Creates a Gensim dictionary from lda_data,
        # mapping each unique movie ID to an integer.
        common_dictionary = Dictionary(lda_data)
        self.common_dictionary = common_dictionary

        # Converts each user’s list of movies into a Bag-of-Words (BoW) format using the dictionary.
        # This represents each user’s movie interactions in a sparse BoW format,
        # where each movie is represented by its unique integer ID and count.
        common_corpus = [common_dictionary.doc2bow(text) for text in lda_data]
        self.common_corpus = common_corpus

    def train(self, factors=50, n_epochs=50):
        lda_model = gensim.models.LdaModel(
            self.common_corpus,
            id2word=self.common_dictionary,
            num_topics=factors,
            passes=n_epochs,
        )

        self.algo = lda_model

    def _get_watched_list(self) -> dict[int, list[int]]:
        # Get the list of movies that each user has evaluated:
        grouped_watched_list = defaultdict(list)
        for row in self.train_data:
            grouped_watched_list[row.user_id].append(row.movie_id)
        return grouped_watched_list

    def predict(self):
        # Use the trained LDA model to get the topic (movie) distribution for each user
        lda_topics = self.algo[self.common_corpus]

        # Get the list of movies that each user has evaluated:
        grouped_watched_list = self._get_watched_list()

        # Initialize a dictionary to store predictions
        pred_user2items = defaultdict(list)

        # Get a list that contains all the users that gives high ratings > 4
        unique_user_ids = sorted(
            set([row.user_id for row in self.hight_rating_train_list])
        )

        # Looping the unique user_ids:
        for i, user_id in enumerate(unique_user_ids):
            # Sort the topics by their scores i.e., to find the dominant topics for a specific user
            user_topic = sorted(lda_topics[i], key=lambda x: -x[1])[0][0]

            # Get the movies related to the dominant topics
            topic_movies = self.algo.get_topic_terms(
                user_topic, topn=len(self.movie_contents)
            )

            # Get the movie ids that the user have watched:
            watched_movie_ids = grouped_watched_list[user_id]

            # Recommend the user with the dominant
            for token_id, score in topic_movies:
                movie_id = int(self.common_dictionary.id2token[token_id])
                if movie_id not in watched_movie_ids:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        self._pred_user2items = pred_user2items
        self._pred_ratings = [0] * len(self.test_data)  # dont predict this value

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

    # Load the necessary data
    input_data = asyncio.run(IntegratedTagsDatas.from_db(user_num=100))

    # print(input_data.ave_ratings[0:10])
    recommender = LDACollaborativeRecommender(input_data)
    recommender.preprocess()
    recommender.train()
    recommender.predict()
    # breakpoint()
    metrics = recommender.evaluate()

    # print(recommender.pred_user2items[8])

    logger.info(
        f"""
        model: LDARecommender
        rmse: {metrics.rmse},
        recall_at_k: {metrics.recall_at_k},
        precision_at_k:{metrics.precision_at_k}
        """
    )
