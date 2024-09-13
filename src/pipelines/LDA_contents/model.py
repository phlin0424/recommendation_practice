from collections import defaultdict, Counter

import gensim
from datareader.ml_10m_data import (
    IntegratedTagsData,
    IntegratedTagsDatas,
)
from utils.evaluation_metrics import Metrics
from utils.helper import indices_mapper
from utils.models import BaseRecommender
from utils.pipeline_logging import configure_logging
from gensim.corpora.dictionary import Dictionary


class LDAContentsRecommender(BaseRecommender):
    def __init__(self, input_data: IntegratedTagsDatas):
        super().__init__(input_data)

        # Load the tag&genres data
        self.movie_contents = input_data.movie_contents

    @staticmethod
    def get_tags_genres(movie_contents: list[IntegratedTagsData]) -> list[list[str]]:
        """Return a list of list which contains the tag& genres of a movie

        Args:
            movie_contents (list[IntegratedTagsData]): _description_

        Returns:
            list[list[str]]: _description_
        """
        genres_tags = []
        for item in movie_contents:
            if item.tags != [""]:
                contents = item.genres + item.tags
            else:
                contents = item.genres  # no null genres
            genres_tags.append(contents)

        return genres_tags

    def preprocess(self):
        # Get a list of the tags + genres tags:
        # (dont split the data when doing this process)
        tag_genre_data = self.get_tags_genres(self.movie_contents)

        # Creates a mapping from each unique word (tag or genre) in the tag_genre_data to a unique integer ID.
        common_dictionary = Dictionary(tag_genre_data)
        self.common_dictionary = common_dictionary

        # convert tag&genres of each movie into bag-of-words (bow) representations
        # where each word (tag or genre) is represented by its integer ID from the dictionary,
        # and the frequency of each word is recorded.
        common_corpus = [common_dictionary.doc2bow(text) for text in tag_genre_data]
        self.common_corpus = common_corpus

    def train(self, factors=50, n_epochs=50):
        #  Training: to identify {factors=50} topics based on the tags and genres of the movies.
        lda_model = gensim.models.LdaModel(
            self.common_corpus,
            id2word=self.common_dictionary,
            num_topics=factors,
            passes=n_epochs,
        )
        self.algo = lda_model

    def extract_topic_scores(self) -> None:
        # Use the trained LDA model to get the topic distribution for each movie
        lda_topics = self.algo[self.common_corpus]

        # Store the topic distribution result
        movie_topics = []
        movie_topic_scores = []
        for movie_index, lda_topic in enumerate(lda_topics):
            # Sort the topics for the current movie by their probability in descending order
            sorted_topic = sorted(lda_topics[movie_index], key=lambda x: -x[1])
            movie_topic, topic_score = sorted_topic[0]
            movie_topics.append(movie_topic)
            movie_topic_scores.append(topic_score)

        # Store the topic distributions
        self.movie_topics = movie_topics
        self.movie_topic_scores = movie_topic_scores

    def _get_indices(self, input_data: list[IntegratedTagsData]):
        # Create a user_id to dict and a movie_id to ind  for the predictor to refer to:
        movie_id_indices, _ = indices_mapper(
            input_data=input_data, id_col_name="movie_id", reverse=True
        )
        return movie_id_indices

    def _get_watched_list(self) -> dict[int, list[int]]:
        # Find the most recently watched movie for each user
        grouped_watched_list = defaultdict(list)
        for row in self.train_data:
            grouped_watched_list[row.user_id].append((row.movie_id, row.timestamp))
        return grouped_watched_list

    def predict(self):
        # extract topic scores and topics based on the trained model
        self.extract_topic_scores()

        # Find the most recently watched movie for each user from the training data
        grouped_watched_list = self._get_watched_list()

        # Get the movie_id - index pair
        movie_id_indices = self._get_indices(self.movie_contents)

        # Initialize a dictionary to store predictions
        pred_user2items = defaultdict(list)

        # Get a list that contains all the users
        unique_user_ids = sorted(set([row.user_id for row in self.test_data]))

        # Looping over all the users to recommend the movies based on topics
        for user_id in unique_user_ids:
            # Sort the watching list of each user to get the most recently watched movie
            movies_watched = grouped_watched_list[user_id]
            sorted_movies = sorted(movies_watched, key=lambda x: x[1], reverse=True)
            recently_watched_movie_ids = [
                movie_id for movie_id, timestamp in sorted_movies
            ]

            # Derive the last 10 movies each user has watched:
            movie_ids = recently_watched_movie_ids[-10:]

            # Get the most frequent topic from the recently watched movies:
            movie_indexes = [movie_id_indices[id] for id in movie_ids]
            topic_counter = Counter([self.movie_topics[i] for i in movie_indexes])
            frequent_topic = topic_counter.most_common(1)[0][0]

            # Regarding this user, find other movies that with the same topic (frequent_topic)
            recommend_candidates = []
            for i, movie_topic in enumerate(self.movie_topics):
                if movie_topic == frequent_topic:
                    recommend_candidates.append(
                        (self.movie_contents[i].movie_id, self.movie_topic_scores[i])
                    )
            recommend_candidates = sorted(
                recommend_candidates, key=lambda x: x[1], reverse=True
            )

            # Recommend movies to the user according to the topic score
            for movie_id in recommend_candidates:
                if movie_id not in movies_watched:
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
    recommender = LDAContentsRecommender(input_data)
    recommender.preprocess()
    recommender.train()
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
