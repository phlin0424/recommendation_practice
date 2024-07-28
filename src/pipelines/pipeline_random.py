from datareader.ml_10m_data import IntegratedDatas
from utils.models import RandomRecommender
from utils.evaluation_metrics import Metrics
from utils.pipeline_logging import configure_logging
import asyncio

logger = configure_logging()


def preprocess(user_num) -> IntegratedDatas:
    integrated_datas = asyncio.run(IntegratedDatas.from_db(user_num=user_num))
    return integrated_datas


def train_model(integrated_datas: IntegratedDatas) -> RandomRecommender:
    random_recommender = RandomRecommender(integrated_datas)
    random_recommender.train()
    return random_recommender


def evaluate_model(random_recommender: RandomRecommender) -> Metrics:
    random_recommender.predict()
    metrics = random_recommender.evaluate()
    return metrics


if __name__ == "__main__":
    input_data = preprocess(user_num=1000)
    recommender = train_model(input_data)
    metrics = evaluate_model(recommender)
    logger.info(metrics)
