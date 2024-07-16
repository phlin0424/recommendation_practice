from pipelines.pipeline_svd.preprocess import preprocess
from pipelines.pipeline_svd.train import train_model
from pipelines.pipeline_svd.evaluate import evaluation_model


def test_reprocess():
    test_size = 0.25
    trainset, testset = preprocess(test_size)

    assert trainset is not None
