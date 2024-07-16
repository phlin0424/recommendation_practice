from pipelines.pipeline_svd.preprocess import preprocess


def test_reprocess():
    test_size = 0.25
    trainset, testset = preprocess(test_size)

    assert trainset is not None
