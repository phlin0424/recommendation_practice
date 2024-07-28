import os
import sys
import pipelines.pipeline_popularity
import pipelines.pipeline_random
import pipelines.pipeline_svd


def run_pipeline(pipeline_name):
    if pipeline_name == "RandomRecommender":
        pipelines.pipeline_random.run_pipeline()
    if pipeline_name == "PopularityRecommender":
        pipelines.pipeline_popularity.run_pipeline()
    elif pipeline_name == "SVD":
        pipelines.pipeline_svd.run_pipeline()
    else:
        print(f"Unknown pipeline: {pipeline_name}")
        sys.exit(1)


if __name__ == "__main__":
    pipeline_name = os.getenv("PIPELINE_NAME", "RandomRecommender")
    run_pipeline(pipeline_name)
