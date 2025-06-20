from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_engineering,
            inputs=["train_cleaned", "test_cleaned"], 
            outputs=["train_features", "test_features"],
            name="feature_engineering",
        ),
    ])
