from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs="train_features",
            outputs=["model", "model_features", "optimal_threshold"],
            name="train_model",
        ),
        node(
            func=evaluate_model,
            inputs=["model", "test_features", "optimal_threshold"],
            outputs="model_metrics",
            name="evaluate_model",
        ),
    ])
