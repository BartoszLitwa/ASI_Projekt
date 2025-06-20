from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["loan_train_features", "parameters"],
                outputs=["loan_model", "model_features", "feature_importance", "cv_scores"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["loan_model", "model_features", "loan_test_features", "parameters"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
