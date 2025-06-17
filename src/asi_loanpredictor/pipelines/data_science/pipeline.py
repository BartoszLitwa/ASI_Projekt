from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs="loan_train_features",
                outputs=["loan_model", "model_features"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["loan_model", "model_features", "loan_test_features"],
                outputs="model_accuracy",
                name="evaluate_model_node",
            ),
        ]
    )
