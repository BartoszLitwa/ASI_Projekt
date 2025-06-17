from kedro.pipeline import Pipeline, node, pipeline
from . import nodes
from .nodes import split_train_test, clean_loan_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(nodes.clean_data, inputs="loan_data", outputs="cleaned_data", name="clean_data"),
        node(nodes.remove_outliers, inputs="cleaned_data", outputs="no_outliers_data", name="remove_outliers"),
        node(nodes.handle_missing_values, inputs="no_outliers_data", outputs="no_missing_data", name="handle_missing"),
        node(nodes.engineer_features, inputs="no_missing_data", outputs="featured_data", name="engineer_features"),
        node(nodes.balance_dataset, inputs="featured_data", outputs="balanced_data", name="balance_dataset"),
        node(nodes.split_data, inputs="balanced_data", outputs=["train_data", "test_data"], name="split_data"),
        node(
            func=split_train_test,
            inputs={
                "data": "loan_data",
                "test_size": "params:split_train_test.test_size",
                "random_state": "params:split_train_test.random_state"
            },
            outputs=["loan_train", "loan_test"],
            name="split_train_test_node",
        ),
        node(
            func=clean_loan_data,
            inputs=["loan_train", "loan_test"],
            outputs=["loan_train_cleaned", "loan_test_cleaned"],
            name="clean_loan_data_node",
        ),
    ])