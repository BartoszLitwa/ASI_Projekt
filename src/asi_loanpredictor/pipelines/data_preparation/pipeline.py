from kedro.pipeline import Pipeline, node, pipeline
from . import nodes
from .nodes import split_train_test, clean_loan_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
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