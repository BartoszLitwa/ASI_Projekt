from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_test, clean_loan_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train_test,
            inputs="loan_data",
            outputs=["train_data", "test_data"],
            name="split_train_test",
        ),
        node(
            func=clean_loan_data,
            inputs=["train_data", "test_data"],
            outputs=["train_cleaned", "test_cleaned"],
            name="clean_loan_data",
        ),
    ])