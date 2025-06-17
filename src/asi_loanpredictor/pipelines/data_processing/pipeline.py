from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    load_shuttles_to_csv,
    preprocess_companies,
    preprocess_reviews,
    preprocess_shuttles,
    feature_engineering,
)


def create_pipeline(**kwargs) -> Pipeline:
    data_processing_pipeline = Pipeline([
        node(
            func=feature_engineering,
            inputs=["loan_train_cleaned", "loan_test_cleaned"],
            outputs=["loan_train_features", "loan_test_features"],
            name="feature_engineering_node",
        ),
    ])
    return data_processing_pipeline
