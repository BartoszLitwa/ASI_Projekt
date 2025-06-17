from kedro.pipeline import Pipeline, node, pipeline
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(nodes.clean_data, inputs="loan_data", outputs="cleaned_data", name="clean_data"),
        node(nodes.remove_outliers, inputs="cleaned_data", outputs="no_outliers_data", name="remove_outliers"),
        node(nodes.handle_missing_values, inputs="no_outliers_data", outputs="no_missing_data", name="handle_missing"),
        node(nodes.engineer_features, inputs="no_missing_data", outputs="featured_data", name="engineer_features"),
        node(nodes.balance_dataset, inputs="featured_data", outputs="balanced_data", name="balance_dataset"),
        node(nodes.split_data, inputs="balanced_data", outputs=["train_data", "test_data"], name="split_data"),
    ])