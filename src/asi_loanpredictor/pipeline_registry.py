"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from asi_loanpredictor.pipelines.data_preparation import pipeline as data_preparation_pipeline
from asi_loanpredictor.pipelines.data_processing import pipeline as data_processing_pipeline
from asi_loanpredictor.pipelines.data_science import pipeline as data_science_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = data_preparation_pipeline.create_pipeline() + data_processing_pipeline.create_pipeline() + data_science_pipeline.create_pipeline()
    pipelines["data_preparation"] = data_preparation_pipeline.create_pipeline()
    pipelines["data_processing"] = data_processing_pipeline.create_pipeline()
    pipelines["data_science"] = data_science_pipeline.create_pipeline()
    return pipelines
