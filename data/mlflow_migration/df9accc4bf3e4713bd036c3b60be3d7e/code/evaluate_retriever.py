import json
import os
import time
from datetime import datetime
from pathlib import Path

from deepeval.evaluate import (
    AsyncConfig,
    DisplayConfig,
    evaluate,
)
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

from insurance_rag.retrieval.retriever import InsuranceRetriever
from insurance_rag.utils.config import load_config


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reports"
    / "retriever"
)


# =============================================================================
# LOAD GOLDEN DATASET
# =============================================================================

def load_golden_dataset(
    dataset_path: Path,
):
    with open(
        dataset_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =============================================================================
# CREATE RETRIEVER
# =============================================================================

def create_retriever():

    cohere_api_key = os.getenv(
        "COHERE_API_KEY"
    )

    if not cohere_api_key:

        raise ValueError(
            "COHERE_API_KEY environment variable is not set."
        )

    return InsuranceRetriever(
        cohere_api_key=cohere_api_key
    )


# =============================================================================
# RETRIEVE CONTEXT
# =============================================================================

def retrieve_context(
    retriever: InsuranceRetriever,
    question: str,
):

    results = retriever.retrieve(
        query=question,
    )

    contexts = [
        result["page_content"]
        for result in results
    ]

    return contexts


# =============================================================================
# BUILD DEEPEVAL TEST CASES
# =============================================================================

def build_test_cases(
    golden_dataset,
    retriever,
):

    test_cases = []

    for item in golden_dataset:

        question = item["input"]

        expected_answer = (
            item["expected_output"]
        )

        contexts = retrieve_context(
            retriever=retriever,
            question=question,
        )

        time.sleep(7)

        test_case = LLMTestCase(
            input=question,
            expected_output=expected_answer,
            retrieval_context=contexts,
        )

        test_cases.append(
            test_case
        )

    return test_cases


# =============================================================================
# RUN RETRIEVER EVALUATION
# =============================================================================

def evaluate_retriever():

    config = load_config()

    dataset_path = (
        PROJECT_ROOT
        / config.dataset.path
    )

    golden_dataset = load_golden_dataset(
        dataset_path
    )

    retriever = create_retriever()

    test_cases = build_test_cases(
        golden_dataset=golden_dataset,
        retriever=retriever,
    )

    eval_model = OpenAIModel(
        model=config.evaluation.model,
        temperature=0,
    )

    contextual_recall = (
        ContextualRecallMetric(
            threshold=config.evaluation.threshold,
            model=eval_model,
            include_reason=True,
        )
    )

    contextual_precision = (
        ContextualPrecisionMetric(
            threshold=config.evaluation.threshold,
            model=eval_model,
            include_reason=True,
        )
    )

    contextual_relevancy = (
        ContextualRelevancyMetric(
            threshold=config.evaluation.threshold,
            model=eval_model,
            include_reason=True,
        )
    )

    metrics = [
        contextual_recall,
        contextual_precision,
        contextual_relevancy,
    ]

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    results_dir = (
        REPORT_ROOT
        / "results"
        / timestamp
    )

    report_dir = (
        REPORT_ROOT
        / "reports"
        / timestamp
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            max_concurrent=
                config.deepeval.max_concurrent,

            throttle_value=
                config.deepeval.throttle_value,

            run_async=
                config.deepeval.run_async,
        ),
        display_config=DisplayConfig(
            results_folder=str(
                results_dir
            ),
            file_type="md",
            file_output_dir=str(
                report_dir
            ),
        ),
    )

    return {
        "evaluation_results": evaluation_results,
        "results_dir": results_dir,
        "report_dir": report_dir,
        "timestamp": timestamp,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    evaluate_retriever()