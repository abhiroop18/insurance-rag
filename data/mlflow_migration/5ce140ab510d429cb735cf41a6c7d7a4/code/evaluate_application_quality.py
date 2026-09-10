import json
import time
from datetime import datetime
from pathlib import Path

from deepeval.evaluate import (
    AsyncConfig,
    DisplayConfig,
    evaluate,
)
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCaseParams

from insurance_rag.utils.config import load_config
from insurance_rag.workflow.rag_workflow import create_rag_graph


# =============================================================================
# PROJECT CONFIGURATION
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
    / "application_quality"
)


# =============================================================================
# LOAD GOLDEN DATASET
# =============================================================================

def load_golden_dataset(
    dataset_path: Path,
):
    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found:\n"
            f"{dataset_path}"
        )

    with open(
        dataset_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =============================================================================
# CREATE RAG WORKFLOW
# =============================================================================

def create_application():

    return create_rag_graph()


# =============================================================================
# RUN APPLICATION
# =============================================================================

def generate_application_answer(
    application,
    question: str,
):

    result = application.invoke(
        {
            "question": question,
            "retrieved_context": [],
            "answer": "",
        }
    )

    if "answer" not in result:

        raise ValueError(
            "RAG workflow did not return 'answer'."
        )

    answer = result["answer"]

    if not isinstance(answer, str):

        raise TypeError(
            "RAG workflow returned a non-string answer."
        )

    return answer


# =============================================================================
# BUILD DEEPEVAL TEST CASES
# =============================================================================

def build_test_cases(
    golden_dataset,
    application,
):

    test_cases = []

    for item in golden_dataset:

        question = item["input"]

        expected_answer = (
            item["expected_output"]
        )

        actual_answer = (
            generate_application_answer(
                application=application,
                question=question,
            )
        )

        # Keep this because your current setup
        # requires throttling between requests.
        time.sleep(7)

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
        )

        test_cases.append(
            test_case
        )

    return test_cases


# =============================================================================
# CREATE QUALITY METRICS
# =============================================================================

def create_quality_metrics(
    config,
):

    eval_model = OpenAIModel(
        model=config.evaluation.model,
        temperature=0,
    )

    # -------------------------------------------------------------------------
    # CORRECTNESS
    # -------------------------------------------------------------------------

    correctness = GEval(
        name="Application Correctness",

        criteria=(
            "Evaluate whether the actual answer correctly answers "
            "the user's question and is consistent with the expected answer. "
            "The answer must not contain incorrect insurance policy claims "
            "or contradict the expected answer."
        ),

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],

        model=eval_model,

        threshold=config.evaluation.threshold,
    )

    # -------------------------------------------------------------------------
    # COMPLETENESS
    # -------------------------------------------------------------------------

    completeness = GEval(
        name="Application Completeness",

        criteria=(
            "Evaluate whether the actual answer contains all important "
            "information needed to answer the user's question. "
            "Compare it with the expected answer and identify whether "
            "important conditions, limits, time periods, exceptions, "
            "or other relevant details have been omitted. "
            "Do not require the actual answer to use the same wording "
            "as the expected answer."
        ),

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],

        model=eval_model,

        threshold=config.evaluation.threshold,
    )

    # -------------------------------------------------------------------------
    # STYLE
    # -------------------------------------------------------------------------

    style = GEval(
        name="Application Style",

        criteria=(
            "Evaluate whether the actual answer is clear, simple, concise, "
            "and professional for an insurance policy assistant. "
            "The answer should use simple language, avoid unnecessary jargon, "
            "avoid unnecessary repetition, and directly address the user's "
            "question. Do not penalize the answer for using different wording "
            "from the expected answer."
        ),

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],

        model=eval_model,

        threshold=config.evaluation.threshold,
    )

    return [
        correctness,
        completeness,
        style,
    ]


# =============================================================================
# RUN APPLICATION QUALITY EVALUATION
# =============================================================================

def evaluate_application_quality():

    config = load_config()

    # -------------------------------------------------------------------------
    # Golden dataset
    # -------------------------------------------------------------------------

    dataset_path = (
        PROJECT_ROOT
        / config.dataset.path
    )

    golden_dataset = load_golden_dataset(
        dataset_path=dataset_path
    )

    # -------------------------------------------------------------------------
    # Create application
    # -------------------------------------------------------------------------

    application = create_application()

    # -------------------------------------------------------------------------
    # Build DeepEval test cases
    # -------------------------------------------------------------------------

    test_cases = build_test_cases(
        golden_dataset=golden_dataset,
        application=application,
    )

    # -------------------------------------------------------------------------
    # Create metrics
    # -------------------------------------------------------------------------

    metrics = create_quality_metrics(
        config=config
    )

    # -------------------------------------------------------------------------
    # Create dated directories
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Run DeepEval
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Return evaluation information
    # -------------------------------------------------------------------------

    return {
        "evaluation_results":
            evaluation_results,

        "results_dir":
            results_dir,

        "report_dir":
            report_dir,

        "timestamp":
            timestamp,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    evaluate_application_quality()