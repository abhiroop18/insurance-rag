import json
import time
from datetime import datetime
from pathlib import Path

from deepeval.evaluate import (
    AsyncConfig,
    DisplayConfig,
    evaluate,
)

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)

from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

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
    / "pipeline"
)


# =============================================================================
# LOAD GOLDEN DATASET
# =============================================================================

def load_golden_dataset(
    dataset_path: Path,
):
    """
    Load the golden evaluation dataset.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found:\n{dataset_path}"
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

def create_pipeline():

    return create_rag_graph()


# =============================================================================
# RUN RAG PIPELINE
# =============================================================================

def run_pipeline(
    pipeline,
    question: str,
):
    """
    Run one question through the complete LangGraph RAG pipeline.

    Expected workflow output:

        {
            "question": ...,
            "retrieved_context": [...],
            "answer": ...
        }
    """

    result = pipeline.invoke(
        {
            "question": question,
        }
    )

    return result


# =============================================================================
# EXTRACT PIPELINE OUTPUT
# =============================================================================

def extract_answer(
    result,
):
    """
    Extract final answer from the LangGraph state.
    """

    answer = result.get("answer")

    if answer is None:
        raise ValueError(
            "RAG workflow did not return 'answer'."
        )

    return answer


def extract_context(
    result,
):
    """
    Extract retrieved context from the LangGraph state.
    """

    contexts = result.get(
        "retrieved_context"
    )

    if contexts is None:
        raise ValueError(
            "RAG workflow did not return "
            "'retrieved_context'."
        )

    if not isinstance(contexts, list):
        raise TypeError(
            "'retrieved_context' must be a list."
        )

    # Make sure DeepEval receives strings.
    contexts = [
        str(context)
        for context in contexts
    ]

    return contexts


# =============================================================================
# BUILD DEEPEVAL TEST CASES
# =============================================================================

def build_test_cases(
    golden_dataset,
    pipeline,
):
    """
    Run the complete RAG pipeline for every golden question
    and convert the outputs into DeepEval test cases.
    """

    test_cases = []

    for item in golden_dataset:

        question = item["input"]

        expected_answer = (
            item["expected_output"]
        )

        # -------------------------------------------------------------
        # Run complete RAG pipeline
        # -------------------------------------------------------------

        result = run_pipeline(
            pipeline=pipeline,
            question=question,
        )

        # -------------------------------------------------------------
        # Extract final answer and retrieved context
        # -------------------------------------------------------------

        actual_answer = extract_answer(
            result
        )

        retrieval_context = extract_context(
            result
        )

        # -------------------------------------------------------------
        # Keep the existing API throttling delay
        # -------------------------------------------------------------

        time.sleep(7)

        # -------------------------------------------------------------
        # Create DeepEval test case
        # -------------------------------------------------------------

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
        )

        test_cases.append(
            test_case
        )

    return test_cases


# =============================================================================
# RUN PIPELINE EVALUATION
# =============================================================================

def evaluate_pipeline():

    config = load_config()

    # -------------------------------------------------------------------------
    # Golden dataset
    # -------------------------------------------------------------------------

    dataset_path = (
        PROJECT_ROOT
        / config.dataset.path
    )

    golden_dataset = load_golden_dataset(
        dataset_path
    )

    # -------------------------------------------------------------------------
    # Create LangGraph RAG pipeline
    # -------------------------------------------------------------------------

    pipeline = create_pipeline()

    # -------------------------------------------------------------------------
    # Generate DeepEval test cases
    # -------------------------------------------------------------------------

    test_cases = build_test_cases(
        golden_dataset=golden_dataset,
        pipeline=pipeline,
    )

    # -------------------------------------------------------------------------
    # Evaluation model
    # -------------------------------------------------------------------------

    eval_model = OpenAIModel(
        model=config.evaluation.model,
        temperature=0,
    )

    # =========================================================================
    # METRIC 1 — FAITHFULNESS
    # =========================================================================

    faithfulness = FaithfulnessMetric(
        threshold=config.evaluation.threshold,
        model=eval_model,
        include_reason=True,
    )

    # =========================================================================
    # METRIC 2 — ANSWER RELEVANCY
    # =========================================================================

    answer_relevancy = AnswerRelevancyMetric(
        threshold=config.evaluation.threshold,
        model=eval_model,
        include_reason=True,
    )

    # =========================================================================
    # METRIC 3 — ANSWER CORRECTNESS
    # =========================================================================

    answer_correctness = GEval(
        name="Answer Correctness",

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],

        criteria="""
        Evaluate whether the actual answer correctly answers
        the user's question.

        Compare the actual output with the expected output.

        Penalize:
        - incorrect facts
        - incorrect policy terms
        - incorrect numbers
        - incorrect conditions
        - missing important information
        - answers that contradict the expected answer

        Minor wording differences should not be penalized
        when the meaning is correct.
        """,

        model=eval_model,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]

    # =========================================================================
    # TIMESTAMPED OUTPUT DIRECTORIES
    # =========================================================================

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

    # =========================================================================
    # RUN DEEPEVAL
    # =========================================================================

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

    # =========================================================================
    # RETURN INFORMATION TO EXPERIMENT PIPELINE
    # =========================================================================

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

    evaluate_pipeline()