import json
from datetime import datetime
from pathlib import Path

from deepeval.evaluate import (
    AsyncConfig,
    DisplayConfig,
    evaluate,
)

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)

from deepeval.models import OpenAIModel

from deepeval.test_case import LLMTestCase

from insurance_rag.generation.generator import (
    InsuranceGenerator,
)

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
    / "generator"
)


# =============================================================================
# LOAD EVALUATION DATASET
# =============================================================================

def load_evaluation_dataset(
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
# CREATE GENERATOR
# =============================================================================

def create_generator():

    return InsuranceGenerator()


# =============================================================================
# BUILD DEEPEVAL TEST CASES
# =============================================================================

def build_test_cases(
    dataset,
    generator,
):

    test_cases = []

    for item in dataset:

        # ---------------------------------------------------------------------
        # Question
        # ---------------------------------------------------------------------

        question = item["input"]

        # ---------------------------------------------------------------------
        # Expected answer
        # ---------------------------------------------------------------------

        expected_answer = (
            item["expected_output"]
        )

        # ---------------------------------------------------------------------
        # Retrieval context
        #
        # Faithfulness requires retrieval context.
        # ---------------------------------------------------------------------

        retrieval_context = (
            item["retrieved_context"]
        )

        if not isinstance(
            retrieval_context,
            list,
        ):

            raise TypeError(
                "retrieved_context must be a list."
            )

        if not all(
            isinstance(context, str)
            for context in retrieval_context
        ):

            raise TypeError(
                "Every item in retrieved_context "
                "must be a string."
            )

        # ---------------------------------------------------------------------
        # Generate answer
        # ---------------------------------------------------------------------

        generation_result = (
            generator.generate(
                question=question,
                contexts=retrieval_context,
            )
        )

        # ---------------------------------------------------------------------
        # Extract answer from generator response
        # ---------------------------------------------------------------------

        actual_answer = (
            generation_result["answer"]
        )

        if not isinstance(
            actual_answer,
            str,
        ):

            raise TypeError(
                "Generator 'answer' must be a string."
            )

        # ---------------------------------------------------------------------
        # Create DeepEval test case
        # ---------------------------------------------------------------------

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
# RUN GENERATOR EVALUATION
# =============================================================================

def evaluate_generator():

    config = load_config()

    # =========================================================================
    # LOAD DATASET
    # =========================================================================

    dataset_path = (
        PROJECT_ROOT
        / config.dataset.path
    )

    dataset = load_evaluation_dataset(
        dataset_path
    )

    # =========================================================================
    # CREATE GENERATOR
    # =========================================================================

    generator = create_generator()

    # =========================================================================
    # BUILD DEEPEVAL TEST CASES
    # =========================================================================

    test_cases = build_test_cases(
        dataset=dataset,
        generator=generator,
    )

    # =========================================================================
    # EVALUATION MODEL
    # =========================================================================

    eval_model = OpenAIModel(
        model=config.evaluation.model,
        temperature=0,
    )

    # =========================================================================
    # FAITHFULNESS
    #
    # Checks whether the generated answer is supported by the retrieved
    # context.
    # =========================================================================

    faithfulness = FaithfulnessMetric(
        threshold=config.evaluation.threshold,
        model=eval_model,
        include_reason=True,
    )

    # =========================================================================
    # ANSWER RELEVANCY
    #
    # Checks whether the generated answer is relevant to the question.
    # =========================================================================

    answer_relevancy = AnswerRelevancyMetric(
        threshold=config.evaluation.threshold,
        model=eval_model,
        include_reason=True,
    )

    metrics = [
        faithfulness,
        answer_relevancy,
    ]

    # =========================================================================
    # CREATE OUTPUT DIRECTORIES
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
    # RETURN RESULTS
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

    evaluate_generator()