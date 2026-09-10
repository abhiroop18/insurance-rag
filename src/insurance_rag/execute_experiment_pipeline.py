from pathlib import Path
import json
import os

import mlflow

from insurance_rag.evaluation.evaluate_retriever import (
    evaluate_retriever,
)

from insurance_rag.evaluation.evaluate_generator import (
    evaluate_generator,
)

from insurance_rag.evaluation.evaluate_pipeline import (
    evaluate_pipeline,
)

from insurance_rag.evaluation.evaluate_application_quality import (
    evaluate_application_quality,
)

from insurance_rag.evaluation.evaluate_application_safety import (
    evaluate_application_safety,
)

from insurance_rag.evaluation.evaluate_application_operational import (
    evaluate_application_operational,
)

from insurance_rag.evaluation.eval_utils import (
    get_latest_results,
    get_latest_report,
    get_metrics_from_results,
    get_test_count,
)

from insurance_rag.utils.config import load_config
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_EXPERIMENT_NAME = (
    "insurance-rag-experiments"
)


# =============================================================================
# EVALUATION DIRECTORIES
# =============================================================================

RETRIEVER_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reports"
    / "retriever"
)

GENERATOR_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reports"
    / "generator"
)

PIPELINE_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reports"
    / "pipeline"
)

APPLICATION_QUALITY_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reports"
    / "application_quality"
)

APPLICATION_SAFETY_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reports"
    / "application_safety"
)

APPLICATION_OPERATIONAL_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reports"
    / "application_operational"
)


# =============================================================================
# CODE PATHS
# =============================================================================

RETRIEVER_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "retrieval"
    / "retriever.py"
)

GENERATOR_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "generation"
    / "generator.py"
)

PIPELINE_STATE_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "workflow"
    / "state.py"
)

PIPELINE_NODES_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "workflow"
    / "nodes.py"
)

PIPELINE_WORKFLOW_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "workflow"
    / "rag_workflow.py"
)

RETRIEVER_EVALUATION_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "evaluation"
    / "evaluate_retriever.py"
)

GENERATOR_EVALUATION_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "evaluation"
    / "evaluate_generator.py"
)

PIPELINE_EVALUATION_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "evaluation"
    / "evaluate_pipeline.py"
)

APPLICATION_QUALITY_EVALUATION_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "evaluation"
    / "evaluate_application_quality.py"
)

APPLICATION_SAFETY_EVALUATION_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "evaluation"
    / "evaluate_application_safety.py"
)

APPLICATION_OPERATIONAL_EVALUATION_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "evaluation"
    / "evaluate_application_operational.py"
)

EVAL_UTILS_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "evaluation"
    / "eval_utils.py"
)

CONFIG_CODE_PATH = (
    PROJECT_ROOT
    / "src"
    / "insurance_rag"
    / "utils"
    / "config.py"
)

PARAMS_PATH = (
    PROJECT_ROOT
    / "params.yaml"
)


# =============================================================================
# LOG CONFIGURATION
# =============================================================================

def log_configuration(config):

    mlflow.log_params({

        # ---------------------------------------------------------------------
        # Retrieval
        # ---------------------------------------------------------------------

        "embedding_model":
            config.vectorstore.embedding_model,

        "dense_top_k":
            config.retrieval.dense_top_k,

        "final_top_k":
            config.retrieval.final_top_k,

        "reranker_model":
            config.retrieval.reranker_model,

        # ---------------------------------------------------------------------
        # Generation
        # ---------------------------------------------------------------------

        "generation_model":
            config.generation.model,

        "generation_temperature":
            config.generation.temperature,

        "generation_prompt_name":
            config.generation.prompt_name,

        "generation_prompt_label":
            config.generation.prompt_label,

        "input_cost_per_1m":
            config.generation.input_cost_per_1m,

        "output_cost_per_1m":
            config.generation.output_cost_per_1m,

        # ---------------------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------------------

        "evaluation_model":
            config.evaluation.model,

        "evaluation_threshold":
            config.evaluation.threshold,

        # ---------------------------------------------------------------------
        # DeepEval
        # ---------------------------------------------------------------------

        "deepeval_max_concurrent":
            config.deepeval.max_concurrent,

        "deepeval_throttle_value":
            config.deepeval.throttle_value,

        "deepeval_run_async":
            config.deepeval.run_async,

        # ---------------------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------------------

        "dataset":
            config.dataset.path,
    })


# =============================================================================
# LOG EVALUATION ARTIFACTS
# =============================================================================

def log_evaluation_artifacts(
    evaluation_name: str,
    results_path: Path,
    report_path: Path,
):

    mlflow.log_artifact(
        str(results_path),
        artifact_path=(
            f"evaluation/{evaluation_name}/results"
        ),
    )

    mlflow.log_artifact(
        str(report_path),
        artifact_path=(
            f"evaluation/{evaluation_name}/reports"
        ),
    )


# =============================================================================
# LOG COMMON ARTIFACTS
# =============================================================================

def log_common_artifacts(
    golden_dataset_path: Path,
):

    # -------------------------------------------------------------------------
    # Golden dataset
    # -------------------------------------------------------------------------

    if golden_dataset_path.exists():

        mlflow.log_artifact(
            str(golden_dataset_path),
            artifact_path="datasets",
        )

    # -------------------------------------------------------------------------
    # params.yaml
    # -------------------------------------------------------------------------

    if PARAMS_PATH.exists():

        mlflow.log_artifact(
            str(PARAMS_PATH),
            artifact_path="configuration",
        )

    # -------------------------------------------------------------------------
    # Source code
    # -------------------------------------------------------------------------

    code_files = [

        # Retrieval
        RETRIEVER_CODE_PATH,

        # Generation
        GENERATOR_CODE_PATH,

        # LangGraph workflow
        PIPELINE_STATE_CODE_PATH,
        PIPELINE_NODES_CODE_PATH,
        PIPELINE_WORKFLOW_CODE_PATH,

        # Evaluation
        RETRIEVER_EVALUATION_CODE_PATH,
        GENERATOR_EVALUATION_CODE_PATH,
        PIPELINE_EVALUATION_CODE_PATH,

        # Application evaluation
        APPLICATION_QUALITY_EVALUATION_CODE_PATH,
        APPLICATION_SAFETY_EVALUATION_CODE_PATH,
        APPLICATION_OPERATIONAL_EVALUATION_CODE_PATH,

        # Evaluation utilities
        EVAL_UTILS_PATH,

        # Configuration
        CONFIG_CODE_PATH,
    ]

    for code_file in code_files:

        if code_file.exists():

            mlflow.log_artifact(
                str(code_file),
                artifact_path="code",
            )


# =============================================================================
# NORMALIZE METRIC NAME
# =============================================================================

def normalize_metric_name(
    metric_name: str,
):

    return (
        metric_name
        .lower()
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


# =============================================================================
# LOG DEEPEVAL METRICS
# =============================================================================

def log_deepeval_metrics(
    evaluation_name: str,
    results_path: Path,
):

    metrics = get_metrics_from_results(
        results_path
    )

    formatted_metrics = {}

    for metric_name, score in metrics.items():

        metric_key = (
            f"{evaluation_name}_"
            f"{normalize_metric_name(metric_name)}"
        )

        formatted_metrics[metric_key] = score

    if formatted_metrics:

        mlflow.log_metrics(
            formatted_metrics
        )

    return metrics


# =============================================================================
# LOG OPERATIONAL METRICS
# =============================================================================

def log_operational_metrics(
    results_path: Path,
):

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    statistics_data = data.get(
        "statistics",
        {},
    )

    if not statistics_data:

        raise ValueError(
            "Operational evaluation results "
            "do not contain a 'statistics' section."
        )

    metrics = {

        "latency_p50_seconds":
            statistics_data.get(
                "latency_p50_seconds",
                0,
            ),

        "latency_p95_seconds":
            statistics_data.get(
                "latency_p95_seconds",
                0,
            ),

        "latency_p99_seconds":
            statistics_data.get(
                "latency_p99_seconds",
                0,
            ),

        "total_input_tokens":
            statistics_data.get(
                "total_input_tokens",
                0,
            ),

        "total_output_tokens":
            statistics_data.get(
                "total_output_tokens",
                0,
            ),

        "total_tokens":
            statistics_data.get(
                "total_tokens",
                0,
            ),

        "total_cost_usd":
            statistics_data.get(
                "total_cost_usd",
                0,
            ),

        "average_cost_usd":
            statistics_data.get(
                "average_cost_usd",
                0,
            ),
    }

    formatted_metrics = {}

    for metric_name, score in metrics.items():

        metric_key = (
            "application_operational_"
            + normalize_metric_name(metric_name)
        )

        formatted_metrics[metric_key] = score

    if formatted_metrics:

        mlflow.log_metrics(
            formatted_metrics
        )

    return metrics


# =============================================================================
# RUN ONE DEEPEVAL EVALUATION
# =============================================================================

def run_deepeval_evaluation(
    evaluation_name: str,
    evaluation_function,
    evaluation_dir: Path,
):

    print()
    print("=" * 80)
    print(
        f"RUNNING {evaluation_name.upper()} EVALUATION"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Execute evaluation
    # -------------------------------------------------------------------------

    evaluation_function()

    # -------------------------------------------------------------------------
    # Find latest results
    # -------------------------------------------------------------------------

    results_path = get_latest_results(
        evaluation_dir / "results"
    )

    # -------------------------------------------------------------------------
    # Find latest report
    # -------------------------------------------------------------------------

    report_path = get_latest_report(
        evaluation_dir / "reports"
    )

    # -------------------------------------------------------------------------
    # Number of test cases
    # -------------------------------------------------------------------------

    num_test_cases = get_test_count(
        results_path
    )

    mlflow.log_param(
        f"{evaluation_name}_num_test_cases",
        num_test_cases,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = log_deepeval_metrics(
        evaluation_name=evaluation_name,
        results_path=results_path,
    )

    # -------------------------------------------------------------------------
    # Artifacts
    # -------------------------------------------------------------------------

    log_evaluation_artifacts(
        evaluation_name=evaluation_name,
        results_path=results_path,
        report_path=report_path,
    )

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------

    print()
    print(
        f"{evaluation_name.capitalize()} evaluation completed."
    )

    print(
        f"Results: {results_path}"
    )

    print(
        f"Report:  {report_path}"
    )

    print(
        f"Tests:   {num_test_cases}"
    )

    print(
        f"Metrics: {metrics}"
    )

    print("=" * 80)

    return {
        "results_path": results_path,
        "report_path": report_path,
        "num_test_cases": num_test_cases,
        "metrics": metrics,
    }


# =============================================================================
# RUN APPLICATION OPERATIONAL EVALUATION
# =============================================================================

def run_operational_evaluation():

    evaluation_name = (
        "application_operational"
    )

    evaluation_dir = (
        APPLICATION_OPERATIONAL_EVALUATION_DIR
    )

    print()
    print("=" * 80)
    print(
        "RUNNING APPLICATION OPERATIONAL EVALUATION"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Execute evaluation
    # -------------------------------------------------------------------------

    evaluation_result = (
        evaluate_application_operational()
    )

    # -------------------------------------------------------------------------
    # Use paths returned directly by the evaluator
    #
    # This is safer than searching for the latest file because the operational
    # evaluator already knows exactly which files belong to this run.
    # -------------------------------------------------------------------------

    results_path = Path(
        evaluation_result["results_path"]
    )

    report_path = Path(
        evaluation_result["report_path"]
    )

    statistics_data = (
        evaluation_result["statistics"]
    )

    # -------------------------------------------------------------------------
    # Log request count
    # -------------------------------------------------------------------------

    mlflow.log_param(
        f"{evaluation_name}_num_requests",
        statistics_data["num_requests"],
    )

    # -------------------------------------------------------------------------
    # Log metrics
    # -------------------------------------------------------------------------

    metrics = log_operational_metrics(
        results_path
    )

    # -------------------------------------------------------------------------
    # Log artifacts
    # -------------------------------------------------------------------------

    log_evaluation_artifacts(
        evaluation_name=evaluation_name,
        results_path=results_path,
        report_path=report_path,
    )

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------

    print()
    print(
        "Application operational evaluation completed."
    )

    print(
        f"Results: {results_path}"
    )

    print(
        f"Report:  {report_path}"
    )

    print(
        f"Requests: "
        f"{statistics_data['num_requests']}"
    )

    print(
        f"Metrics: {metrics}"
    )

    print("=" * 80)

    return {
        "results_path": results_path,
        "report_path": report_path,
        "num_requests":
            statistics_data["num_requests"],
        "metrics": metrics,
    }


# =============================================================================
# EXECUTE EXPERIMENT
# =============================================================================

def execute_experiment():

    config = load_config()

    # -------------------------------------------------------------------------
    # Configure MLflow
    # -------------------------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    # -------------------------------------------------------------------------
    # Golden dataset
    # -------------------------------------------------------------------------

    golden_dataset_path = (
        PROJECT_ROOT
        / config.dataset.path
    )

    # -------------------------------------------------------------------------
    # Start MLflow run
    # -------------------------------------------------------------------------

    with mlflow.start_run():

        # =====================================================================
        # CONFIGURATION
        # =====================================================================

        log_configuration(
            config
        )

        # =====================================================================
        # RETRIEVER EVALUATION
        # =====================================================================

        run_deepeval_evaluation(
            evaluation_name="retriever",
            evaluation_function=evaluate_retriever,
            evaluation_dir=RETRIEVER_EVALUATION_DIR,
        )

        # =====================================================================
        # GENERATOR EVALUATION
        # =====================================================================

        run_deepeval_evaluation(
            evaluation_name="generator",
            evaluation_function=evaluate_generator,
            evaluation_dir=GENERATOR_EVALUATION_DIR,
        )

        # =====================================================================
        # END-TO-END PIPELINE EVALUATION
        # =====================================================================

        run_deepeval_evaluation(
            evaluation_name="pipeline",
            evaluation_function=evaluate_pipeline,
            evaluation_dir=PIPELINE_EVALUATION_DIR,
        )

        # =====================================================================
        # APPLICATION QUALITY EVALUATION
        # =====================================================================

        run_deepeval_evaluation(
            evaluation_name="application_quality",
            evaluation_function=evaluate_application_quality,
            evaluation_dir=APPLICATION_QUALITY_EVALUATION_DIR,
        )

        # =====================================================================
        # APPLICATION SAFETY EVALUATION
        # =====================================================================

        run_deepeval_evaluation(
            evaluation_name="application_safety",
            evaluation_function=evaluate_application_safety,
            evaluation_dir=APPLICATION_SAFETY_EVALUATION_DIR,
        )

        # =====================================================================
        # APPLICATION OPERATIONAL EVALUATION
        # =====================================================================

        run_operational_evaluation()

        # =====================================================================
        # COMMON ARTIFACTS
        # =====================================================================

        log_common_artifacts(
            golden_dataset_path=golden_dataset_path,
        )

        # =====================================================================
        # TAGS
        # =====================================================================

        mlflow.set_tags({

            "evaluation_type":
                (
                    "retriever + generator + pipeline "
                    "+ application quality + application safety "
                    "+ application operational"
                ),

            "framework":
                "DeepEval",

            "retrieval_pipeline":
                "dense + reranking",

            "workflow":
                "LangGraph",

            "application_quality":
                (
                    "correctness + completeness + style"
                ),

            "application_safety":
                (
                    "scope adherence + leakage + toxicity"
                ),

            "application_operational":
                (
                    "latency + token usage + cost"
                ),

            "status":
                "completed",
        })


        mlflow.set_tag("use", "candidate")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    execute_experiment()