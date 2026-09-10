
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from insurance_rag.utils.config import load_config
from insurance_rag.workflow.rag_workflow import (
    create_rag_graph,
)


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
    / "application_operational"
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
# MODEL COST
# =============================================================================

def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
):

    input_cost = (
        input_tokens
        / 1_000_000
        * input_cost_per_1m
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * output_cost_per_1m
    )

    return (
        input_cost
        + output_cost
    )


# =============================================================================
# RUN ONE REQUEST
# =============================================================================

def run_request(
    pipeline,
    question: str,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
):

    start_time = time.perf_counter()

    result = pipeline.invoke(
        {
            "question": question,
            "retrieved_context": [],
            "answer": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model": "",
        }
    )

    end_time = time.perf_counter()

    latency = (
        end_time
        - start_time
    )

    input_tokens = result.get(
        "input_tokens",
        0,
    )

    output_tokens = result.get(
        "output_tokens",
        0,
    )

    total_tokens = result.get(
        "total_tokens",
        0,
    )

    cost = calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_per_1m=input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
    )

    return {
        "question": question,
        "latency_seconds": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
        "model": result.get(
            "model",
            "",
        ),
    }


# =============================================================================
# PERCENTILES
# =============================================================================

def calculate_percentiles(
    latencies: list[float],
):

    if not latencies:

        raise ValueError(
            "No latency measurements available."
        )

    if len(latencies) == 1:

        return {
            "p50": latencies[0],
            "p95": latencies[0],
            "p99": latencies[0],
        }

    quantiles = statistics.quantiles(
        latencies,
        n=100,
        method="inclusive",
    )

    return {
        "p50": quantiles[49],
        "p95": quantiles[94],
        "p99": quantiles[98],
    }


# =============================================================================
# CREATE REPORT
# =============================================================================

def create_report(
    statistics_data,
    results,
    report_path: Path,
):

    lines = [

        "# Application Operational Evaluation",
        "",
        f"Total requests: {statistics_data['num_requests']}",
        "",
        "## Latency",
        "",
        f"- P50: {statistics_data['latency_p50_seconds']:.4f} seconds",
        f"- P95: {statistics_data['latency_p95_seconds']:.4f} seconds",
        f"- P99: {statistics_data['latency_p99_seconds']:.4f} seconds",
        "",
        "## Cost",
        "",
        f"- Total cost: ${statistics_data['total_cost_usd']:.6f}",
        f"- Average cost/request: ${statistics_data['average_cost_usd']:.6f}",
        "",
        "## Token Usage",
        "",
        f"- Input tokens: {statistics_data['total_input_tokens']}",
        f"- Output tokens: {statistics_data['total_output_tokens']}",
        f"- Total tokens: {statistics_data['total_tokens']}",
        "",
    ]

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results,
    statistics_data,
    results_path: Path,
):

    payload = {
        "statistics": statistics_data,
        "requests": results,
    }

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_application_operational():

    config = load_config()

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    dataset_path = (
        PROJECT_ROOT
        / config.dataset.path
    )

    golden_dataset = load_golden_dataset(
        dataset_path
    )

    # -------------------------------------------------------------------------
    # Model pricing
    #
    # Replace these values with the pricing of the model configured in
    # config.generation.model.
    # -------------------------------------------------------------------------

    input_cost_per_1m = (
        config.generation.input_cost_per_1m
    )

    output_cost_per_1m = (
        config.generation.output_cost_per_1m
    )

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    pipeline = create_rag_graph()

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    results_dir = (
        REPORT_ROOT
        / "results"
        / timestamp
    )

    reports_dir = (
        REPORT_ROOT
        / "reports"
        / timestamp
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Run requests
    # -------------------------------------------------------------------------

    results = []

    for item in golden_dataset:

        question = item["input"]

        result = run_request(
            pipeline=pipeline,
            question=question,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # Aggregate latency
    # -------------------------------------------------------------------------

    latencies = [
        item["latency_seconds"]
        for item in results
    ]

    percentiles = calculate_percentiles(
        latencies
    )

    # -------------------------------------------------------------------------
    # Aggregate token usage
    # -------------------------------------------------------------------------

    total_input_tokens = sum(
        item["input_tokens"]
        for item in results
    )

    total_output_tokens = sum(
        item["output_tokens"]
        for item in results
    )

    total_tokens = sum(
        item["total_tokens"]
        for item in results
    )

    # -------------------------------------------------------------------------
    # Aggregate cost
    # -------------------------------------------------------------------------

    total_cost = sum(
        item["cost_usd"]
        for item in results
    )

    average_cost = (
        total_cost
        / len(results)
        if results
        else 0
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    statistics_data = {

        "num_requests":
            len(results),

        "latency_p50_seconds":
            percentiles["p50"],

        "latency_p95_seconds":
            percentiles["p95"],

        "latency_p99_seconds":
            percentiles["p99"],

        "total_input_tokens":
            total_input_tokens,

        "total_output_tokens":
            total_output_tokens,

        "total_tokens":
            total_tokens,

        "total_cost_usd":
            total_cost,

        "average_cost_usd":
            average_cost,

        "model":
            config.generation.model,
    }

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------

    results_path = (
        results_dir
        / f"evaluation_{timestamp.replace('-', '').replace(':', '')}.json"
    )

    save_results(
        results=results,
        statistics_data=statistics_data,
        results_path=results_path,
    )

    # -------------------------------------------------------------------------
    # Save Markdown report
    # -------------------------------------------------------------------------

    report_path = (
        reports_dir
        / f"evaluation_{timestamp.replace('-', '').replace(':', '')}.md"
    )

    create_report(
        statistics_data=statistics_data,
        results=results,
        report_path=report_path,
    )

    print()
    print("=" * 80)
    print("APPLICATION OPERATIONAL EVALUATION")
    print("=" * 80)
    print()
    print(
        f"Requests: {statistics_data['num_requests']}"
    )
    print()
    print("Latency:")
    print(
        f"  P50: {statistics_data['latency_p50_seconds']:.4f}s"
    )
    print(
        f"  P95: {statistics_data['latency_p95_seconds']:.4f}s"
    )
    print(
        f"  P99: {statistics_data['latency_p99_seconds']:.4f}s"
    )
    print()
    print("Cost:")
    print(
        f"  Total: ${statistics_data['total_cost_usd']:.6f}"
    )
    print(
        f"  Average/request: ${statistics_data['average_cost_usd']:.6f}"
    )
    print()
    print(
        f"Results saved to: {results_path}"
    )
    print(
        f"Report saved to: {report_path}"
    )
    print()

    return {
        "statistics": statistics_data,
        "results_path": results_path,
        "report_path": report_path,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    evaluate_application_operational()