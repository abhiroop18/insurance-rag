from pathlib import Path
import json
import statistics

import mlflow
from mlflow import MlflowClient


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

MLFLOW_TRACKING_URI = (
    "http://127.0.0.1:5000"
)

MLFLOW_EXPERIMENT_NAME = (
    "insurance-rag-experiments"
)

# MLflow tag used to identify the 3 noise runs
NOISE_TAG_KEY = "use"
NOISE_TAG_VALUE = "noise_threshold"

# We expect exactly 3 noise runs
EXPECTED_NOISE_RUNS = 3

# Noise threshold = multiplier × sample standard deviation
NOISE_MULTIPLIER = 2.0


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "regression"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "noise_thresholds.json"
)


# =============================================================================
# METRICS USED FOR REGRESSION TESTING
# =============================================================================

METRICS = {

    # -------------------------------------------------------------------------
    # Application Quality
    # -------------------------------------------------------------------------

    "application_quality_application_correctness_geval": {
        "direction": "higher_is_better",
        "category": "quality",
    },

    "application_quality_application_completeness_geval": {
        "direction": "higher_is_better",
        "category": "quality",
    },

    "application_quality_application_style_geval": {
        "direction": "higher_is_better",
        "category": "quality",
    },

    # -------------------------------------------------------------------------
    # Application Safety
    # -------------------------------------------------------------------------

    "application_safety_scope_adherence_geval": {
        "direction": "higher_is_better",
        "category": "safety",
    },

    "application_safety_leakage_geval": {
        "direction": "higher_is_better",
        "category": "safety",
    },

    "application_safety_toxicity": {
        # IMPORTANT:
        # Change this to "lower_is_better" if this is an actual toxicity
        # score where 0 = no toxicity and higher = more toxicity.
        #
        # If this metric is instead a pass rate where 1 = safe and 0 = unsafe,
        # change this to "higher_is_better".
        "direction": "higher_is_better",
        "category": "safety",
    },

    # -------------------------------------------------------------------------
    # Operational
    # -------------------------------------------------------------------------

    "application_operational_latency_p50_seconds": {
        "direction": "lower_is_better",
        "category": "operational",
    },

    "application_operational_latency_p95_seconds": {
        "direction": "lower_is_better",
        "category": "operational",
    },

    "application_operational_latency_p99_seconds": {
        "direction": "lower_is_better",
        "category": "operational",
    },

    "application_operational_total_input_tokens": {
        "direction": "lower_is_better",
        "category": "operational",
    },

    "application_operational_total_output_tokens": {
        "direction": "lower_is_better",
        "category": "operational",
    },

    "application_operational_total_tokens": {
        "direction": "lower_is_better",
        "category": "operational",
    },

    "application_operational_total_cost_usd": {
        "direction": "lower_is_better",
        "category": "operational",
    },

    "application_operational_average_cost_usd": {
        "direction": "lower_is_better",
        "category": "operational",
    },
}


# =============================================================================
# GET EXPERIMENT
# =============================================================================

def get_experiment(
    client: MlflowClient,
):
    """
    Get the MLflow experiment by name.
    """

    experiment = (
        client.get_experiment_by_name(
            MLFLOW_EXPERIMENT_NAME
        )
    )

    if experiment is None:

        raise RuntimeError(
            f"MLflow experiment not found: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

    return experiment


# =============================================================================
# GET NOISE RUNS
# =============================================================================

def get_noise_runs(
    client: MlflowClient,
    experiment_id: str,
):
    """
    Get completed MLflow runs tagged:

        use = noise_threshold
    """

    runs = client.search_runs(

        experiment_ids=[
            experiment_id
        ],

        filter_string=(
            f"tags.{NOISE_TAG_KEY} = "
            f"'{NOISE_TAG_VALUE}'"
        ),

        order_by=[
            "start_time ASC"
        ],
    )

    # Only use successfully completed runs.
    finished_runs = [
        run
        for run in runs
        if run.info.status == "FINISHED"
    ]

    return finished_runs


# =============================================================================
# VALIDATE NOISE RUNS
# =============================================================================

def validate_noise_runs(
    runs,
):
    """
    Make sure we have exactly the expected number of noise runs.
    """

    if len(runs) != EXPECTED_NOISE_RUNS:

        raise RuntimeError(
            "\n"
            f"Expected exactly "
            f"{EXPECTED_NOISE_RUNS} finished noise runs, "
            f"but found {len(runs)}.\n\n"
            f"Required MLflow tag:\n"
            f"    {NOISE_TAG_KEY}={NOISE_TAG_VALUE}\n"
        )


# =============================================================================
# CALCULATE METRIC STATISTICS
# =============================================================================

def calculate_metric_statistics(
    metric_name: str,
    values: list[float],
):
    """
    Calculate statistics for one metric.

    Noise threshold:

        2 × sample standard deviation
    """

    if len(values) != EXPECTED_NOISE_RUNS:

        raise RuntimeError(
            f"Metric '{metric_name}' has "
            f"{len(values)} values, but expected "
            f"{EXPECTED_NOISE_RUNS}."
        )

    mean_value = statistics.mean(
        values
    )

    std_value = statistics.stdev(
        values
    )

    minimum = min(values)
    maximum = max(values)

    value_range = (
        maximum - minimum
    )

    noise_threshold = (
        NOISE_MULTIPLIER
        * std_value
    )

    return {
        "values": values,
        "mean": mean_value,
        "std": std_value,
        "min": minimum,
        "max": maximum,
        "range": value_range,
        "noise_multiplier": NOISE_MULTIPLIER,
        "noise_threshold": noise_threshold,
    }


# =============================================================================
# CALCULATE ALL THRESHOLDS
# =============================================================================

def calculate_noise_thresholds(
    runs,
):
    """
    Calculate noise thresholds for all configured metrics.
    """

    results = {}

    for metric_name, config in METRICS.items():

        values = []

        missing_runs = []

        for run in runs:

            if metric_name not in run.data.metrics:

                missing_runs.append(
                    run.info.run_id
                )

                continue

            values.append(
                run.data.metrics[metric_name]
            )

        # ---------------------------------------------------------------------
        # Every noise run must contain every metric.
        # ---------------------------------------------------------------------

        if missing_runs:

            raise RuntimeError(
                "\n"
                f"Metric '{metric_name}' is missing "
                f"from one or more noise runs.\n\n"
                f"Missing run IDs:\n"
                + "\n".join(
                    f"  - {run_id}"
                    for run_id in missing_runs
                )
            )

        statistics_data = (
            calculate_metric_statistics(
                metric_name=metric_name,
                values=values,
            )
        )

        results[metric_name] = {

            "category":
                config["category"],

            "direction":
                config["direction"],

            **statistics_data,
        }

    return results


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    experiment,
    runs,
    thresholds,
):
    """
    Save calculated thresholds to JSON.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {

        "experiment": {
            "name":
                experiment.name,

            "experiment_id":
                experiment.experiment_id,

            "tracking_uri":
                MLFLOW_TRACKING_URI,
        },

        "configuration": {

            "noise_tag": {
                "key":
                    NOISE_TAG_KEY,

                "value":
                    NOISE_TAG_VALUE,
            },

            "number_of_runs":
                EXPECTED_NOISE_RUNS,

            "noise_multiplier":
                NOISE_MULTIPLIER,

            "method":
                (
                    "2 x sample standard deviation"
                ),
        },

        "runs": [
            {
                "run_id":
                    run.info.run_id,

                "run_name":
                    run.info.run_name,

                "status":
                    run.info.status,

                "start_time":
                    run.info.start_time,
            }

            for run in runs
        ],

        "metrics":
            thresholds,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(
    runs,
    thresholds,
):
    """
    Print a readable summary to the terminal.
    """

    print()
    print("=" * 120)
    print("NOISE THRESHOLD CALCULATION")
    print("=" * 120)

    print()
    print(
        f"Number of noise runs: "
        f"{len(runs)}"
    )

    print(
        f"Noise multiplier: "
        f"{NOISE_MULTIPLIER}"
    )

    print(
        "Method: "
        "2 × sample standard deviation"
    )

    print()
    print("Noise runs:")
    print("-" * 120)

    for index, run in enumerate(
        runs,
        start=1,
    ):

        print(
            f"{index}. "
            f"{run.info.run_id}"
        )

    print()
    print(
        f"{'Metric':<65}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Threshold':>15}"
    )

    print("-" * 120)

    for metric_name, data in thresholds.items():

        print(
            f"{metric_name:<65}"
            f"{data['mean']:>12.6f}"
            f"{data['std']:>12.6f}"
            f"{data['noise_threshold']:>15.6f}"
        )

    print("-" * 120)

    print()
    print(
        f"Threshold file:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 120)


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Configure MLflow
    # -------------------------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # -------------------------------------------------------------------------
    # Get experiment
    # -------------------------------------------------------------------------

    experiment = get_experiment(
        client
    )

    # -------------------------------------------------------------------------
    # Get noise runs
    # -------------------------------------------------------------------------

    runs = get_noise_runs(
        client=client,
        experiment_id=experiment.experiment_id,
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_noise_runs(
        runs
    )

    # -------------------------------------------------------------------------
    # Calculate thresholds
    # -------------------------------------------------------------------------

    thresholds = calculate_noise_thresholds(
        runs
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_results(
        experiment=experiment,
        runs=runs,
        thresholds=thresholds,
    )

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    print_results(
        runs=runs,
        thresholds=thresholds,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()