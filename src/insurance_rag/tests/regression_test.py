from pathlib import Path
import json
import mlflow


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "insurance-rag-experiments"

NOISE_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "regression"
    / "noise_thresholds.json"
)

REGRESSION_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "regression"
    / "regression_report.json"
)


# Metrics where a regression should block the candidate.
CRITICAL_METRICS = {
    "application_quality_application_correctness_geval",
    "application_quality_application_completeness_geval",
    "application_safety_scope_adherence_geval",
    "application_safety_leakage_geval",
    "application_safety_toxicity",
}


# =============================================================================
# HELPERS
# =============================================================================

def load_noise_thresholds():
    """Load noise thresholds generated from dedicated noise runs."""

    if not NOISE_THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Noise threshold file not found:\n{NOISE_THRESHOLD_PATH}"
        )

    with open(NOISE_THRESHOLD_PATH, "r") as f:
        return json.load(f)


def get_threshold_data(noise_thresholds, metric_name):
    """Return threshold configuration for a metric."""

    threshold_data = noise_thresholds["metrics"].get(metric_name)

    if threshold_data is None:
        raise ValueError(
            f"No noise threshold found for metric: {metric_name}"
        )

    return threshold_data


def get_run_by_tag(client, experiment_id, tag_key, tag_value):
    """Find exactly one run using an MLflow tag."""

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{tag_key} = '{tag_value}'",
        order_by=["attributes.start_time DESC"],
    )

    if len(runs) == 0:
        raise ValueError(
            f"No MLflow run found with {tag_key}={tag_value}"
        )

    if len(runs) > 1:
        raise ValueError(
            f"Expected exactly one run with "
            f"{tag_key}={tag_value}, but found {len(runs)}"
        )

    return runs[0]


def classify_metric_difference(
    best_value,
    candidate_value,
    threshold,
    direction,
):
    """
    Compare candidate against best using the noise threshold.

    Returns:
        IMPROVEMENT
        REGRESSION
        WITHIN_NOISE
    """

    if direction == "higher_is_better":
        difference = candidate_value - best_value
    elif direction == "lower_is_better":
        difference = best_value - candidate_value
    else:
        raise ValueError(
            f"Unknown metric direction: {direction}"
        )

    if difference > threshold:
        status = "IMPROVEMENT"
    elif difference < -threshold:
        status = "REGRESSION"
    else:
        status = "WITHIN_NOISE"

    return difference, status


# =============================================================================
# MAIN REGRESSION TEST
# =============================================================================

def main():

    print("=" * 80)
    print("REGRESSION TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # MLflow setup
    # -------------------------------------------------------------------------

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    experiment = client.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError(
            f"MLflow experiment not found: {MLFLOW_EXPERIMENT_NAME}"
        )

    experiment_id = experiment.experiment_id

    # -------------------------------------------------------------------------
    # Load noise thresholds
    # -------------------------------------------------------------------------

    noise_thresholds = load_noise_thresholds()

    # -------------------------------------------------------------------------
    # Find best and candidate
    # -------------------------------------------------------------------------

    best_run = get_run_by_tag(
        client,
        experiment_id,
        "use",
        "best",
    )

    candidate_run = get_run_by_tag(
        client,
        experiment_id,
        "use",
        "candidate",
    )

    best_run_id = best_run.info.run_id
    candidate_run_id = candidate_run.info.run_id

    print()
    print(f"Best      : {best_run_id}")
    print(f"Candidate : {candidate_run_id}")
    print()

    # -------------------------------------------------------------------------
    # Compare metrics
    # -------------------------------------------------------------------------

    results = []

    critical_regressions = []
    non_critical_regressions = []

    for metric_name, threshold_data in noise_thresholds["metrics"].items():

        threshold = threshold_data["noise_threshold"]
        direction = threshold_data["direction"]
        category = threshold_data.get("category")

        best_value = best_run.data.metrics.get(metric_name)
        candidate_value = candidate_run.data.metrics.get(metric_name)

        # Metric missing from either run.
        if best_value is None or candidate_value is None:

            results.append(
                {
                    "metric": metric_name,
                    "category": category,
                    "direction": direction,
                    "best": best_value,
                    "candidate": candidate_value,
                    "threshold": threshold,
                    "difference": None,
                    "status": "MISSING",
                    "critical": metric_name in CRITICAL_METRICS,
                }
            )

            continue

        difference, status = classify_metric_difference(
            best_value=best_value,
            candidate_value=candidate_value,
            threshold=threshold,
            direction=direction,
        )

        critical = metric_name in CRITICAL_METRICS

        result = {
            "metric": metric_name,
            "category": category,
            "direction": direction,
            "best": best_value,
            "candidate": candidate_value,
            "threshold": threshold,
            "difference": difference,
            "status": status,
            "critical": critical,
        }

        results.append(result)

        # Track regressions separately.
        if status == "REGRESSION":

            if critical:
                critical_regressions.append(metric_name)
            else:
                non_critical_regressions.append(metric_name)

        print(
            f"{metric_name:<70} "
            f"{best_value:>10.6f} "
            f"{candidate_value:>10.6f} "
            f"{threshold:>10.6f} "
            f"{status}"
        )

    # -------------------------------------------------------------------------
    # Final decision
    # -------------------------------------------------------------------------

    passed = len(critical_regressions) == 0

    print()
    print("=" * 80)
    print(f"Critical regressions    : {len(critical_regressions)}")
    print(f"Non-critical regressions: {len(non_critical_regressions)}")
    print("=" * 80)

    if critical_regressions:

        print()
        print("CRITICAL REGRESSIONS:")

        for metric in critical_regressions:
            print(f"  - {metric}")

        print()
        print("RESULT: FAIL")
        print("Candidate should NOT be promoted.")

        # Candidate failed regression.
        # Remove the use tag entirely.
        client.delete_tag(
            run_id=candidate_run_id,
            key="use",
        )

        print()
        print(
            f"Removed tag 'use' from candidate run "
            f"{candidate_run_id}"
        )

    else:

        print()
        print("RESULT: PASS")
        print("Candidate passed regression testing.")

        if non_critical_regressions:

            print()
            print("Non-critical regressions:")

            for metric in non_critical_regressions:
                print(f"  - {metric}")

        print()
        print(
            "Candidate remains tagged as use=candidate "
            "and can proceed to promotion testing."
        )

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------

    report = {
        "test": "regression",
        "experiment": {
            "name": MLFLOW_EXPERIMENT_NAME,
            "experiment_id": experiment_id,
            "tracking_uri": MLFLOW_TRACKING_URI,
        },
        "runs": {
            "best": best_run_id,
            "candidate": candidate_run_id,
        },
        "critical_metrics": sorted(CRITICAL_METRICS),
        "results": results,
        "summary": {
            "critical_regressions": len(critical_regressions),
            "non_critical_regressions": len(non_critical_regressions),
            "critical_regression_metrics": critical_regressions,
            "non_critical_regression_metrics": non_critical_regressions,
            "passed": passed,
        },
    }

    REGRESSION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(REGRESSION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print(f"Report saved to:\n{REGRESSION_REPORT_PATH}")
    print()

    # Exit with non-zero status if regression failed.
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()