from pathlib import Path
import json
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()
# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_EXPERIMENT_NAME = "insurance-rag-experiments"

NOISE_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "regression"
    / "noise_thresholds.json"
)

PROMOTION_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "regression"
    / "promotion_report.json"
)


# =============================================================================
# METRICS USED FOR PROMOTION
# =============================================================================

PROMOTION_METRICS = {
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


def get_run_by_tag(client, experiment_id, tag_key, tag_value):
    """Find exactly one MLflow run using a tag."""

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


def get_threshold_data(noise_thresholds, metric_name):
    """
    Get threshold information from the nested metrics section.

    noise_thresholds.json structure:

        {
            ...
            "metrics": {
                "metric_name": {
                    "direction": "...",
                    "noise_threshold": ...
                }
            }
        }
    """

    threshold_data = noise_thresholds["metrics"].get(metric_name)

    if threshold_data is None:
        raise ValueError(
            f"No noise threshold found for promotion metric: "
            f"{metric_name}"
        )

    return threshold_data


def classify_metric_difference(
    best_value,
    candidate_value,
    threshold,
    direction,
):
    """
    Compare candidate against best.

    Returns:
        difference
        status

    A difference greater than the noise threshold is meaningful.
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
# MAIN PROMOTION TEST
# =============================================================================

def main():

    print("=" * 80)
    print("PROMOTION TEST")
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
    # Compare promotion metrics
    # -------------------------------------------------------------------------

    results = []

    meaningful_improvements = []
    meaningful_regressions = []

    for metric_name in PROMOTION_METRICS:

        threshold_data = get_threshold_data(
            noise_thresholds,
            metric_name,
        )

        threshold = threshold_data["noise_threshold"]
        direction = threshold_data["direction"]
        category = threshold_data.get("category")

        best_value = best_run.data.metrics.get(metric_name)
        candidate_value = candidate_run.data.metrics.get(metric_name)

        # ---------------------------------------------------------------------
        # Missing metric
        # ---------------------------------------------------------------------

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
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Compare
        # ---------------------------------------------------------------------

        difference, status = classify_metric_difference(
            best_value=best_value,
            candidate_value=candidate_value,
            threshold=threshold,
            direction=direction,
        )

        result = {
            "metric": metric_name,
            "category": category,
            "direction": direction,
            "best": best_value,
            "candidate": candidate_value,
            "threshold": threshold,
            "difference": difference,
            "status": status,
        }

        results.append(result)

        # ---------------------------------------------------------------------
        # Track meaningful changes
        # ---------------------------------------------------------------------

        if status == "IMPROVEMENT":

            meaningful_improvements.append(metric_name)

        elif status == "REGRESSION":

            meaningful_regressions.append(metric_name)

        print(
            f"{metric_name:<70} "
            f"{best_value:>10.6f} "
            f"{candidate_value:>10.6f} "
            f"{threshold:>10.6f} "
            f"{status}"
        )

    # -------------------------------------------------------------------------
    # Promotion decision
    # -------------------------------------------------------------------------
    #
    # Candidate is promoted only if:
    #
    #   1. There is at least one meaningful improvement
    #   2. There are zero meaningful regressions
    #
    # WITHIN_NOISE does not count as either improvement or regression.
    # -------------------------------------------------------------------------

    passed = (
        len(meaningful_improvements) > 0
        and len(meaningful_regressions) == 0
    )

    print()
    print("=" * 80)
    print(
        f"Meaningful improvements : "
        f"{len(meaningful_improvements)}"
    )
    print(
        f"Meaningful regressions   : "
        f"{len(meaningful_regressions)}"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Promotion PASSED
    # -------------------------------------------------------------------------

    if passed:

        print()
        print("RESULT: PASS")
        print("Candidate is better than the current best.")

        # Old best becomes archive.
        client.set_tag(
            best_run_id,
            "use",
            "archive",
        )

        # Candidate becomes new best.
        client.set_tag(
            candidate_run_id,
            "use",
            "best",
        )

        print()
        print(
            f"Updated old best {best_run_id}: "
            f"use=archive"
        )

        print(
            f"Updated candidate {candidate_run_id}: "
            f"use=best"
        )

        print()
        print("Promotion completed successfully.")

    # -------------------------------------------------------------------------
    # Promotion FAILED
    # -------------------------------------------------------------------------

    else:

        print()
        print("RESULT: FAIL")

        if not meaningful_improvements:

            print(
                "Candidate has no meaningful improvement "
                "over the current best."
            )

        if meaningful_regressions:

            print()
            print("MEANINGFUL REGRESSIONS:")

            for metric in meaningful_regressions:
                print(f"  - {metric}")

        print()
        print("Candidate will NOT be promoted.")

        # Remove candidate tag entirely.
        client.delete_tag(
            run_id=candidate_run_id,
            key="use",
        )

        print()
        print(
            f"Removed tag 'use' from candidate run "
            f"{candidate_run_id}"
        )

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------

    report = {
        "test": "promotion",
        "experiment": {
            "name": MLFLOW_EXPERIMENT_NAME,
            "experiment_id": experiment_id,
            "tracking_uri": MLFLOW_TRACKING_URI,
        },
        "runs": {
            "best": best_run_id,
            "candidate": candidate_run_id,
        },
        "promotion_metrics": sorted(PROMOTION_METRICS),
        "results": results,
        "summary": {
            "meaningful_improvements": len(
                meaningful_improvements
            ),
            "meaningful_regressions": len(
                meaningful_regressions
            ),
            "improvement_metrics": meaningful_improvements,
            "regression_metrics": meaningful_regressions,
            "passed": passed,
        },
    }

    PROMOTION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(PROMOTION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print(f"Report saved to:\n{PROMOTION_REPORT_PATH}")
    print()

    # Exit with non-zero status if promotion failed.
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()