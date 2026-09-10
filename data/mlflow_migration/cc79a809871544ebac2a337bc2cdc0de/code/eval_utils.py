import json
from pathlib import Path


# =============================================================================
# FIND LATEST FILE
# =============================================================================

def get_latest_file(
    directory: Path,
    pattern: str,
) -> Path:

    files = list(
        directory.rglob(pattern)
    )

    if not files:

        raise FileNotFoundError(
            f"No files found in {directory}"
        )

    return max(
        files,
        key=lambda file: file.stat().st_mtime,
    )


# =============================================================================
# GET LATEST RESULTS
# =============================================================================

def get_latest_results(
    results_dir: Path,
) -> Path:

    return get_latest_file(
        directory=results_dir,
        pattern="*.json",
    )


# =============================================================================
# GET LATEST REPORT
# =============================================================================

def get_latest_report(
    reports_dir: Path,
) -> Path:

    return get_latest_file(
        directory=reports_dir,
        pattern="*.md",
    )


# =============================================================================
# GET METRICS
# =============================================================================

def get_metrics_from_results(
    results_path: Path,
) -> dict:

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as f:

        results = json.load(f)

    metrics = {}

    for test_result in results.get(
        "testCases",
        [],
    ):

        for metric in test_result.get(
            "metricsData",
            [],
        ):

            name = metric.get("name")
            score = metric.get("score")

            if (
                name is None
                or score is None
            ):
                continue

            metrics.setdefault(
                name,
                [],
            ).append(
                float(score)
            )

    return {
        metric_name: (
            sum(scores) / len(scores)
        )
        for metric_name, scores
        in metrics.items()
        if scores
    }


# =============================================================================
# GET TEST COUNT
# =============================================================================

def get_test_count(
    results_path: Path,
) -> int:

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as f:

        results = json.load(f)

    return len(
        results.get(
            "testCases",
            [],
        )
    )