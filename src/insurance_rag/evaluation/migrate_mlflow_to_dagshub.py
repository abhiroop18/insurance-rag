from pathlib import Path
import os
import mlflow


# =============================================================================
# CONFIGURATION
# =============================================================================

LOCAL_MLFLOW_URI = "http://127.0.0.1:5000"

DAGSHUB_MLFLOW_URI = (
    "https://dagshub.com/abhiroop18/insurance-rag.mlflow"
)

EXPERIMENT_NAME = "insurance-rag-experiments"

SOURCE_EXPERIMENT_ID = "3"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Temporary directory used for downloading artifacts from local MLflow.
ARTIFACT_DOWNLOAD_DIR = (
    PROJECT_ROOT / "data" / "mlflow_migration"
)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_environment():
    """Validate DagsHub credentials."""

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not username:
        raise EnvironmentError(
            "MLFLOW_TRACKING_USERNAME is not set."
        )

    if not password:
        raise EnvironmentError(
            "MLFLOW_TRACKING_PASSWORD is not set."
        )


# =============================================================================
# SOURCE MLflow
# =============================================================================

def get_source_client():
    """Create MLflow client connected to the local server."""

    return mlflow.MlflowClient(
        tracking_uri=LOCAL_MLFLOW_URI
    )


# =============================================================================
# DESTINATION MLflow
# =============================================================================

def get_destination_client():
    """Create MLflow client connected to DagsHub."""

    return mlflow.MlflowClient(
        tracking_uri=DAGSHUB_MLFLOW_URI
    )


# =============================================================================
# DESTINATION EXPERIMENT
# =============================================================================

def get_or_create_destination_experiment(client):
    """Get or create the experiment on DagsHub."""

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is not None:
        return experiment

    print(
        f"Creating experiment '{EXPERIMENT_NAME}' on DagsHub..."
    )

    experiment_id = client.create_experiment(
        EXPERIMENT_NAME
    )

    return client.get_experiment(
        experiment_id
    )


# =============================================================================
# CHECK WHETHER RUN WAS ALREADY MIGRATED
# =============================================================================

def already_migrated(client, experiment_id, source_run_id):
    """
    Check whether a local MLflow run has already been migrated.

    We identify migrated runs using:
        source_mlflow_run_id=<local run id>
    """

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.source_mlflow_run_id = '{source_run_id}'"
        ),
    )

    return len(runs) > 0


# =============================================================================
# COPY ARTIFACTS
# =============================================================================

def migrate_artifacts(
    source_client,
    destination_client,
    source_run,
    destination_run_id,
):
    """
    Download artifacts from local MLflow and upload them to DagsHub.
    """

    source_run_id = source_run.info.run_id

    artifact_root = (
        ARTIFACT_DOWNLOAD_DIR / source_run_id
    )

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        print(
            f"  Downloading artifacts for {source_run_id}..."
        )

        downloaded_path = source_client.download_artifacts(
            source_run_id,
            "",
            dst_path=str(artifact_root),
        )

    except Exception as exc:

        print(
            f"  WARNING: Could not download artifacts: {exc}"
        )

        return

    downloaded_path = Path(downloaded_path)

    if not downloaded_path.exists():
        return

    # Recursively upload files.
    for file_path in downloaded_path.rglob("*"):

        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(
            downloaded_path
        )

        artifact_path = str(
            relative_path.parent
        )

        if artifact_path == ".":
            artifact_path = None

        try:

            destination_client.log_artifact(
                destination_run_id,
                str(file_path),
                artifact_path=artifact_path,
            )

        except Exception as exc:

            print(
                f"  WARNING: Could not upload artifact "
                f"{relative_path}: {exc}"
            )


# =============================================================================
# MIGRATE ONE RUN
# =============================================================================

def migrate_run(
    source_client,
    destination_client,
    destination_experiment_id,
    source_run,
):
    """Copy one MLflow run from local MLflow to DagsHub."""

    source_run_id = source_run.info.run_id

    # -------------------------------------------------------------------------
    # Skip if already migrated
    # -------------------------------------------------------------------------

    if already_migrated(
        destination_client,
        destination_experiment_id,
        source_run_id,
    ):

        print(
            f"SKIP {source_run_id} "
            f"(already migrated)"
        )

        return None

    # -------------------------------------------------------------------------
    # Run information
    # -------------------------------------------------------------------------

    run_name = source_run.data.tags.get(
        "mlflow.runName"
    )

    tags = dict(source_run.data.tags)

    # Keep a reference to the original local run.
    tags["source_mlflow_run_id"] = source_run_id
    tags["source_mlflow_tracking_uri"] = LOCAL_MLFLOW_URI

    # -------------------------------------------------------------------------
    # Create destination run
    # -------------------------------------------------------------------------

    print()
    print(
        f"Migrating run: {source_run_id}"
    )

    if run_name:
        print(
            f"  Run name: {run_name}"
        )

    destination_run = destination_client.create_run(
        experiment_id=destination_experiment_id,
        start_time=source_run.info.start_time,
        tags=tags,
    )

    destination_run_id = destination_run.info.run_id

    print(
        f"  DagsHub run: {destination_run_id}"
    )

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    if source_run.data.params:

        destination_client.log_batch(
            destination_run_id,
            metrics=[],
            params=[
                mlflow.entities.Param(
                    key=key,
                    value=value,
                )
                for key, value in source_run.data.params.items()
            ],
            tags=[],
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    for metric_name, metric_value in source_run.data.metrics.items():

        destination_client.log_metric(
            destination_run_id,
            metric_name,
            metric_value,
        )

    # -------------------------------------------------------------------------
    # Artifacts
    # -------------------------------------------------------------------------

    migrate_artifacts(
        source_client=source_client,
        destination_client=destination_client,
        source_run=source_run,
        destination_run_id=destination_run_id,
    )

    # -------------------------------------------------------------------------
    # Preserve original run status
    # -------------------------------------------------------------------------

    destination_status = source_run.info.status

    try:

        destination_client.set_terminated(
            destination_run_id,
            status=destination_status,
            end_time=source_run.info.end_time,
        )

    except Exception as exc:

        print(
            f"  WARNING: Could not preserve run status: {exc}"
        )

        # At minimum terminate the run.
        destination_client.set_terminated(
            destination_run_id
        )

    print(
        f"  SUCCESS: {source_run_id} -> "
        f"{destination_run_id}"
    )

    return destination_run_id


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("MLFLOW → DAGSHUB MIGRATION")
    print("=" * 80)

    validate_environment()

    # -------------------------------------------------------------------------
    # Connect to local MLflow
    # -------------------------------------------------------------------------

    print()
    print(
        f"Source      : {LOCAL_MLFLOW_URI}"
    )

    source_client = get_source_client()

    source_experiment = source_client.get_experiment(
        SOURCE_EXPERIMENT_ID
    )

    if source_experiment is None:
        raise ValueError(
            f"Local MLflow experiment "
            f"{SOURCE_EXPERIMENT_ID} not found."
        )

    print(
        f"Experiment  : {source_experiment.name}"
    )

    # -------------------------------------------------------------------------
    # Connect to DagsHub
    # -------------------------------------------------------------------------

    print(
        f"Destination : {DAGSHUB_MLFLOW_URI}"
    )

    destination_client = get_destination_client()

    destination_experiment = (
        get_or_create_destination_experiment(
            destination_client
        )
    )

    destination_experiment_id = (
        destination_experiment.experiment_id
    )

    print(
        f"DagsHub experiment ID: "
        f"{destination_experiment_id}"
    )

    # -------------------------------------------------------------------------
    # Get all source runs
    # -------------------------------------------------------------------------

    source_runs = source_client.search_runs(
        experiment_ids=[SOURCE_EXPERIMENT_ID],
        order_by=["attributes.start_time ASC"],
        max_results=5000,
    )

    print()
    print(
        f"Found {len(source_runs)} local MLflow runs."
    )

    # -------------------------------------------------------------------------
    # Migrate
    # -------------------------------------------------------------------------

    migrated_count = 0
    skipped_count = 0
    failed_count = 0

    for source_run in source_runs:

        try:

            result = migrate_run(
                source_client=source_client,
                destination_client=destination_client,
                destination_experiment_id=(
                    destination_experiment_id
                ),
                source_run=source_run,
            )

            if result is None:
                skipped_count += 1
            else:
                migrated_count += 1

        except Exception as exc:

            failed_count += 1

            print()
            print(
                f"ERROR migrating run "
                f"{source_run.info.run_id}:"
            )
            print(f"  {exc}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MIGRATION COMPLETE")
    print("=" * 80)

    print(
        f"Total source runs : {len(source_runs)}"
    )

    print(
        f"Migrated           : {migrated_count}"
    )

    print(
        f"Skipped            : {skipped_count}"
    )

    print(
        f"Failed             : {failed_count}"
    )

    print()

    if failed_count > 0:

        print(
            "WARNING: Some runs failed to migrate."
        )

        raise SystemExit(1)

    print(
        "All runs migrated successfully."
    )


if __name__ == "__main__":
    main()