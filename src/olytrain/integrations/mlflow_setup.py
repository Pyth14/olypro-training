"""MLflow server configuration and initialization."""

import mlflow

from olytrain.config import ARTIFACT_ROOT, DEFAULT_EXPERIMENTS, MLFLOW_DIR, MLFLOW_TRACKING_URI


def ensure_mlflow() -> None:
    """Ensure MLflow is configured with SQLite backend and default experiments exist."""
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    for experiment_name in DEFAULT_EXPERIMENTS:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            client.create_experiment(
                experiment_name,
                artifact_location=str(ARTIFACT_ROOT / experiment_name),
            )
