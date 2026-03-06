"""Tests for MLflow setup and initialization."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import mlflow

from olytrain.integrations.mlflow_setup import ensure_mlflow


def test_ensure_mlflow_creates_dirs_and_experiments():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "mlflow-test"
        db_path = tmp_path / "mlflow.db"
        artifact_root = tmp_path / "artifacts"
        tracking_uri = f"sqlite:///{db_path}"

        with (
            patch("olytrain.integrations.mlflow_setup.MLFLOW_DIR", tmp_path),
            patch("olytrain.integrations.mlflow_setup.ARTIFACT_ROOT", artifact_root),
            patch("olytrain.integrations.mlflow_setup.MLFLOW_TRACKING_URI", tracking_uri),
        ):
            ensure_mlflow()

            assert tmp_path.exists()
            assert artifact_root.exists()

            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiments = {e.name for e in client.search_experiments()}
            assert "movenet" in experiments
            assert "vision" in experiments


def test_ensure_mlflow_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "mlflow-test"
        db_path = tmp_path / "mlflow.db"
        artifact_root = tmp_path / "artifacts"
        tracking_uri = f"sqlite:///{db_path}"

        with (
            patch("olytrain.integrations.mlflow_setup.MLFLOW_DIR", tmp_path),
            patch("olytrain.integrations.mlflow_setup.ARTIFACT_ROOT", artifact_root),
            patch("olytrain.integrations.mlflow_setup.MLFLOW_TRACKING_URI", tracking_uri),
        ):
            ensure_mlflow()
            ensure_mlflow()  # Should not raise

            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            movenet_exps = [
                e for e in client.search_experiments() if e.name == "movenet"
            ]
            assert len(movenet_exps) == 1
