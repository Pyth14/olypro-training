"""Tests for runs list and compare commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import mlflow
from click.testing import CliRunner

from olytrain.cli import cli


def test_runs_list_no_runs():
    """Test runs list with no runs in a temp MLflow DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        artifact_root = tmp_path / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)

        with (
            patch("olytrain.integrations.mlflow_setup.MLFLOW_DIR", tmp_path),
            patch("olytrain.integrations.mlflow_setup.ARTIFACT_ROOT", artifact_root),
            patch("olytrain.integrations.mlflow_setup.MLFLOW_TRACKING_URI", tracking_uri),
        ):
            mlflow.set_tracking_uri(tracking_uri)

            runner = CliRunner()
            result = runner.invoke(cli, ["runs", "list"])
            assert result.exit_code == 0
            assert "No runs found" in result.output or "No experiments found" in result.output


def test_runs_compare_with_real_runs():
    """Test runs compare with two real MLflow runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        artifact_root = tmp_path / "artifacts"
        tmp_path.mkdir(exist_ok=True)
        artifact_root.mkdir(exist_ok=True)

        with (
            patch("olytrain.integrations.mlflow_setup.MLFLOW_DIR", tmp_path),
            patch("olytrain.integrations.mlflow_setup.ARTIFACT_ROOT", artifact_root),
            patch("olytrain.integrations.mlflow_setup.MLFLOW_TRACKING_URI", tracking_uri),
        ):
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("test-compare")

            with mlflow.start_run() as run1:
                mlflow.log_param("lr", "0.001")
                mlflow.log_metric("loss", 0.5)
            run1_id = run1.info.run_id

            with mlflow.start_run() as run2:
                mlflow.log_param("lr", "0.01")
                mlflow.log_metric("loss", 0.3)
            run2_id = run2.info.run_id

            runner = CliRunner()
            result = runner.invoke(cli, ["runs", "compare", run1_id, run2_id])
            assert result.exit_code == 0
            assert "Parameters" in result.output
            assert "Metrics" in result.output
