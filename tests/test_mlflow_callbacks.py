"""Tests for MLflow callback classes."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import mlflow

from olytrain.integrations.mlflow_task import MLflowTaskCallback, flatten_config
from olytrain.integrations.mlflow_rtmpose import MLflowRTMPoseCallback


class TestFlattenConfig:
    def test_flat_dict(self) -> None:
        assert flatten_config({"lr": 0.001, "epochs": 10}) == {"lr": 0.001, "epochs": 10}

    def test_nested_dict(self) -> None:
        result = flatten_config({"model": {"backbone": "resnet", "layers": 50}})
        assert result == {"model/backbone": "resnet", "model/layers": 50}

    def test_deeply_nested(self) -> None:
        result = flatten_config({"a": {"b": {"c": 1}}})
        assert result == {"a/b/c": 1}

    def test_empty_dict(self) -> None:
        assert flatten_config({}) == {}

    def test_mixed_nesting(self) -> None:
        result = flatten_config({"lr": 0.01, "model": {"name": "resnet"}, "seed": 42})
        assert result == {"lr": 0.01, "model/name": "resnet", "seed": 42}


def _make_mlflow_patches(tmp_path: Path, tracking_uri: str):
    """Return context managers that patch mlflow_setup and mlflow_task/rtmpose config."""
    return (
        patch("olytrain.integrations.mlflow_setup.MLFLOW_DIR", tmp_path),
        patch("olytrain.integrations.mlflow_setup.ARTIFACT_ROOT", tmp_path / "artifacts"),
        patch("olytrain.integrations.mlflow_setup.MLFLOW_TRACKING_URI", tracking_uri),
    )


class TestMLflowTaskCallback:
    def test_full_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "mlflow-test"
            tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
            p1, p2, p3 = _make_mlflow_patches(tmp_path, tracking_uri)

            with p1, p2, p3:
                config = {"lr": 0.001, "model": {"backbone": "resnet"}}
                cb = MLflowTaskCallback(
                    config=config,
                    experiment_name="movenet",
                    run_name="test-run",
                    tags={"version": "1"},
                )

                cb.on_train_start()

                # Verify run is active
                run = mlflow.active_run()
                assert run is not None
                assert run.info.run_name == "test-run"

                cb.on_batch_end(step=1, metrics={"loss": 0.5})
                cb.on_batch_end(step=2, metrics={"loss": 0.4})
                cb.on_epoch_end(epoch=1, metrics={"val_loss": 0.3})

                # Create a dummy checkpoint file for artifact logging
                ckpt_file = Path(tmpdir) / "checkpoint.pt"
                ckpt_file.write_text("dummy")
                cb.on_checkpoint(path=str(ckpt_file), metric_value=0.95)

                cb.on_train_end()

                # Verify metrics were logged
                client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                run_data = client.get_run(run.info.run_id)
                assert run_data.data.params["lr"] == "0.001"
                assert run_data.data.params["model/backbone"] == "resnet"
                assert run_data.data.tags["version"] == "1"
                assert float(run_data.data.metrics["loss"]) == 0.4
                assert float(run_data.data.metrics["val_loss"]) == 0.3
                assert float(run_data.data.metrics["best_metric"]) == 0.95

                # Verify artifact was logged
                artifacts = client.list_artifacts(run.info.run_id)
                artifact_names = [a.path for a in artifacts]
                assert "checkpoint.pt" in artifact_names

    def test_ddp_rank_nonzero_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "mlflow-test"
            tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
            p1, p2, p3 = _make_mlflow_patches(tmp_path, tracking_uri)

            with p1, p2, p3, patch.dict(os.environ, {"LOCAL_RANK": "1"}):
                config = {"lr": 0.001}
                cb = MLflowTaskCallback(config=config)
                cb.on_train_start()

                assert not cb._active

                # These should all be no-ops and not raise
                cb.on_batch_end(step=1, metrics={"loss": 0.5})
                cb.on_epoch_end(epoch=1, metrics={"val_loss": 0.3})
                cb.on_checkpoint(path="/fake/path")
                cb.on_train_end()

                # No MLflow run should be active
                assert mlflow.active_run() is None


class TestMLflowRTMPoseCallback:
    def test_full_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "mlflow-test"
            tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
            p1, p2, p3 = _make_mlflow_patches(tmp_path, tracking_uri)

            with p1, p2, p3:
                config = {"lr": 0.002, "model": {"type": "rtmpose"}}
                cb = MLflowRTMPoseCallback(
                    config=config,
                    run_name="rtmpose-test",
                )

                # Verify default experiment name
                assert cb._experiment_name == "movenet-rtmpose"

                cb.on_train_start()

                run = mlflow.active_run()
                assert run is not None

                cb.on_batch_end(step=1, metrics={"loss": 0.6})
                cb.on_epoch_end(epoch=1, metrics={"ap": 0.75})

                ckpt_file = Path(tmpdir) / "rtmpose_ckpt.pt"
                ckpt_file.write_text("dummy rtmpose checkpoint")
                cb.on_checkpoint(path=str(ckpt_file), metric_value=0.80)

                cb.on_train_end()

                client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                run_data = client.get_run(run.info.run_id)
                assert run_data.data.params["model/type"] == "rtmpose"
                assert float(run_data.data.metrics["ap"]) == 0.75
                assert float(run_data.data.metrics["best_metric"]) == 0.80

    def test_ddp_rank_nonzero_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "mlflow-test"
            tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
            p1, p2, p3 = _make_mlflow_patches(tmp_path, tracking_uri)

            with p1, p2, p3, patch.dict(os.environ, {"LOCAL_RANK": "1"}):
                cb = MLflowRTMPoseCallback(config={"lr": 0.001})
                cb.on_train_start()

                assert not cb._active
                cb.on_batch_end(step=1, metrics={"loss": 0.5})
                cb.on_epoch_end(epoch=1, metrics={"val_loss": 0.3})
                cb.on_checkpoint(path="/fake/path")
                cb.on_train_end()
                assert mlflow.active_run() is None
