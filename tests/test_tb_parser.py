"""Tests for TBEventParser and CLI ingest command."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
from click.testing import CliRunner
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from olytrain.cli import cli
from olytrain.integrations.mlflow_vision import TBEventParser, _normalize_tag


def _make_scalar_event(tag: str, value: float, step: int, wall_time: float = 1.0) -> Event:
    """Create a TensorBoard scalar event."""
    summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
    return Event(wall_time=wall_time, step=step, summary=summary)


def _write_events_to_dir(event_dir: str, events: list[Event]) -> None:
    """Write events to a TensorBoard event file using EventFileWriter."""
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(event_dir)
    for event in events:
        writer.add_event(event)
    writer.flush()
    writer.close()


def _setup_mlflow(tmpdir: Path) -> str:
    """Create a temp MLflow tracking URI and configure mlflow."""
    db_path = tmpdir / "mlflow.db"
    artifact_root = tmpdir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def test_normalize_tag_strips_prefix() -> None:
    assert _normalize_tag("train/total_loss") == "total_loss"
    assert _normalize_tag("eval/AP50") == "AP50"
    assert _normalize_tag("total_loss") == "total_loss"
    assert _normalize_tag("some/deep/prefix/cls_loss") == "cls_loss"


def test_parse_and_log_basic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        event_dir = str(tmp_path / "events")
        Path(event_dir).mkdir()

        events = [
            _make_scalar_event("train/total_loss", 2.5, step=0),
            _make_scalar_event("train/total_loss", 1.5, step=1),
            _make_scalar_event("eval/AP", 0.35, step=1),
            _make_scalar_event("untracked_metric", 99.0, step=0),
        ]
        _write_events_to_dir(event_dir, events)

        tracking_uri = _setup_mlflow(tmp_path / "mlflow")

        with patch("olytrain.integrations.mlflow_vision.ensure_mlflow"):
            mlflow.set_experiment("vision")
            parser = TBEventParser(event_dir, experiment_name="vision")
            parser.parse_and_log(run_name="test-run")

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiments = client.get_experiment_by_name("vision")
        assert experiments is not None

        runs = client.search_runs([experiments.experiment_id])
        assert len(runs) == 1
        run = runs[0]

        total_loss_history = client.get_metric_history(run.info.run_id, "total_loss")
        assert len(total_loss_history) == 2
        assert total_loss_history[0].value == pytest.approx(2.5, abs=1e-5)
        assert total_loss_history[1].value == pytest.approx(1.5, abs=1e-5)

        ap_history = client.get_metric_history(run.info.run_id, "AP")
        assert len(ap_history) == 1
        assert ap_history[0].value == pytest.approx(0.35, abs=1e-5)

        # untracked_metric should NOT be logged
        untracked = client.get_metric_history(run.info.run_id, "untracked_metric")
        assert len(untracked) == 0


def test_parse_and_log_with_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        event_dir = str(tmp_path / "events")
        Path(event_dir).mkdir()

        events = [_make_scalar_event("total_loss", 1.0, step=0)]
        _write_events_to_dir(event_dir, events)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  name: retinanet\n  backbone: resnet50\nlr: 0.01\n")

        tracking_uri = _setup_mlflow(tmp_path / "mlflow")

        with patch("olytrain.integrations.mlflow_vision.ensure_mlflow"):
            mlflow.set_experiment("vision")
            parser = TBEventParser(event_dir)
            parser.parse_and_log(config_path=str(config_file))

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiments = client.get_experiment_by_name("vision")
        runs = client.search_runs([experiments.experiment_id])
        assert len(runs) == 1
        run = runs[0]

        assert run.data.params["model.name"] == "retinanet"
        assert run.data.params["model.backbone"] == "resnet50"
        assert run.data.params["lr"] == "0.01"


def test_parse_and_log_prefixed_tags() -> None:
    """Tags with prefixes like eval/ or train/ should be normalized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        event_dir = str(tmp_path / "events")
        Path(event_dir).mkdir()

        events = [
            _make_scalar_event("eval/AP50", 0.6, step=5),
            _make_scalar_event("eval/AP75", 0.4, step=5),
            _make_scalar_event("train/cls_loss", 0.8, step=5),
        ]
        _write_events_to_dir(event_dir, events)

        tracking_uri = _setup_mlflow(tmp_path / "mlflow")

        with patch("olytrain.integrations.mlflow_vision.ensure_mlflow"):
            mlflow.set_experiment("vision")
            parser = TBEventParser(event_dir)
            parser.parse_and_log()

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiments = client.get_experiment_by_name("vision")
        runs = client.search_runs([experiments.experiment_id])
        run = runs[0]

        ap50 = client.get_metric_history(run.info.run_id, "AP50")
        assert len(ap50) == 1
        assert ap50[0].value == pytest.approx(0.6, abs=1e-5)

        ap75 = client.get_metric_history(run.info.run_id, "AP75")
        assert len(ap75) == 1

        cls_loss = client.get_metric_history(run.info.run_id, "cls_loss")
        assert len(cls_loss) == 1
        assert cls_loss[0].value == pytest.approx(0.8, abs=1e-5)


def test_cli_ingest_command() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        event_dir = str(tmp_path / "events")
        Path(event_dir).mkdir()

        events = [_make_scalar_event("total_loss", 1.0, step=0)]
        _write_events_to_dir(event_dir, events)

        _setup_mlflow(tmp_path / "mlflow")

        runner = CliRunner()
        with patch("olytrain.integrations.mlflow_vision.ensure_mlflow"):
            mlflow.set_experiment("vision")
            result = runner.invoke(cli, ["runs", "ingest", event_dir])

        assert result.exit_code == 0, result.output
        assert "Ingested events from" in result.output
