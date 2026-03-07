"""Tests for init, status commands, color-coded metrics, and checkpoint path defaults."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from olytrain.checkpoint.manager import CheckpointManager
from olytrain.cli import cli
from olytrain.cli.runs import _color_metric
from olytrain.config import CHECKPOINT_DIRS

# --- init command tests ---


def test_init_runs_without_error():
    """olytrain init should complete without crashing, even when projects are absent."""
    runner = CliRunner()
    with patch("olytrain.cli.init.OLYPRO_MOVENET_DIR", Path("/nonexistent/movenet")), patch(
        "olytrain.cli.init.OLYPRO_VISION_DIR", Path("/nonexistent/vision")
    ):
        result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "MLflow DB ready" in result.output
    assert "Not found" in result.output


def test_init_detects_existing_project(tmp_path: Path):
    """init should detect existing project dirs and sub-paths."""
    movenet = tmp_path / "olypro-movenet"
    movenet.mkdir()
    (movenet / "data").mkdir()
    (movenet / "output").mkdir()
    (movenet / "config.py").write_text("# config")

    runner = CliRunner()
    with patch("olytrain.cli.init.OLYPRO_MOVENET_DIR", movenet), patch(
        "olytrain.cli.init.OLYPRO_VISION_DIR", Path("/nonexistent/vision")
    ):
        result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "data/ directory" in result.output
    assert "output/ directory" in result.output
    assert "config.py" in result.output


def test_init_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0
    assert "Initialize olytrain" in result.output


# --- status command tests ---


def test_status_runs_without_error():
    """olytrain status should complete without crashing with empty MLflow DB."""
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Recent MLflow Runs" in result.output
    assert "Checkpoint Disk Usage" in result.output
    assert "Dataset Info" in result.output


def test_status_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--help"])
    assert result.exit_code == 0
    assert "Show overview" in result.output


# --- _color_metric tests ---


def test_color_metric_loss_green():
    result = _color_metric("val/loss", 0.1)
    assert "[green]" in result
    assert "0.1000" in result


def test_color_metric_loss_yellow():
    result = _color_metric("train_loss", 0.5)
    assert "[yellow]" in result
    assert "0.5000" in result


def test_color_metric_loss_red():
    result = _color_metric("loss", 0.9)
    assert "[red]" in result
    assert "0.9000" in result


def test_color_metric_accuracy_green():
    result = _color_metric("val/acc", 0.95)
    assert "[green]" in result


def test_color_metric_accuracy_yellow():
    result = _color_metric("pck@0.5", 0.6)
    assert "[yellow]" in result


def test_color_metric_accuracy_red():
    result = _color_metric("acc", 0.2)
    assert "[red]" in result


def test_color_metric_ap_green():
    result = _color_metric("AP50", 0.7)
    assert "[green]" in result


def test_color_metric_ap_yellow():
    result = _color_metric("AP75", 0.35)
    assert "[yellow]" in result


def test_color_metric_ap_red():
    result = _color_metric("AP", 0.1)
    assert "[red]" in result


def test_color_metric_map():
    result = _color_metric("mAP", 0.6)
    assert "[green]" in result


def test_color_metric_unknown():
    result = _color_metric("lr", 0.001)
    assert "0.0010" in result
    assert "[green]" not in result
    assert "[red]" not in result


# --- checkpoint manager default paths ---


def test_checkpoint_manager_uses_checkpoint_dirs():
    """Default CheckpointManager should use CHECKPOINT_DIRS (subdirs, not project roots)."""
    mgr = CheckpointManager()
    assert mgr.project_dirs == CHECKPOINT_DIRS
    assert "output" in str(mgr.project_dirs["movenet"])
    assert "models" in str(mgr.project_dirs["vision"])


def test_checkpoint_manager_custom_dirs(tmp_path: Path):
    """CheckpointManager should accept custom dirs override."""
    custom = {"test": tmp_path}
    mgr = CheckpointManager(project_dirs=custom)
    assert mgr.project_dirs == custom


# --- CLI help includes new commands ---


def test_all_subcommands_visible_in_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    for cmd in ["dashboard", "runs", "dataset", "checkpoint", "eval", "config", "init", "status"]:
        assert cmd in result.output, f"'{cmd}' not found in CLI help"
