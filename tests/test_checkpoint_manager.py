"""Tests for checkpoint manager — discovery, listing, pruning, and CLI."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from olytrain.checkpoint.manager import CHECKPOINT_EXTENSIONS, CheckpointInfo, CheckpointManager
from olytrain.cli import cli


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    """Create a temp directory with dummy checkpoint files."""
    base = tmp_path / "flat_project"
    base.mkdir()
    for name in ["model_epoch1.pth", "model_epoch2.pt", "best.ckpt"]:
        (base / name).write_bytes(b"\x00" * 1024)
        time.sleep(0.05)  # ensure distinct mtimes
    # A non-checkpoint file that should be ignored
    (base / "readme.txt").write_text("not a checkpoint")
    return base


@pytest.fixture()
def nested_ckpt_dir(tmp_path: Path) -> Path:
    """Create a nested directory structure with checkpoint files."""
    base = tmp_path / "nested_project"
    base.mkdir()
    subdir = base / "run1" / "checkpoints"
    subdir.mkdir(parents=True)
    (subdir / "epoch5.safetensors").write_bytes(b"\x00" * 2048)
    (base / "latest.pth").write_bytes(b"\x00" * 512)
    return base


class TestCheckpointInfo:
    def test_size_mb(self) -> None:
        info = CheckpointInfo(
            path=Path("/fake/model.pth"),
            size_bytes=10 * 1024 * 1024,
            modified=datetime.now(),
            project="test",
        )
        assert info.size_mb == pytest.approx(10.0)

    def test_age_days(self) -> None:
        from datetime import timedelta

        old_time = datetime.now() - timedelta(days=3)
        info = CheckpointInfo(
            path=Path("/fake/model.pth"),
            size_bytes=100,
            modified=old_time,
            project="test",
        )
        assert info.age_days == pytest.approx(3.0, abs=0.1)

    def test_default_optional_fields(self) -> None:
        info = CheckpointInfo(
            path=Path("/fake/model.pth"),
            size_bytes=100,
            modified=datetime.now(),
            project="test",
        )
        assert info.metric_value is None
        assert info.mlflow_run_id is None


class TestCheckpointDiscover:
    def test_discover_finds_checkpoint_files(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager()
        results = mgr.discover(ckpt_dir, "test")
        assert len(results) == 3
        extensions = {r.path.suffix for r in results}
        assert extensions <= CHECKPOINT_EXTENSIONS

    def test_discover_ignores_non_checkpoint_files(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager()
        results = mgr.discover(ckpt_dir, "test")
        paths = [r.path.name for r in results]
        assert "readme.txt" not in paths

    def test_discover_nonexistent_dir(self) -> None:
        mgr = CheckpointManager()
        results = mgr.discover(Path("/nonexistent/dir"), "test")
        assert results == []

    def test_discover_recursive(self, nested_ckpt_dir: Path) -> None:
        mgr = CheckpointManager()
        results = mgr.discover(nested_ckpt_dir, "test")
        assert len(results) == 2
        names = {r.path.name for r in results}
        assert "epoch5.safetensors" in names
        assert "latest.pth" in names

    def test_discover_sorted_by_modified_descending(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager()
        results = mgr.discover(ckpt_dir, "test")
        for i in range(len(results) - 1):
            assert results[i].modified >= results[i + 1].modified

    def test_discover_sets_project_name(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager()
        results = mgr.discover(ckpt_dir, "myproject")
        assert all(r.project == "myproject" for r in results)


class TestCheckpointList:
    def test_list_all_projects(self, ckpt_dir: Path, nested_ckpt_dir: Path) -> None:
        mgr = CheckpointManager(
            project_dirs={"proj_a": ckpt_dir, "proj_b": nested_ckpt_dir}
        )
        results = mgr.list()
        assert len(results) == 5

    def test_list_filtered_by_project(self, ckpt_dir: Path, nested_ckpt_dir: Path) -> None:
        mgr = CheckpointManager(
            project_dirs={"proj_a": ckpt_dir, "proj_b": nested_ckpt_dir}
        )
        results = mgr.list(project="proj_a")
        assert len(results) == 3
        assert all(r.project == "proj_a" for r in results)

    def test_list_nonexistent_project(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager(project_dirs={"proj_a": ckpt_dir})
        results = mgr.list(project="nonexistent")
        assert results == []


class TestCheckpointPrune:
    def test_prune_dry_run_does_not_delete(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager(project_dirs={"test": ckpt_dir})
        to_delete = mgr.prune("test", keep_best=1, dry_run=True)
        assert len(to_delete) == 2
        # Files should still exist
        assert all(c.path.exists() for c in to_delete)

    def test_prune_execute_deletes_files(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager(project_dirs={"test": ckpt_dir})
        to_delete = mgr.prune("test", keep_best=1, dry_run=False)
        assert len(to_delete) == 2
        # Deleted files should not exist
        assert not any(c.path.exists() for c in to_delete)
        # Kept file should still exist
        remaining = mgr.list(project="test")
        assert len(remaining) == 1

    def test_prune_keep_all(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager(project_dirs={"test": ckpt_dir})
        to_delete = mgr.prune("test", keep_best=10, dry_run=True)
        assert to_delete == []

    def test_prune_keep_zero(self, ckpt_dir: Path) -> None:
        mgr = CheckpointManager(project_dirs={"test": ckpt_dir})
        to_delete = mgr.prune("test", keep_best=0, dry_run=True)
        assert len(to_delete) == 3


class TestCheckpointCLI:
    def test_list_command_no_checkpoints(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "olytrain.checkpoint.manager.OLYPRO_MOVENET_DIR", tmp_path / "empty_mv"
        )
        monkeypatch.setattr(
            "olytrain.checkpoint.manager.OLYPRO_VISION_DIR", tmp_path / "empty_vis"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["checkpoint", "list"])
        assert result.exit_code == 0
        assert "No checkpoints found" in result.output

    def test_list_command_with_checkpoints(self, ckpt_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "olytrain.checkpoint.manager.OLYPRO_MOVENET_DIR", ckpt_dir
        )
        monkeypatch.setattr(
            "olytrain.checkpoint.manager.OLYPRO_VISION_DIR", ckpt_dir.parent / "empty"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["checkpoint", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Checkpoints" in result.output

    def test_prune_command_dry_run(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["checkpoint", "prune", "nonexistent"])
        assert result.exit_code == 0
        assert "Nothing to prune" in result.output

    def test_sync_command_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["checkpoint", "sync", "--help"])
        assert result.exit_code == 0
        assert "Sync checkpoints" in result.output
