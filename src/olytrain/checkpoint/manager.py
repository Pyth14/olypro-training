"""Checkpoint discovery, metadata, and pruning."""

from __future__ import annotations

import builtins
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from olytrain.config import OLYPRO_MOVENET_DIR, OLYPRO_VISION_DIR

CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".ckpt", ".index", ".h5", ".keras", ".safetensors"}


@dataclass
class CheckpointInfo:
    path: Path
    size_bytes: int
    modified: datetime
    project: str
    metric_value: float | None = None
    mlflow_run_id: str | None = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def age_days(self) -> float:
        return (datetime.now() - self.modified).total_seconds() / 86400


class CheckpointManager:
    """Discover, list, and prune checkpoints across projects."""

    def __init__(self, project_dirs: dict[str, Path] | None = None) -> None:
        self.project_dirs = project_dirs or {
            "movenet": OLYPRO_MOVENET_DIR,
            "vision": OLYPRO_VISION_DIR,
        }

    def discover(self, project_dir: Path, project_name: str = "unknown") -> list[CheckpointInfo]:
        """Scan a directory recursively for checkpoint files."""
        checkpoints: list[CheckpointInfo] = []
        if not project_dir.exists():
            return checkpoints
        for root, _dirs, files in os.walk(project_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix in CHECKPOINT_EXTENSIONS:
                    stat = p.stat()
                    checkpoints.append(
                        CheckpointInfo(
                            path=p,
                            size_bytes=stat.st_size,
                            modified=datetime.fromtimestamp(stat.st_mtime),
                            project=project_name,
                        )
                    )
        return sorted(checkpoints, key=lambda c: c.modified, reverse=True)

    def list(self, project: str | None = None) -> list[CheckpointInfo]:
        """List checkpoints, optionally filtered by project."""
        results: list[CheckpointInfo] = []
        for name, dir_path in self.project_dirs.items():
            if project and name != project:
                continue
            results.extend(self.discover(dir_path, name))
        return sorted(results, key=lambda c: c.modified, reverse=True)

    def prune(
        self,
        project: str,
        keep_best: int = 5,
        metric: str = "val/acc",
        dry_run: bool = True,
    ) -> builtins.list[CheckpointInfo]:
        """Identify checkpoints to delete, keeping the N most recent.

        Returns list of checkpoints that would be (or were) deleted.
        If dry_run=False, actually deletes them.
        """
        all_ckpts = self.list(project=project)
        to_delete = all_ckpts[keep_best:]
        if not dry_run:
            for ckpt in to_delete:
                ckpt.path.unlink(missing_ok=True)
        return to_delete

    def link_to_mlflow(self, checkpoint_path: str, run_id: str) -> None:
        """Associate a checkpoint with an MLflow run by logging it as artifact."""
        import mlflow

        from olytrain.integrations.mlflow_setup import ensure_mlflow

        ensure_mlflow()
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(checkpoint_path)
