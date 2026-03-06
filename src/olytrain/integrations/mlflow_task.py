"""MLflow callback for movenet Task-based training."""

from __future__ import annotations

import os
from typing import Any

import mlflow

from olytrain.integrations.mlflow_setup import ensure_mlflow


def flatten_config(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts using '/' as separator.

    Example: {"model": {"backbone": "resnet"}} -> {"model/backbone": "resnet"}
    """
    items: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            items.update(flatten_config(value, new_key))
        else:
            items[new_key] = value
    return items


class MLflowTaskCallback:
    """Drop-in callback for movenet Task-based training loops."""

    def __init__(
        self,
        config: dict[str, Any],
        experiment_name: str = "movenet",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._config = config
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tags = tags
        self._active = True

    def on_train_start(self) -> None:
        """Start an MLflow run. No-op if not rank 0 in DDP."""
        if os.environ.get("LOCAL_RANK", "0") != "0":
            self._active = False
            return
        ensure_mlflow()
        mlflow.set_experiment(self._experiment_name)
        mlflow.start_run(run_name=self._run_name, tags=self._tags)
        mlflow.log_params(flatten_config(self._config))

    def on_batch_end(self, step: int, metrics: dict[str, float]) -> None:
        """Log batch-level metrics."""
        if not self._active:
            return
        mlflow.log_metrics(metrics, step=step)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch-level metrics."""
        if not self._active:
            return
        mlflow.log_metrics(metrics, step=epoch)

    def on_checkpoint(self, path: str, metric_value: float | None = None) -> None:
        """Log a checkpoint artifact."""
        if not self._active:
            return
        mlflow.log_artifact(path)
        if metric_value is not None:
            mlflow.log_metric("best_metric", metric_value)

    def on_train_end(self) -> None:
        """End the MLflow run."""
        if not self._active:
            return
        mlflow.end_run()
