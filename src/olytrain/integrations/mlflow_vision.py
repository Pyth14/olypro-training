"""TensorBoard event parser for TF-Vision -> MLflow ingestion."""

from __future__ import annotations

import time

import mlflow
import yaml
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,  # type: ignore[import-untyped]
)

from olytrain.integrations.mlflow_setup import ensure_mlflow

TRACKED_SCALARS = [
    "total_loss",
    "cls_loss",
    "box_loss",
    "regularization_loss",
    "AP",
    "AP50",
    "AP75",
]


def _normalize_tag(tag: str) -> str:
    """Strip common prefixes like 'train/' or 'eval/' from a tag name."""
    parts = tag.split("/")
    if len(parts) > 1:
        return parts[-1]
    return tag


def _match_tracked(tag: str) -> str | None:
    """Return the tracked scalar name if tag contains it, else None."""
    normalized = _normalize_tag(tag)
    for scalar in TRACKED_SCALARS:
        if normalized == scalar:
            return scalar
    return None


class TBEventParser:
    """Parse TensorBoard event files and log metrics to MLflow."""

    def __init__(self, event_dir: str, experiment_name: str = "vision"):
        self.event_dir = event_dir
        self.experiment_name = experiment_name

    def parse_and_log(
        self, run_name: str | None = None, config_path: str | None = None
    ) -> None:
        """One-shot parse of TensorBoard events into MLflow."""
        ensure_mlflow()
        mlflow.set_experiment(self.experiment_name)

        ea = EventAccumulator(self.event_dir)
        ea.Reload()

        available_tags = ea.Tags().get("scalars", [])

        with mlflow.start_run(run_name=run_name):
            for tag in available_tags:
                metric_name = _match_tracked(tag)
                if metric_name is None:
                    continue
                for event in ea.Scalars(tag):
                    mlflow.log_metric(metric_name, event.value, step=event.step)

            if config_path is not None:
                mlflow.log_artifact(config_path)
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                if isinstance(config_data, dict):
                    flat = _flatten_dict(config_data)
                    mlflow.log_params(flat)

    def watch(self, interval: int = 30) -> None:
        """Polling mode -- re-parse events every ``interval`` seconds."""
        try:
            while True:
                self.parse_and_log()
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


def _flatten_dict(d: dict[str, object], prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict into dot-separated keys for MLflow params."""
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = str(v)
    return items
