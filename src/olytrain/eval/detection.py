"""Object detection evaluation — per-class AP breakdown and bbox overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image as PILImage
from rich.table import Table


def class_ap_chart(results: dict[str, float], class_names: list[str]) -> Figure:
    """Generate a horizontal bar chart of per-class AP.

    Args:
        results: dict mapping class_name -> AP value
        class_names: ordered list of class names
    """
    aps = [results.get(name, 0.0) for name in class_names]

    fig, ax = plt.subplots(figsize=(8, max(len(class_names) * 0.4, 3)))
    y_pos = range(len(class_names))
    colors = ["green" if ap >= 0.5 else "orange" if ap >= 0.3 else "red" for ap in aps]
    ax.barh(list(y_pos), aps, color=colors, alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(class_names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average Precision")
    ax.set_title("Per-Class AP")

    for i, v in enumerate(aps):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def detection_overlay(
    image: PILImage.Image | np.ndarray,
    pred_boxes: list[dict[str, Any]],
    gt_boxes: list[dict[str, Any]],
    class_names: list[str] | None = None,
) -> Figure:
    """Overlay predicted and ground-truth bounding boxes on an image.

    Args:
        image: PIL Image or numpy array
        pred_boxes: list of dicts with keys: bbox (x,y,w,h), class_id, score
        gt_boxes: list of dicts with keys: bbox (x,y,w,h), class_id
        class_names: optional mapping from class_id to name
    """
    if isinstance(image, PILImage.Image):
        image = np.array(image)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for box in gt_boxes:
        x, y, w, h = box["bbox"]
        rect = Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="green", facecolor="none"
        )
        ax.add_patch(rect)
        class_id: int = box["class_id"]
        label = (
            class_names[class_id]
            if class_names and class_id < len(class_names)
            else str(class_id)
        )
        ax.text(x, y - 2, f"GT: {label}", color="green", fontsize=8)

    for box in pred_boxes:
        x, y, w, h = box["bbox"]
        score: float = box.get("score", 0.0)
        rect = Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        class_id = box["class_id"]
        label = (
            class_names[class_id]
            if class_names and class_id < len(class_names)
            else str(class_id)
        )
        ax.text(x, y - 2, f"{label}: {score:.2f}", color="red", fontsize=8)

    ax.set_title("Detection Overlay")
    ax.axis("off")
    fig.tight_layout()
    return fig


def detection_report(
    eval_results: dict[str, float], class_names: list[str]
) -> Table:
    """Generate a Rich table of per-class detection results."""
    table = Table(title="Per-Class Detection Results")
    table.add_column("Class")
    table.add_column("AP", justify="right")
    table.add_column("Status")

    for name in class_names:
        ap = eval_results.get(name, 0.0)
        status = (
            "[green]Good[/green]"
            if ap >= 0.5
            else ("[yellow]Fair[/yellow]" if ap >= 0.3 else "[red]Poor[/red]")
        )
        table.add_row(name, f"{ap:.3f}", status)

    # Add mAP row
    all_aps = [eval_results.get(name, 0.0) for name in class_names]
    mean_ap = sum(all_aps) / len(all_aps) if all_aps else 0.0
    table.add_row("[bold]mAP[/bold]", f"[bold]{mean_ap:.3f}[/bold]", "")

    return table
