"""Pose estimation evaluation — keypoint accuracy heatmaps and overlays."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image as PILImage
from rich.table import Table


def keypoint_accuracy_heatmap(
    results: dict[str, float], keypoint_names: list[str]
) -> Figure:
    """Generate a heatmap of per-keypoint accuracy.

    Args:
        results: dict mapping keypoint_name -> accuracy (0-1)
        keypoint_names: ordered list of keypoint names

    Returns:
        matplotlib Figure
    """
    accuracies = [results.get(name, 0.0) for name in keypoint_names]
    data = np.array(accuracies).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(max(len(keypoint_names) * 0.8, 6), 3))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keypoint_names)))
    ax.set_xticklabels(keypoint_names, rotation=45, ha="right")
    ax.set_yticks([])

    for i, val in enumerate(accuracies):
        ax.text(i, 0, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.suptitle("Per-Keypoint Accuracy")
    fig.tight_layout()
    return fig


def prediction_overlay(
    image: PILImage.Image | np.ndarray,
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    skeleton: list[tuple[int, int]] | None = None,
) -> Figure:
    """Overlay predicted and ground-truth keypoints on an image.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        pred_keypoints: (N, 2) or (N, 3) array of predicted keypoints [x, y, (conf)]
        gt_keypoints: (N, 2) or (N, 3) array of ground-truth keypoints
        skeleton: list of (i, j) pairs defining connections between keypoints

    Returns:
        matplotlib Figure
    """
    if isinstance(image, PILImage.Image):
        image = np.array(image)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    # Draw skeleton connections
    if skeleton:
        for i, j in skeleton:
            if i < len(gt_keypoints) and j < len(gt_keypoints):
                ax.plot(
                    [gt_keypoints[i, 0], gt_keypoints[j, 0]],
                    [gt_keypoints[i, 1], gt_keypoints[j, 1]],
                    "g-",
                    alpha=0.5,
                    linewidth=1,
                )
            if i < len(pred_keypoints) and j < len(pred_keypoints):
                ax.plot(
                    [pred_keypoints[i, 0], pred_keypoints[j, 0]],
                    [pred_keypoints[i, 1], pred_keypoints[j, 1]],
                    "r-",
                    alpha=0.5,
                    linewidth=1,
                )

    # Draw keypoints
    ax.scatter(
        gt_keypoints[:, 0], gt_keypoints[:, 1], c="green", s=30, label="GT", zorder=5
    )
    ax.scatter(
        pred_keypoints[:, 0],
        pred_keypoints[:, 1],
        c="red",
        s=30,
        marker="x",
        label="Pred",
        zorder=5,
    )

    ax.legend()
    ax.set_title("Prediction Overlay")
    ax.axis("off")
    fig.tight_layout()
    return fig


def per_keypoint_report(
    eval_results: dict[str, float], keypoint_names: list[str]
) -> Table:
    """Generate a Rich table of per-keypoint evaluation results."""
    table = Table(title="Per-Keypoint Accuracy")
    table.add_column("Keypoint")
    table.add_column("Accuracy", justify="right")
    table.add_column("Status")

    for name in keypoint_names:
        acc = eval_results.get(name, 0.0)
        status = (
            "[green]Good[/green]"
            if acc >= 0.8
            else ("[yellow]Fair[/yellow]" if acc >= 0.5 else "[red]Poor[/red]")
        )
        table.add_row(name, f"{acc:.3f}", status)

    return table
