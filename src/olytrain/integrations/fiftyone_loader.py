"""Dataset loading utilities for FiftyOne."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import fiftyone as fo

    HAS_FIFTYONE = True
except ImportError:
    fo = None  # type: ignore[assignment]
    HAS_FIFTYONE = False


def _require_fiftyone() -> None:
    if not HAS_FIFTYONE:
        raise ImportError(
            "fiftyone is required for dataset inspection. "
            "Install with: pip install olytrain[fiftyone]"
        )


def load_coco_keypoints(
    annotation_json: str, image_dir: str, name: str | None = None
) -> Any:
    """Load a COCO keypoints dataset into FiftyOne."""
    _require_fiftyone()
    dataset_name = name or Path(annotation_json).stem
    return fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_dir,
        labels_path=annotation_json,
        name=dataset_name,
    )


def load_coco_detection(
    annotation_json: str, image_dir: str, name: str | None = None
) -> Any:
    """Load a COCO detection dataset into FiftyOne."""
    _require_fiftyone()
    dataset_name = name or Path(annotation_json).stem
    return fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_dir,
        labels_path=annotation_json,
        name=dataset_name,
    )


def load_voc_detection(image_dir: str, name: str | None = None) -> Any:
    """Load a Pascal VOC detection dataset into FiftyOne."""
    _require_fiftyone()
    dataset_name = name or Path(image_dir).name
    return fo.Dataset.from_dir(
        dataset_type=fo.types.VOCDetectionDataset,
        data_path=image_dir,
        name=dataset_name,
    )


def dataset_stats(annotation_json: str) -> dict[str, Any]:
    """Compute dataset statistics from a COCO annotation file.

    Does NOT require FiftyOne -- parses the JSON directly.
    Returns: dict with keys: num_images, num_annotations, class_distribution,
             image_sizes (min/max/mean width and height).
    """
    with open(annotation_json) as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}

    # Class distribution
    class_counts = Counter(
        categories.get(ann["category_id"], f"unknown-{ann['category_id']}")
        for ann in annotations
    )

    # Image sizes
    widths = [img["width"] for img in images]
    heights = [img["height"] for img in images]

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "class_distribution": dict(class_counts.most_common()),
        "image_sizes": {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "mean_width": sum(widths) / len(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "mean_height": sum(heights) / len(heights) if heights else 0,
        },
    }
