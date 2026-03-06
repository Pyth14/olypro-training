"""Tests for FiftyOne loader integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from olytrain.integrations.fiftyone_loader import (
    _require_fiftyone,
    dataset_stats,
)


@pytest.fixture()
def coco_json(tmp_path: Path) -> Path:
    """Create a minimal COCO annotation file."""
    data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 800, "height": 600},
            {"id": 3, "file_name": "img3.jpg", "width": 1024, "height": 768},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 15, 15]},
            {"id": 4, "image_id": 3, "category_id": 1, "bbox": [0, 0, 30, 30]},
            {"id": 5, "image_id": 3, "category_id": 3, "bbox": [20, 20, 40, 40]},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "car"},
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    return path


class TestRequireFiftyone:
    def test_raises_when_fiftyone_not_installed(self) -> None:
        with patch(
            "olytrain.integrations.fiftyone_loader.HAS_FIFTYONE", False
        ):
            with pytest.raises(ImportError, match="fiftyone is required"):
                _require_fiftyone()

    def test_no_error_when_fiftyone_installed(self) -> None:
        with patch(
            "olytrain.integrations.fiftyone_loader.HAS_FIFTYONE", True
        ):
            _require_fiftyone()  # should not raise


class TestDatasetStats:
    def test_num_images(self, coco_json: Path) -> None:
        result = dataset_stats(str(coco_json))
        assert result["num_images"] == 3

    def test_num_annotations(self, coco_json: Path) -> None:
        result = dataset_stats(str(coco_json))
        assert result["num_annotations"] == 5

    def test_class_distribution(self, coco_json: Path) -> None:
        result = dataset_stats(str(coco_json))
        dist = result["class_distribution"]
        assert dist["person"] == 3
        assert dist["dog"] == 1
        assert dist["car"] == 1

    def test_class_distribution_ordering(self, coco_json: Path) -> None:
        result = dataset_stats(str(coco_json))
        classes = list(result["class_distribution"].keys())
        assert classes[0] == "person"  # most common first

    def test_image_sizes(self, coco_json: Path) -> None:
        result = dataset_stats(str(coco_json))
        sizes = result["image_sizes"]
        assert sizes["min_width"] == 640
        assert sizes["max_width"] == 1024
        assert sizes["mean_width"] == pytest.approx(821.333, rel=1e-2)
        assert sizes["min_height"] == 480
        assert sizes["max_height"] == 768
        assert sizes["mean_height"] == pytest.approx(616.0, rel=1e-2)

    def test_empty_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
        result = dataset_stats(str(path))
        assert result["num_images"] == 0
        assert result["num_annotations"] == 0
        assert result["class_distribution"] == {}
        assert result["image_sizes"]["min_width"] == 0

    def test_unknown_category(self, tmp_path: Path) -> None:
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 99, "bbox": [0, 0, 10, 10]}
            ],
            "categories": [],
        }
        path = tmp_path / "unknown.json"
        path.write_text(json.dumps(data))
        result = dataset_stats(str(path))
        assert "unknown-99" in result["class_distribution"]


class TestStatsCommand:
    def test_stats_cli(self, coco_json: Path) -> None:
        from olytrain.cli.dataset import dataset

        runner = CliRunner()
        result = runner.invoke(dataset, ["stats", str(coco_json)])
        assert result.exit_code == 0
        assert "Images: 3" in result.output
        assert "Annotations: 5" in result.output
        assert "person" in result.output
