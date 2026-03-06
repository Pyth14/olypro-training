"""Tests for evaluation visualization functions."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image as PILImage
from rich.table import Table

matplotlib.use("Agg")

from olytrain.eval.detection import class_ap_chart, detection_overlay, detection_report
from olytrain.eval.pose import keypoint_accuracy_heatmap, per_keypoint_report, prediction_overlay

# -- Pose eval tests --


class TestKeypointAccuracyHeatmap:
    def test_returns_figure(self) -> None:
        names = ["nose", "left_eye", "right_eye"]
        results = {"nose": 0.95, "left_eye": 0.8, "right_eye": 0.6}
        fig = keypoint_accuracy_heatmap(results, names)
        try:
            assert isinstance(fig, Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_missing_keypoint_defaults_to_zero(self) -> None:
        names = ["nose", "left_eye"]
        results = {"nose": 0.9}
        fig = keypoint_accuracy_heatmap(results, names)
        try:
            assert isinstance(fig, Figure)
            ax = fig.axes[0]
            # The heatmap data should have 0.0 for missing keypoints
            images = ax.get_images()
            assert len(images) == 1
            data = images[0].get_array()
            assert data[0, 1] == pytest.approx(0.0)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


class TestPredictionOverlay:
    def _make_dummy_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pred_kps = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        gt_kps = np.array([[12.0, 22.0], [32.0, 42.0], [52.0, 62.0]])
        return image, pred_kps, gt_kps

    def test_returns_figure_with_ndarray(self) -> None:
        image, pred_kps, gt_kps = self._make_dummy_data()
        fig = prediction_overlay(image, pred_kps, gt_kps)
        try:
            assert isinstance(fig, Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_returns_figure_with_pil_image(self) -> None:
        pil_image = PILImage.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        pred_kps = np.array([[10.0, 20.0], [30.0, 40.0]])
        gt_kps = np.array([[12.0, 22.0], [32.0, 42.0]])
        fig = prediction_overlay(pil_image, pred_kps, gt_kps)
        try:
            assert isinstance(fig, Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_with_skeleton(self) -> None:
        image, pred_kps, gt_kps = self._make_dummy_data()
        skeleton = [(0, 1), (1, 2)]
        fig = prediction_overlay(image, pred_kps, gt_kps, skeleton=skeleton)
        try:
            assert isinstance(fig, Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


class TestPerKeypointReport:
    def test_returns_table(self) -> None:
        names = ["nose", "left_eye", "right_eye"]
        results = {"nose": 0.95, "left_eye": 0.6, "right_eye": 0.3}
        table = per_keypoint_report(results, names)
        assert isinstance(table, Table)
        assert table.row_count == 3


# -- Detection eval tests --


class TestClassApChart:
    def test_returns_figure(self) -> None:
        names = ["car", "person", "bike"]
        results = {"car": 0.7, "person": 0.4, "bike": 0.2}
        fig = class_ap_chart(results, names)
        try:
            assert isinstance(fig, Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


class TestDetectionOverlay:
    def test_returns_figure(self) -> None:
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        gt_boxes = [{"bbox": (10, 10, 50, 50), "class_id": 0}]
        pred_boxes = [{"bbox": (12, 12, 48, 48), "class_id": 0, "score": 0.9}]
        class_names = ["car", "person"]
        fig = detection_overlay(image, pred_boxes, gt_boxes, class_names)
        try:
            assert isinstance(fig, Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_with_pil_image(self) -> None:
        pil_image = PILImage.fromarray(
            np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        )
        gt_boxes = [{"bbox": (10, 10, 50, 50), "class_id": 0}]
        pred_boxes = [{"bbox": (12, 12, 48, 48), "class_id": 0, "score": 0.9}]
        fig = detection_overlay(pil_image, pred_boxes, gt_boxes)
        try:
            assert isinstance(fig, Figure)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


class TestDetectionReport:
    def test_returns_table_with_map_row(self) -> None:
        names = ["car", "person", "bike"]
        results = {"car": 0.7, "person": 0.4, "bike": 0.2}
        table = detection_report(results, names)
        assert isinstance(table, Table)
        # 3 class rows + 1 mAP row
        assert table.row_count == 4
