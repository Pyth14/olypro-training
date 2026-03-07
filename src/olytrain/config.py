"""Shared constants, paths, and settings for olytrain."""

from pathlib import Path

MLFLOW_DIR = Path.home() / ".olypro-mlflow"
MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
ARTIFACT_ROOT = MLFLOW_DIR / "artifacts"

DEFAULT_EXPERIMENTS = ("movenet", "vision")

OLYPRO_MOVENET_DIR = Path.home() / "src" / "olypro-movenet"
OLYPRO_VISION_DIR = Path.home() / "src" / "olypro-vision"

# Project-specific checkpoint directories
CHECKPOINT_DIRS = {
    "movenet": OLYPRO_MOVENET_DIR / "output",
    "vision": OLYPRO_VISION_DIR / "models",
}
