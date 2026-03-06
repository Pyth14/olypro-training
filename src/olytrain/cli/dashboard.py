"""Dashboard command — launches MLflow UI."""

import subprocess
import sys

import click

from olytrain.config import ARTIFACT_ROOT, MLFLOW_TRACKING_URI
from olytrain.integrations.mlflow_setup import ensure_mlflow


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind MLflow UI to.")
@click.option("--port", default=5000, type=int, help="Port to bind MLflow UI to.")
def dashboard(host: str, port: int) -> None:
    """Launch the MLflow dashboard UI."""
    ensure_mlflow()
    click.echo(f"Starting MLflow UI at http://{host}:{port}")
    click.echo(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                MLFLOW_TRACKING_URI,
                "--default-artifact-root",
                str(ARTIFACT_ROOT),
                "--host",
                host,
                "--port",
                str(port),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        click.echo("\nMLflow UI stopped.")
