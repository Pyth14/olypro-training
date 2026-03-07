"""Init command -- detect projects, verify paths, set up MLflow."""

from __future__ import annotations

import click
from rich.console import Console

from olytrain.config import (
    MLFLOW_TRACKING_URI,
    OLYPRO_MOVENET_DIR,
    OLYPRO_VISION_DIR,
)


def _check(console: Console, label: str, exists: bool) -> None:
    """Print a checkmark or cross for a path check."""
    if exists:
        console.print(f"  [green]v[/green] {label}")
    else:
        console.print(f"  [red]x[/red] {label}")


@click.command()
def init() -> None:
    """Initialize olytrain -- detect projects, verify paths, set up MLflow."""
    from olytrain.integrations.mlflow_setup import ensure_mlflow

    console = Console()

    console.print("[bold]Setting up MLflow...[/bold]")
    ensure_mlflow()
    console.print(f"  [green]v[/green] MLflow DB ready at {MLFLOW_TRACKING_URI}")

    console.print()
    console.print("[bold]Checking olypro-movenet...[/bold]")
    if OLYPRO_MOVENET_DIR.exists():
        console.print(f"  [green]v[/green] Found {OLYPRO_MOVENET_DIR}")
        _check(console, "data/ directory", (OLYPRO_MOVENET_DIR / "data").is_dir())
        _check(console, "output/ directory", (OLYPRO_MOVENET_DIR / "output").is_dir())
        _check(console, "config.py", (OLYPRO_MOVENET_DIR / "config.py").is_file())
    else:
        console.print(f"  [red]x[/red] Not found: {OLYPRO_MOVENET_DIR}")

    console.print()
    console.print("[bold]Checking olypro-vision...[/bold]")
    if OLYPRO_VISION_DIR.exists():
        console.print(f"  [green]v[/green] Found {OLYPRO_VISION_DIR}")
        _check(console, "annotations/ directory", (OLYPRO_VISION_DIR / "annotations").is_dir())
        _check(console, "configs/ directory", (OLYPRO_VISION_DIR / "configs").is_dir())
        _check(console, "models/ directory", (OLYPRO_VISION_DIR / "models").is_dir())
    else:
        console.print(f"  [red]x[/red] Not found: {OLYPRO_VISION_DIR}")

    console.print()
    console.print(f"[bold]MLflow tracking URI:[/bold] {MLFLOW_TRACKING_URI}")
    console.print("Run [cyan]olytrain dashboard[/cyan] to start the UI")
