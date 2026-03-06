"""Runs commands -- list, compare, ingest."""

from __future__ import annotations

import click


@click.group()
def runs() -> None:
    """Manage and inspect training runs."""


@runs.command("list")
def list_runs() -> None:
    """List training runs."""
    click.echo("Not yet implemented")


@runs.command()
def compare() -> None:
    """Compare two training runs side-by-side."""
    click.echo("Not yet implemented")


@runs.command()
@click.argument("tb_dir", type=click.Path(exists=True))
@click.option(
    "--config", "config_path", type=click.Path(exists=True), help="Training config YAML to log."
)
@click.option("--experiment", default="vision", help="MLflow experiment name.")
@click.option("--run-name", default=None, help="MLflow run name.")
def ingest(
    tb_dir: str, config_path: str | None, experiment: str, run_name: str | None
) -> None:
    """Ingest TensorBoard events from TB_DIR into MLflow."""
    from olytrain.integrations.mlflow_vision import TBEventParser

    parser = TBEventParser(tb_dir, experiment_name=experiment)
    parser.parse_and_log(run_name=run_name, config_path=config_path)
    click.echo(f"Ingested events from {tb_dir}")
