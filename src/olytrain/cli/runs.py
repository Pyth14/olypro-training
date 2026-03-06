"""Runs commands — list, compare, ingest."""

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
def ingest() -> None:
    """Ingest TensorBoard events into MLflow."""
    click.echo("Not yet implemented")
