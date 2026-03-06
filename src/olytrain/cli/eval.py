"""Eval commands — run evaluation and generate reports."""

import click


@click.group("eval")
def eval_cmd() -> None:
    """Run evaluation and generate reports."""


@eval_cmd.command()
def movenet() -> None:
    """Evaluate a movenet checkpoint."""
    click.echo("Not yet implemented")


@eval_cmd.command()
def vision() -> None:
    """Evaluate a vision checkpoint."""
    click.echo("Not yet implemented")
