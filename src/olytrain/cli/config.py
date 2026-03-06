"""Config commands — diff and show training configs."""

import click


@click.group()
def config() -> None:
    """Manage and compare training configurations."""


@config.command()
def diff() -> None:
    """Diff two training configurations."""
    click.echo("Not yet implemented")


@config.command()
def show() -> None:
    """Pretty-print a training configuration."""
    click.echo("Not yet implemented")
