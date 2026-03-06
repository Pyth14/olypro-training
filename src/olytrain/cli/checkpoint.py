"""Checkpoint commands — list, prune, sync."""

import click


@click.group()
def checkpoint() -> None:
    """Manage training checkpoints."""


@checkpoint.command("list")
def list_checkpoints() -> None:
    """List checkpoints across projects."""
    click.echo("Not yet implemented")


@checkpoint.command()
def prune() -> None:
    """Prune old checkpoints, keeping the best."""
    click.echo("Not yet implemented")


@checkpoint.command()
def sync() -> None:
    """Sync checkpoints to/from remote."""
    click.echo("Not yet implemented")
