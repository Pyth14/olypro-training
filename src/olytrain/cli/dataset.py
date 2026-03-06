"""Dataset commands — inspect and stats."""

import click


@click.group()
def dataset() -> None:
    """Inspect and analyze datasets."""


@dataset.command()
def inspect() -> None:
    """Launch FiftyOne dataset inspector."""
    click.echo("Not yet implemented")


@dataset.command()
def stats() -> None:
    """Print dataset distribution statistics."""
    click.echo("Not yet implemented")
