"""Eval commands — run evaluation and generate reports."""

import click


@click.group("eval")
def eval_cmd() -> None:
    """Run evaluation and generate reports."""


@eval_cmd.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Save figures to directory.")
def movenet(checkpoint: str, output: str | None) -> None:
    """Evaluate a movenet checkpoint."""
    click.echo(f"Evaluation for checkpoint: {checkpoint}")
    click.echo("Model loading and evaluation not yet integrated.")
    click.echo("Use olytrain eval with a trained model to generate reports.")


@eval_cmd.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.option("--output", "-o", type=click.Path(), default=None, help="Save figures to directory.")
def vision(checkpoint: str, config_path: str | None, output: str | None) -> None:
    """Evaluate a vision checkpoint."""
    click.echo(f"Evaluation for checkpoint: {checkpoint}")
    click.echo("Model loading and evaluation not yet integrated.")
    click.echo("Use olytrain eval with a trained model to generate reports.")
