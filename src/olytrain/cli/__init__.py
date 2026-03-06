"""Click-based CLI for olytrain."""

import click

from olytrain.cli.checkpoint import checkpoint
from olytrain.cli.config import config
from olytrain.cli.dashboard import dashboard
from olytrain.cli.dataset import dataset
from olytrain.cli.eval import eval_cmd
from olytrain.cli.runs import runs


@click.group()
@click.version_option(package_name="olytrain")
def cli() -> None:
    """Training infrastructure for olypro projects."""


cli.add_command(dashboard)
cli.add_command(runs)
cli.add_command(dataset)
cli.add_command(checkpoint)
cli.add_command(eval_cmd)
cli.add_command(config)
