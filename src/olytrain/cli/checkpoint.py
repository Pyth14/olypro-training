"""Checkpoint commands -- list, prune, sync."""

import click


@click.group()
def checkpoint() -> None:
    """Manage training checkpoints."""


@checkpoint.command("list")
@click.option("--project", default=None, help="Filter by project name.")
def list_checkpoints(project: str | None) -> None:
    """List checkpoints across projects."""
    from rich.console import Console
    from rich.table import Table

    from olytrain.checkpoint.manager import CheckpointManager

    mgr = CheckpointManager()
    ckpts = mgr.list(project=project)
    if not ckpts:
        click.echo("No checkpoints found.")
        return
    console = Console()
    table = Table(title="Checkpoints")
    table.add_column("Project")
    table.add_column("Path")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Modified")
    for c in ckpts:
        table.add_row(
            c.project, str(c.path), f"{c.size_mb:.1f}", c.modified.strftime("%Y-%m-%d %H:%M")
        )
    console.print(table)


@checkpoint.command()
@click.argument("project")
@click.option("--keep", default=5, type=int, help="Number of checkpoints to keep.")
@click.option("--execute", is_flag=True, help="Actually delete (default is dry-run).")
def prune(project: str, keep: int, execute: bool) -> None:
    """Prune old checkpoints, keeping the best."""
    from olytrain.checkpoint.manager import CheckpointManager

    mgr = CheckpointManager()
    to_delete = mgr.prune(project=project, keep_best=keep, dry_run=not execute)
    if not to_delete:
        click.echo("Nothing to prune.")
        return
    action = "Deleted" if execute else "Would delete"
    for c in to_delete:
        click.echo(f"  {action}: {c.path} ({c.size_mb:.1f} MB)")


@checkpoint.command()
@click.argument("local_path")
@click.argument("remote_host")
@click.argument("remote_path")
@click.option("--download", is_flag=True, help="Download from remote instead of upload.")
def sync(local_path: str, remote_host: str, remote_path: str, download: bool) -> None:
    """Sync checkpoints to/from remote."""
    from olytrain.checkpoint.sync import sync_from_remote, sync_to_remote

    if download:
        sync_from_remote(remote_host, remote_path, local_path)
    else:
        sync_to_remote(local_path, remote_host, remote_path)
