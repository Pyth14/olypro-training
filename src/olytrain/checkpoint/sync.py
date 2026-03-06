"""Cloud sync for checkpoints (wraps scp/rsync)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import click


def sync_to_remote(local_path: str, remote_host: str, remote_path: str) -> None:
    """Upload a checkpoint to a remote host via rsync."""
    cmd = ["rsync", "-avz", "--progress", local_path, f"{remote_host}:{remote_path}"]
    click.echo(f"Syncing {local_path} -> {remote_host}:{remote_path}")
    subprocess.run(cmd, check=True)


def sync_from_remote(remote_host: str, remote_path: str, local_path: str) -> None:
    """Download a checkpoint from a remote host via rsync."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avz", "--progress", f"{remote_host}:{remote_path}", local_path]
    click.echo(f"Syncing {remote_host}:{remote_path} -> {local_path}")
    subprocess.run(cmd, check=True)


def watch_and_sync(
    local_dir: str, remote_host: str, remote_dir: str, interval: int = 3600
) -> None:
    """Periodically sync a local checkpoint directory to remote."""
    click.echo(f"Watching {local_dir} every {interval}s, syncing to {remote_host}:{remote_dir}")
    try:
        while True:
            sync_to_remote(local_dir, remote_host, remote_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nSync stopped.")
