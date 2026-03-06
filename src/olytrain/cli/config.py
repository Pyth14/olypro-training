"""Config commands — diff and show training configs."""

from __future__ import annotations

from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table


def _load_config(path: str) -> dict:
    """Load a config file (YAML or Python dict repr)."""
    text = Path(path).read_text()
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError:
        pass
    # Fall back to Python literal eval for dict-style configs
    import ast

    return dict(ast.literal_eval(text))


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict into dot-separated keys."""
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = str(v)
    return items


@click.group()
def config() -> None:
    """Manage and compare training configurations."""


@config.command()
@click.argument("config1", type=click.Path(exists=True))
@click.argument("config2", type=click.Path(exists=True))
def diff(config1: str, config2: str) -> None:
    """Diff two training configurations side-by-side."""
    c1 = _flatten(_load_config(config1))
    c2 = _flatten(_load_config(config2))
    all_keys = sorted(set(c1) | set(c2))

    console = Console()
    table = Table(title="Config Diff")
    table.add_column("Key")
    table.add_column(Path(config1).name, style="cyan")
    table.add_column(Path(config2).name, style="magenta")
    table.add_column("Status")

    for key in all_keys:
        v1 = c1.get(key, "[dim]<missing>[/dim]")
        v2 = c2.get(key, "[dim]<missing>[/dim]")
        if key not in c1:
            status = "[green]+ added[/green]"
        elif key not in c2:
            status = "[red]- removed[/red]"
        elif v1 != v2:
            status = "[yellow]~ changed[/yellow]"
        else:
            status = "="
        table.add_row(key, v1, v2, status)

    console.print(table)


@config.command()
@click.argument("config_path", type=click.Path(exists=True))
def show(config_path: str) -> None:
    """Pretty-print a training configuration."""
    data = _load_config(config_path)
    flat = _flatten(data)

    console = Console()
    table = Table(title=f"Config: {Path(config_path).name}")
    table.add_column("Key")
    table.add_column("Value")

    for key in sorted(flat):
        table.add_row(key, flat[key])

    console.print(table)
