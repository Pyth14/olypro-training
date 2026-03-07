"""Runs commands -- list, compare, ingest."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table


def _color_metric(name: str, value: float) -> str:
    """Color a metric value based on known thresholds."""
    if "loss" in name:
        # Lower is better for loss
        if value < 0.3:
            return f"[green]{value:.4f}[/green]"
        if value < 0.7:
            return f"[yellow]{value:.4f}[/yellow]"
        return f"[red]{value:.4f}[/red]"
    if "acc" in name or "pck" in name:
        # Higher is better for accuracy
        if value >= 0.8:
            return f"[green]{value:.4f}[/green]"
        if value >= 0.5:
            return f"[yellow]{value:.4f}[/yellow]"
        return f"[red]{value:.4f}[/red]"
    if name.startswith("AP") or name == "mAP":
        if value >= 0.5:
            return f"[green]{value:.4f}[/green]"
        if value >= 0.3:
            return f"[yellow]{value:.4f}[/yellow]"
        return f"[red]{value:.4f}[/red]"
    return f"{value:.4f}"


@click.group()
def runs() -> None:
    """Manage and inspect training runs."""


@runs.command("list")
@click.option("--experiment", default=None, help="Filter by experiment name.")
@click.option("--sort", "sort_by", default=None, help="Sort by metric name.")
@click.option("--limit", default=20, type=int, help="Max runs to show.")
def list_runs(experiment: str | None, sort_by: str | None, limit: int) -> None:
    """List training runs."""
    import mlflow

    from olytrain.integrations.mlflow_setup import ensure_mlflow

    ensure_mlflow()

    if experiment:
        exp = mlflow.get_experiment_by_name(experiment)
        if not exp:
            click.echo(f"Experiment '{experiment}' not found.")
            return
        exp_ids = [exp.experiment_id]
    else:
        exp_ids = [
            e.experiment_id
            for e in mlflow.search_experiments()
            if e.name != "Default"
        ]

    if not exp_ids:
        click.echo("No experiments found.")
        return

    order = [f"metrics.{sort_by} DESC"] if sort_by else ["start_time DESC"]
    runs_df = mlflow.search_runs(
        experiment_ids=exp_ids, order_by=order, max_results=limit,
        output_format="pandas",
    )

    if runs_df.empty:  # type: ignore[union-attr]
        click.echo("No runs found.")
        return

    console = Console()
    table = Table(title="Training Runs")
    table.add_column("Run ID", style="dim")
    table.add_column("Name")
    table.add_column("Experiment")
    table.add_column("Status")
    table.add_column("Start Time")

    # Add metric columns dynamically
    metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")][:5]  # type: ignore[union-attr]
    for col in metric_cols:
        table.add_column(col.replace("metrics.", ""), justify="right")

    for _, row in runs_df.iterrows():  # type: ignore[union-attr]
        values = [
            row.get("run_id", "")[:8],
            row.get("tags.mlflow.runName", ""),
            row.get("experiment_id", ""),
            row.get("status", ""),
            str(row.get("start_time", ""))[:19],
        ]
        for col in metric_cols:
            val = row.get(col)
            if val is not None and val == val:
                metric_name = col.replace("metrics.", "")
                values.append(_color_metric(metric_name, val))
            else:
                values.append("")
        table.add_row(*values)

    console.print(table)


@runs.command()
@click.argument("run1")
@click.argument("run2")
def compare(run1: str, run2: str) -> None:
    """Compare two training runs side-by-side."""
    import mlflow

    from olytrain.integrations.mlflow_setup import ensure_mlflow

    ensure_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        r1 = client.get_run(run1)
        r2 = client.get_run(run2)
    except mlflow.exceptions.MlflowException as e:
        click.echo(f"Error: {e}")
        return

    console = Console()

    # Compare params
    p1, p2 = r1.data.params, r2.data.params
    all_params = sorted(set(p1) | set(p2))
    if all_params:
        ptable = Table(title="Parameters")
        ptable.add_column("Param")
        ptable.add_column(f"Run {run1[:8]}", style="cyan")
        ptable.add_column(f"Run {run2[:8]}", style="magenta")
        ptable.add_column("Match")
        for p in all_params:
            v1 = p1.get(p, "-")
            v2 = p2.get(p, "-")
            match = "[green]=[/green]" if v1 == v2 else "[yellow]!=[/yellow]"
            ptable.add_row(p, v1, v2, match)
        console.print(ptable)

    # Compare metrics
    m1, m2 = r1.data.metrics, r2.data.metrics
    all_metrics = sorted(set(m1) | set(m2))
    if all_metrics:
        mtable = Table(title="Metrics")
        mtable.add_column("Metric")
        mtable.add_column(f"Run {run1[:8]}", justify="right", style="cyan")
        mtable.add_column(f"Run {run2[:8]}", justify="right", style="magenta")
        mtable.add_column("Delta", justify="right")
        for m in all_metrics:
            v1 = m1.get(m)
            v2 = m2.get(m)
            s1 = f"{v1:.4f}" if v1 is not None else "-"
            s2 = f"{v2:.4f}" if v2 is not None else "-"
            delta = ""
            if v1 is not None and v2 is not None:
                d = v2 - v1
                color = "green" if d > 0 else "red" if d < 0 else "white"
                delta = f"[{color}]{d:+.4f}[/{color}]"
            mtable.add_row(m, s1, s2, delta)
        console.print(mtable)


@runs.command()
@click.argument("tb_dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    help="Training config YAML to log.",
)
@click.option("--experiment", default="vision", help="MLflow experiment name.")
@click.option("--run-name", default=None, help="MLflow run name.")
def ingest(
    tb_dir: str, config_path: str | None, experiment: str, run_name: str | None
) -> None:
    """Ingest TensorBoard events from TB_DIR into MLflow."""
    from olytrain.integrations.mlflow_vision import TBEventParser

    parser = TBEventParser(tb_dir, experiment_name=experiment)
    parser.parse_and_log(run_name=run_name, config_path=config_path)
    click.echo(f"Ingested events from {tb_dir}")
