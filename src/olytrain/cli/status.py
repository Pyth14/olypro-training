"""Status command -- show overview of experiments, runs, and checkpoints."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table


@click.command()
def status() -> None:
    """Show overview of experiments, runs, and checkpoints."""
    import mlflow

    from olytrain.checkpoint.manager import CheckpointManager
    from olytrain.config import OLYPRO_VISION_DIR
    from olytrain.integrations.mlflow_setup import ensure_mlflow

    console = Console()

    # --- MLflow Runs ---
    console.print("[bold]Recent MLflow Runs[/bold]")
    ensure_mlflow()

    experiments = [e for e in mlflow.search_experiments() if e.name != "Default"]
    if not experiments:
        console.print("  No experiments found.")
    else:
        table = Table()
        table.add_column("Run Name")
        table.add_column("Experiment")
        table.add_column("Status")
        table.add_column("Key Metric", justify="right")

        any_runs = False
        for exp in experiments:
            runs_df = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=5,
                output_format="pandas",
            )
            if runs_df.empty:  # type: ignore[union-attr]
                continue
            any_runs = True
            metric_cols = [
                c for c in runs_df.columns if c.startswith("metrics.")  # type: ignore[union-attr]
            ][:1]
            for _, row in runs_df.iterrows():  # type: ignore[union-attr]
                run_name = row.get("tags.mlflow.runName", "")
                run_status = row.get("status", "")
                key_metric = ""
                if metric_cols:
                    val = row.get(metric_cols[0])
                    if val is not None and val == val:
                        key_metric = f"{metric_cols[0].replace('metrics.', '')}: {val:.4f}"
                table.add_row(str(run_name), exp.name, str(run_status), key_metric)

        if any_runs:
            console.print(table)
        else:
            console.print("  No runs found.")

    # --- Checkpoints ---
    console.print()
    console.print("[bold]Checkpoint Disk Usage[/bold]")
    mgr = CheckpointManager()
    any_checkpoints = False
    for project_name, project_dir in mgr.project_dirs.items():
        ckpts = mgr.discover(project_dir, project_name)
        if ckpts:
            any_checkpoints = True
            total_mb = sum(c.size_mb for c in ckpts)
            console.print(f"  {project_name}: {len(ckpts)} checkpoint(s), {total_mb:.1f} MB")
        else:
            console.print(f"  {project_name}: no checkpoints found")
    if not any_checkpoints:
        console.print("  No checkpoints found across projects.")

    # --- Dataset Info ---
    console.print()
    console.print("[bold]Dataset Info[/bold]")
    annotation_paths = [
        OLYPRO_VISION_DIR / "annotations" / "instances_train.json",
        OLYPRO_VISION_DIR / "annotations" / "instances_val.json",
    ]
    any_datasets = False
    for ann_path in annotation_paths:
        if ann_path.is_file():
            any_datasets = True
            try:
                from olytrain.integrations.fiftyone_loader import dataset_stats

                s = dataset_stats(str(ann_path))
                num_classes = len(s.get("class_distribution", {}))
                console.print(
                    f"  {ann_path.name}: {s['num_images']} images, "
                    f"{num_classes} classes"
                )
            except Exception as exc:
                console.print(f"  {ann_path.name}: error reading - {exc}")
    if not any_datasets:
        console.print("  No annotation files found.")
