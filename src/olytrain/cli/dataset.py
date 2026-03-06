"""Dataset commands -- inspect and stats."""

import click


@click.group()
def dataset() -> None:
    """Inspect and analyze datasets."""


@dataset.command()
@click.argument("annotation_json", type=click.Path(exists=True))
@click.argument("image_dir", type=click.Path(exists=True))
@click.option(
    "--type",
    "dataset_type",
    type=click.Choice(["keypoints", "detection", "voc"]),
    default="detection",
)
@click.option("--name", default=None, help="Dataset name in FiftyOne.")
def inspect(
    annotation_json: str, image_dir: str, dataset_type: str, name: str | None
) -> None:
    """Launch FiftyOne dataset inspector."""
    from olytrain.integrations.fiftyone_loader import (
        load_coco_detection,
        load_coco_keypoints,
        load_voc_detection,
    )

    if dataset_type == "keypoints":
        ds = load_coco_keypoints(annotation_json, image_dir, name=name)
    elif dataset_type == "voc":
        ds = load_voc_detection(image_dir, name=name)
    else:
        ds = load_coco_detection(annotation_json, image_dir, name=name)

    import fiftyone as fo

    session = fo.launch_app(ds)
    session.wait()


@dataset.command()
@click.argument("annotation_json", type=click.Path(exists=True))
def stats(annotation_json: str) -> None:
    """Print dataset distribution statistics."""
    from rich.console import Console
    from rich.table import Table

    from olytrain.integrations.fiftyone_loader import dataset_stats

    s = dataset_stats(annotation_json)
    console = Console()
    console.print(
        f"Images: {s['num_images']}, Annotations: {s['num_annotations']}"
    )

    table = Table(title="Class Distribution")
    table.add_column("Class")
    table.add_column("Count", justify="right")
    for cls, count in s["class_distribution"].items():
        table.add_row(cls, str(count))
    console.print(table)

    sizes = s["image_sizes"]
    console.print(
        f"Width: {sizes['min_width']}-{sizes['max_width']}"
        f" (mean {sizes['mean_width']:.0f})"
    )
    console.print(
        f"Height: {sizes['min_height']}-{sizes['max_height']}"
        f" (mean {sizes['mean_height']:.0f})"
    )
