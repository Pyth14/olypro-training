# Getting Started with olytrain

This guide walks you through installing olytrain, tracking your first training run, and using the tools to manage experiments across olypro projects.

## Prerequisites

- Python 3.10+
- An existing olypro-movenet or olypro-vision checkout (optional, for integration)

## 1. Installation

```bash
# Clone the repo
git clone https://github.com/Pyth14/olypro-training.git
cd olypro-training

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install olytrain
pip install -e .

# For development (adds pytest, ruff, mypy):
pip install -e ".[dev]"

# For dataset inspection with FiftyOne:
pip install -e ".[fiftyone]"
```

Verify it works:

```bash
olytrain --version
# olytrain, version 0.1.0
```

## 2. Launch the Dashboard

olytrain uses MLflow with a local SQLite database -- zero configuration needed.

```bash
olytrain dashboard
```

Open http://127.0.0.1:5000 in your browser. You'll see the MLflow UI with two pre-created experiments: **movenet** and **vision**.

> **Tip:** Use `olytrain dashboard --port 8080` if port 5000 is taken.

## 3. Add Tracking to Your Training Script

### For olypro-movenet (Task-based training)

Add these lines to your training script:

```python
# At the top of your script
try:
    from olytrain.integrations.mlflow_task import MLflowTaskCallback
    callback = MLflowTaskCallback(
        config=your_config_dict,      # Your training config
        experiment_name="movenet",     # Shows up in MLflow UI
        run_name="resnet50-lr0.001",   # Human-readable run name
    )
except ImportError:
    callback = None  # olytrain not installed -- training still works
```

Then call the hooks at the right places in your training loop:

```python
# Before training starts
if callback:
    callback.on_train_start()

# Inside your batch loop
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    if callback:
        callback.on_batch_end(step, {"train/loss": loss})

# At the end of each epoch
if callback:
    callback.on_epoch_end(epoch, {
        "val/acc": val_accuracy,
        "val/loss": val_loss,
    })

# When saving a checkpoint
if callback:
    callback.on_checkpoint("checkpoints/best.pth", metric_value=val_accuracy)

# After training finishes
if callback:
    callback.on_train_end()
```

**That's it.** Your training metrics, parameters, and checkpoints now appear in the MLflow dashboard.

### For olypro-movenet (RTMPose training)

Same pattern, different class:

```python
try:
    from olytrain.integrations.mlflow_rtmpose import MLflowRTMPoseCallback
    callback = MLflowRTMPoseCallback(config=config, run_name="rtmpose-s-coco")
except ImportError:
    callback = None
```

### For olypro-vision (TensorFlow)

No code changes needed! olypro-vision already writes TensorBoard events. Just ingest them after training:

```bash
olytrain runs ingest /path/to/model_dir/train \
    --experiment vision \
    --run-name "efficientdet-d1-run3" \
    --config /path/to/config.yaml
```

This reads the TensorBoard event files and logs all metrics (loss, AP, AP50, AP75) into MLflow.

## 4. Browse and Compare Runs

### In the browser

```bash
olytrain dashboard
```

Click an experiment, select multiple runs, and use MLflow's built-in comparison charts.

### From the command line

```bash
# List recent runs
olytrain runs list --experiment movenet

# Sort by a metric
olytrain runs list --experiment movenet --sort val/acc

# Compare two specific runs
olytrain runs compare <run_id_1> <run_id_2>
```

The compare command shows a side-by-side table of parameters and metrics with colored deltas.

## 5. Inspect Your Dataset

### Quick stats (no extra dependencies)

```bash
olytrain dataset stats /path/to/annotations.json
```

This parses the COCO JSON and prints:
- Number of images and annotations
- Class distribution table
- Image size ranges

### Visual inspection (requires FiftyOne)

```bash
pip install olytrain[fiftyone]

olytrain dataset inspect annotations.json images/ --type keypoints
olytrain dataset inspect annotations.json images/ --type detection
```

This launches the FiftyOne app in your browser where you can browse images, filter by class, and see annotation overlays.

## 6. Manage Checkpoints

### List all checkpoints

```bash
olytrain checkpoint list
olytrain checkpoint list --project movenet
```

### Prune old checkpoints

```bash
# Preview what would be deleted (dry run)
olytrain checkpoint prune movenet --keep 5

# Actually delete
olytrain checkpoint prune movenet --keep 5 --execute
```

### Sync to/from a remote machine

```bash
# Upload
olytrain checkpoint sync ./checkpoints/ gpu-server /data/checkpoints/

# Download
olytrain checkpoint sync ./local/ gpu-server /data/checkpoints/ --download
```

## 7. Compare Training Configs

When experimenting with different hyperparameters:

```bash
# View a config
olytrain config show config.yaml

# Diff two configs
olytrain config diff baseline.yaml experiment.yaml
```

The diff highlights added, removed, and changed parameters in a color-coded table.

## 8. DDP / Multi-GPU Training

The callbacks are DDP-safe by default. Only rank 0 logs to MLflow -- other ranks are automatic no-ops. No extra configuration needed.

```bash
# This just works
torchrun --nproc_per_node=4 train.py
```

## Common Workflows

### "I just finished a training run, now what?"

1. `olytrain dashboard` -- check your run appeared
2. `olytrain runs list --experiment movenet --sort val/acc` -- see where it ranks
3. `olytrain runs compare <old_best> <new_run>` -- compare against previous best
4. `olytrain checkpoint prune movenet --keep 5 --execute` -- clean up old checkpoints

### "I want to try a new config"

1. `olytrain config diff old_config.yaml new_config.yaml` -- review changes
2. Run training with the callback
3. `olytrain runs compare` -- see if it helped

### "I need to share results with the team"

1. `olytrain dashboard --host 0.0.0.0` -- expose MLflow on the network
2. Share the URL -- teammates can browse all experiments and runs

## Troubleshooting

**"olytrain: command not found"**
Make sure you activated the venv: `source .venv/bin/activate`

**"No experiments found"**
Run `olytrain dashboard` once to initialize the MLflow database.

**Metrics not appearing in MLflow**
- Check that `on_train_start()` was called before logging metrics
- Check that `on_train_end()` was called to finalize the run
- For TF-Vision, make sure you're pointing at the right TensorBoard event directory

**FiftyOne won't launch**
Install with: `pip install olytrain[fiftyone]`

## Quick Reference

| Command | What it does |
|---------|-------------|
| `olytrain dashboard` | Launch MLflow UI |
| `olytrain runs list` | List training runs |
| `olytrain runs compare A B` | Compare two runs |
| `olytrain runs ingest DIR` | Import TensorBoard events |
| `olytrain dataset stats FILE` | Dataset statistics |
| `olytrain dataset inspect FILE DIR` | Visual dataset browser |
| `olytrain checkpoint list` | List checkpoints |
| `olytrain checkpoint prune PROJECT` | Clean up old checkpoints |
| `olytrain checkpoint sync` | Upload/download checkpoints |
| `olytrain config show FILE` | Pretty-print config |
| `olytrain config diff A B` | Compare two configs |
| `olytrain eval movenet CKPT` | Evaluate pose model |
| `olytrain eval vision CKPT` | Evaluate detection model |
