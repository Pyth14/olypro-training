---
marp: true
theme: default
paginate: true
header: "**olytrain** -- Training Infrastructure for Olypro"
footer: "Pyth14 / olypro-training"
style: |
  section {
    font-size: 24px;
  }
  h1 {
    color: #2563eb;
  }
  h2 {
    color: #1e40af;
  }
  code {
    background: #f1f5f9;
    border-radius: 4px;
    padding: 2px 6px;
  }
  table {
    font-size: 20px;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
---

# olytrain

### Unified Training Infrastructure for Olypro Projects

**What:** A shared Python package for experiment tracking, dataset inspection, checkpoint management, and evaluation.

**Why:** Both olypro-movenet and olypro-vision had ad-hoc training scripts with no unified way to track experiments, compare runs, or manage checkpoints.

**How:** MLflow + FiftyOne + custom tooling, wrapped in a simple CLI.

---

# The Problem

### Before olytrain

- Training metrics logged to TensorBoard or print statements
- No way to compare runs across experiments
- Checkpoints accumulate with no cleanup strategy
- Dataset issues discovered late in training
- Config changes tracked in git commits (if at all)
- Each project had its own ad-hoc tooling

### The cost

Wasted GPU hours, lost experiments, "which config produced that result?"

---

# The Solution

### olytrain gives you

| Capability | Tool | Command |
|-----------|------|---------|
| Experiment tracking | MLflow | `olytrain dashboard` |
| Run comparison | MLflow + Rich | `olytrain runs compare` |
| Dataset inspection | FiftyOne | `olytrain dataset inspect` |
| Checkpoint management | Custom | `olytrain checkpoint list/prune` |
| Config diffing | Custom | `olytrain config diff` |
| Eval reports | Matplotlib + Rich | `olytrain eval` |

**One `pip install`, zero infrastructure.**

---

# Architecture

```
olypro-movenet                    olypro-vision
     |                                  |
     | MLflowTaskCallback               | TensorBoard events
     | (5 lines of code)                | (no code changes)
     |                                  |
     v                                  v
  +-------------------------------------------------+
  |                   olytrain                       |
  |                                                  |
  |  MLflow (SQLite)  |  FiftyOne  |  Checkpoints   |
  |  - params         |  - browse  |  - discover     |
  |  - metrics        |  - filter  |  - prune        |
  |  - artifacts      |  - stats   |  - sync         |
  +-------------------------------------------------+
                        |
                   CLI + Web UI
```

---

# Design Principles

### 1. Composition over modification

Callbacks are standalone objects. Training scripts get ~5 lines added, guarded by `try/except ImportError`. Both projects work fine without olytrain installed.

### 2. Zero-setup infrastructure

MLflow backend is SQLite. No server to deploy, no database to manage. Just `pip install` and go.

### 3. Optional dependencies

FiftyOne is heavy. It's optional: `pip install olytrain[fiftyone]`. Base package stays lightweight for cloud instances.

### 4. DDP-safe

Callbacks check `LOCAL_RANK`. Only rank 0 writes to MLflow. No duplicate metrics, no race conditions.

---

# Getting Started (2 minutes)

### Install

```bash
pip install -e /path/to/olypro-training
```

### Launch the dashboard

```bash
olytrain dashboard
```

### Open http://localhost:5000

You'll see two experiments pre-created: **movenet** and **vision**.

That's it. You're ready to start tracking.

---

# Adding Tracking to movenet

### Add to your training script

```python
try:
    from olytrain.integrations.mlflow_task import MLflowTaskCallback
    cb = MLflowTaskCallback(config=config, run_name="my-experiment")
except ImportError:
    cb = None
```

### Call hooks in your training loop

```python
if cb: cb.on_train_start()

for epoch in range(num_epochs):
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        if cb: cb.on_batch_end(step, {"train/loss": loss})

    val_acc = validate()
    if cb: cb.on_epoch_end(epoch, {"val/acc": val_acc})

if cb: cb.on_train_end()
```

---

# Adding Tracking to vision

### No code changes needed!

olypro-vision already writes TensorBoard events. Just ingest them:

```bash
olytrain runs ingest /path/to/model_dir/train \
    --experiment vision \
    --run-name "efficientdet-d1" \
    --config config.yaml
```

### What gets logged

- `total_loss`, `cls_loss`, `box_loss`, `regularization_loss`
- `AP`, `AP50`, `AP75`
- Config parameters (from YAML)

### Live monitoring

```bash
olytrain runs ingest /path/to/events --watch
```

Polls every 30 seconds during training.

---

# The MLflow Dashboard

### What you see

- **Experiments list** -- movenet, vision, grouped by project
- **Runs table** -- all runs with status, timestamps, metrics
- **Run detail** -- parameters, metric charts, artifacts
- **Comparison view** -- select multiple runs, overlay metric curves

### Key features

- **Search & filter** -- `metrics.val_acc > 0.8 AND params.backbone = "resnet50"`
- **Metric charts** -- interactive, zoomable, per-batch resolution
- **Artifact storage** -- checkpoints, configs, eval images

---

# Comparing Runs

### From the CLI

```bash
$ olytrain runs compare <run_1> <run_2>

              Parameters
+------------+-----------+-----------+-------+
| Param      | Run A     | Run B     | Match |
+------------+-----------+-----------+-------+
| backbone   | resnet50  | mobilenet | !=    |
| lr         | 0.001     | 0.01      | !=    |
| batch_size | 32        | 64        | !=    |
+------------+-----------+-----------+-------+

              Metrics
+------------+---------+---------+---------+
| Metric     | Run A   | Run B   | Delta   |
+------------+---------+---------+---------+
| val/acc    | 0.8200  | 0.8500  | +0.0300 |
| val/loss   | 0.4200  | 0.3800  | -0.0400 |
+------------+---------+---------+---------+
```

---

# Dataset Inspection

### Quick stats (no extra deps)

```bash
$ olytrain dataset stats annotations.json

Images: 5000, Annotations: 23847
+----------+-------+
| Class    | Count |
+----------+-------+
| person   | 12450 |
| dog      |  3200 |
| cat      |  2890 |
+----------+-------+
Width: 640-1920 (mean 1056)
Height: 480-1080 (mean 672)
```

### Visual browser (FiftyOne)

```bash
olytrain dataset inspect annotations.json images/ --type keypoints
```

Opens a web app where you can browse images, see keypoint overlays, filter by class, and spot annotation errors.

---

# Checkpoint Management

### The problem

After 50 training runs, you have 200+ checkpoint files consuming hundreds of GB.

### The solution

```bash
# See what you have
$ olytrain checkpoint list --project movenet

+----------+---------------------------+--------+------------------+
| Project  | Path                      | Size   | Modified         |
+----------+---------------------------+--------+------------------+
| movenet  | checkpoints/epoch_50.pth  | 245 MB | 2026-03-06 14:30 |
| movenet  | checkpoints/epoch_49.pth  | 245 MB | 2026-03-06 12:15 |
| ...      | ...                       | ...    | ...              |
+----------+---------------------------+--------+------------------+

# Keep the 5 most recent, delete the rest
$ olytrain checkpoint prune movenet --keep 5 --execute

# Sync best checkpoint to GPU server
$ olytrain checkpoint sync best.pth gpu-server /data/models/
```

---

# Config Diffing

### Before a new experiment

```bash
$ olytrain config diff baseline.yaml experiment.yaml

+--------------------+----------+----------+-----------+
| Key                | baseline | new      | Status    |
+--------------------+----------+----------+-----------+
| model.backbone     | resnet50 | mobilev2 | ~ changed |
| training.lr        | 0.001    | 0.01     | ~ changed |
| training.scheduler | -        | cosine   | + added   |
| augment.cutout     | true     | -        | - removed |
+--------------------+----------+----------+-----------+
```

Instantly see what changed before committing GPU time.

---

# Evaluation Reports

### Pose estimation

```bash
olytrain eval movenet checkpoint.pth
```

Generates:
- Per-keypoint accuracy heatmap (color-coded nose-to-ankle)
- Prediction overlay images (green = GT, red = predicted)
- Rich table with Good/Fair/Poor status per keypoint

### Object detection

```bash
olytrain eval vision checkpoint.h5
```

Generates:
- Per-class AP bar chart
- Bounding box overlay images
- Detection report with mAP summary

---

# Typical Workflow

```
1. olytrain config diff old.yaml new.yaml    # Review config changes
                    |
2. python train.py                            # Train with callback
                    |
3. olytrain dashboard                         # Check metrics in browser
                    |
4. olytrain runs compare old_best new_run     # Did it improve?
                    |
    +--- YES -------+------- NO ---+
    |                              |
5a. olytrain checkpoint            5b. Adjust config,
    prune --keep 5 --execute           go to step 1
    |
6. olytrain checkpoint sync
   best.pth gpu-server /models/
```

---

# What Changes in Your Code

### olypro-movenet: ~5 lines per training script

```python
# Add at top (guarded import)
try:
    from olytrain.integrations.mlflow_task import MLflowTaskCallback
    cb = MLflowTaskCallback(config=cfg, run_name="exp-name")
except ImportError:
    cb = None

# Add at training hooks (already exist in task.py)
if cb: cb.on_train_start()
if cb: cb.on_batch_end(step, metrics)
if cb: cb.on_epoch_end(epoch, metrics)
if cb: cb.on_train_end()
```

### olypro-vision: 0 lines

TensorBoard events are parsed externally. Run `olytrain runs ingest` after training.

---

# FAQ

**Q: Do I have to use olytrain?**
A: No. It's optional. Training works without it. The `try/except ImportError` guard means your scripts don't break if olytrain isn't installed.

**Q: Does it slow down training?**
A: Negligible. MLflow logging is ~1ms per call. Callbacks are no-ops on non-rank-0 GPUs.

**Q: Where is data stored?**
A: `~/.olypro-mlflow/mlflow.db` (SQLite) and `~/.olypro-mlflow/artifacts/`. Everything is local.

**Q: Can the team share a dashboard?**
A: Yes. Run `olytrain dashboard --host 0.0.0.0` on a shared machine. Or deploy MLflow server for production use.

**Q: What about Weights & Biases / Neptune / etc?**
A: MLflow is open-source, self-hosted, and free. No accounts, no vendor lock-in, no data leaving your network.

---

# Next Steps

1. **Install olytrain** in your environment
2. **Run `olytrain dashboard`** to see the UI
3. **Add the callback** to your next training run
4. **Check your dataset** with `olytrain dataset stats`
5. **Clean up checkpoints** with `olytrain checkpoint prune`

### Resources

- Repo: `github.com/Pyth14/olypro-training`
- Tutorial: `docs/getting-started.md`
- CLI help: `olytrain --help`

### Questions?
