# olytrain

Shared training infrastructure for olypro projects. Integrates MLflow experiment tracking, FiftyOne dataset inspection, checkpoint management, and evaluation tooling.

## Installation

```bash
pip install -e .

# With optional FiftyOne support:
pip install -e ".[fiftyone]"

# Development:
pip install -e ".[dev]"
```

## Quick Start

### Launch MLflow Dashboard

```bash
olytrain dashboard
olytrain dashboard --host 0.0.0.0 --port 8080
```

### Training Callbacks (PyTorch)

Add MLflow logging to movenet training with ~5 lines:

```python
try:
    from olytrain.integrations.mlflow_task import MLflowTaskCallback
    callback = MLflowTaskCallback(config=training_config, run_name="experiment-1")
except ImportError:
    callback = None

# In training loop:
if callback:
    callback.on_train_start()
    # ... on_batch_end(step, {"loss": loss}) ...
    # ... on_epoch_end(epoch, {"val/acc": acc}) ...
    callback.on_train_end()
```

For RTMPose training:

```python
from olytrain.integrations.mlflow_rtmpose import MLflowRTMPoseCallback
callback = MLflowRTMPoseCallback(config=config)
```

### Ingest TensorBoard Events (TF-Vision)

```bash
olytrain runs ingest /path/to/tb/events --experiment vision --run-name "efficientdet-d1"
olytrain runs ingest /path/to/tb/events --config training_config.yaml
```

### Browse & Compare Runs

```bash
olytrain runs list --experiment movenet --sort val/acc
olytrain runs compare <run_id_1> <run_id_2>
```

### Dataset Inspection

```bash
# View stats (no FiftyOne needed):
olytrain dataset stats annotations.json

# Visual inspection (requires fiftyone):
olytrain dataset inspect annotations.json images/ --type keypoints
olytrain dataset inspect annotations.json images/ --type detection
```

### Checkpoint Management

```bash
olytrain checkpoint list
olytrain checkpoint list --project movenet
olytrain checkpoint prune movenet --keep 5
olytrain checkpoint prune movenet --keep 3 --execute
olytrain checkpoint sync /local/path remote-host /remote/path
olytrain checkpoint sync /local/path remote-host /remote/path --download
```

### Config Diffing

```bash
olytrain config show config.yaml
olytrain config diff config_v1.yaml config_v2.yaml
```

### Evaluation

```bash
olytrain eval movenet checkpoint.pth --output results/
olytrain eval vision checkpoint.h5 --config config.yaml --output results/
```

## Architecture

- **Composition over modification** — Callbacks are standalone objects, training scripts need ~5 lines added
- **SQLite backend** — Zero-setup MLflow with local SQLite database
- **Optional FiftyOne** — Dataset inspection is opt-in (`pip install olytrain[fiftyone]`)
- **DDP-safe** — Callbacks only log from rank 0

## Documentation

- **[Getting Started Tutorial](docs/getting-started.md)** -- step-by-step guide from install to first tracked run
- **[Onboarding Presentation](docs/presentation.md)** -- Marp slide deck for team walkthroughs (render with `npx @marp-team/marp-cli docs/presentation.md -o slides.html`)

## Project Structure

```
src/olytrain/
├── cli/               # Click-based CLI commands
├── integrations/      # MLflow callbacks, TB parser, FiftyOne loaders
├── checkpoint/        # Checkpoint discovery, pruning, sync
├── eval/              # Evaluation visualizations (pose + detection)
└── config.py          # Shared constants and paths
```
