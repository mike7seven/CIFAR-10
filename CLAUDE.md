# CLAUDE.md

## Project Overview

CIFAR-10/100 image classifiers using both **PyTorch** (MPS acceleration) and **Apple MLX** (native Metal). Supports CIFAR-10 and CIFAR-100 via CLI `--dataset` flag.

## Tech Stack

- **Language:** Python 3.12
- **Frameworks:** PyTorch + torchvision (root), Apple MLX (`cifar-mlx/`)
- **Package Manager:** uv (`uv sync` to install, `uv run` to execute)
- **Hardware Target:** Apple M-Series Macs (MPS for PyTorch, native Metal for MLX)

## Key Files

- `cifar-tutorial.py` — PyTorch training script (`main.py` is an unused stub)
- `cifar-mlx/train.py` — MLX training script
- `cifar-mlx/model.py` — MLX model definitions (SmallCNN + ResNet-18)
- `cifar-mlx/dataset.py` — MLX data loading with augmentation
- `Results.md` — Detailed training metrics and per-class accuracy
- `Conclusion.md` — Scientific findings and next steps

## Commands

```bash
# PyTorch (CIFAR-10, default)
uv run python cifar-tutorial.py

# PyTorch (CIFAR-100)
uv run python cifar-tutorial.py --dataset cifar100

# PyTorch CLI flags: --dataset, --epochs, --batch-size, --seed

# MLX (from cifar-mlx/ directory)
cd cifar-mlx && uv run python train.py
cd cifar-mlx && uv run python train.py --model resnet18 --scheduler onecycle
cd cifar-mlx && uv run python train.py --dataset cifar100
```

## Architecture

### PyTorch (`cifar-tutorial.py`)
- **Model:** ResNet-18 adapted for CIFAR (3x3 conv1, no maxpool) (~11.2M params)
- **Optimizer:** SGD (lr=0.001, momentum=0.9, weight_decay=5e-4)
- **Scheduler:** OneCycleLR (max_lr=0.1, 30% warmup, cosine annealing)
- **Data Augmentation:** RandomCrop(32, padding=4), RandomHorizontalFlip
- **Batch Size:** 64
- **Epochs:** 40
- **Seed:** 1111
- **Best Accuracy:** 92% (CIFAR-10)

### MLX (`cifar-mlx/`)
- **Models:** SmallCNN (~0.5M params) or ResNet-18 (~11.2M params)
- **Optimizers:** AdamW (default) or SGD with OneCycleLR (`--scheduler onecycle`)
- **Data Augmentation:** Same as PyTorch (RandomCrop, RandomHFlip)
- **Separate venv:** `cifar-mlx/` has its own `pyproject.toml` and `.venv`

## Device Selection

**PyTorch:** Auto-detects MPS > CUDA > CPU. All training runs use MPS only.
**MLX:** Uses Metal GPU by default. Pass `--cpu` to force CPU.

## Training Runs

**Never run training jobs in parallel** — both PyTorch (MPS) and MLX (Metal) saturate the GPU. Running two at once will thrash the GPU, overheat the machine, and produce slower results than running sequentially. Always wait for one training run to finish before starting the next.

## Post-Training Checklist

After every training run, **always update `Results.md`** with:
- Overall accuracy and training time
- Per-class accuracy table
- Comparison against previous results
- Update the Acceptable Accuracy Targets table

## Model Checkpoints

- PyTorch: `cifar_net_<timestamp>.pth` in project root
- MLX: `cifar_mlx_<timestamp>.npz` in `cifar-mlx/`
- Checkpoints are not tracked in git
