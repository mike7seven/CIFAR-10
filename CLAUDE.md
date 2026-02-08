# CLAUDE.md

## Project Overview

CIFAR-10 image classifier using PyTorch CNNs with Apple Silicon MPS acceleration. Based on the official PyTorch tutorial, extended with data augmentation, OneCycleLR scheduling, and a wider network.

## Tech Stack

- **Language:** Python 3.12
- **Framework:** PyTorch + torchvision
- **Package Manager:** uv (`uv sync` to install, `uv run` to execute)
- **Hardware Target:** Apple M-Series Macs via MPS (Metal Performance Shaders)

## Key Files

- `cifar-tutorial.py` — Main training script (the actual code; `main.py` is an unused stub)
- `Results.md` — Detailed training metrics and per-class accuracy
- `Conclusion.md` — Scientific findings and next steps
- `.claude/skills.md` — Claude skill definitions

## Commands

```bash
# Train with MPS (default)
uv run python cifar-tutorial.py

# Train on CPU (for benchmarking)
USE_CPU=1 uv run python cifar-tutorial.py
```

## Architecture

- **Model:** 2 conv layers (64, 128 channels) + 3 FC layers (~1.98M params)
- **Optimizer:** SGD (lr=0.001, momentum=0.9, weight_decay=5e-4)
- **Scheduler:** OneCycleLR (max_lr=0.1, 30% warmup, cosine annealing)
- **Data Augmentation:** RandomCrop(32, padding=4), RandomHorizontalFlip
- **Batch Size:** 64
- **Epochs:** 40
- **Seed:** 1111
- **Best Accuracy:** 85%

## Device Selection

The script auto-detects hardware: MPS > CUDA > CPU. Set `USE_CPU=1` env var to force CPU.

## Model Checkpoints

Saved as `cifar_net_<timestamp>.pth` in the project root. These are not tracked in git.
