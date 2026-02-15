# CIFAR-MLX

CIFAR-10/100 image classifier using Apple's [MLX framework](https://github.com/ml-explore/mlx), optimized for Apple Silicon via Metal (no MPS translation layer).

## Setup

```bash
cd cifar-mlx
uv sync
```

## Usage

```bash
# Small CNN on CIFAR-10 (quick experiment)
uv run python train.py

# ResNet-18 with OneCycleLR scheduling (best accuracy)
uv run python train.py --model resnet18 --scheduler onecycle --epochs 40

# CIFAR-100
uv run python train.py --dataset cifar100 --model resnet18 --scheduler onecycle

# Force CPU
uv run python train.py --cpu
```

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--model` | `small-cnn` | `small-cnn` or `resnet18` |
| `--epochs` | `30` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--seed` | `1111` | Random seed |
| `--lr` | `3e-4` | Learning rate (AdamW only) |
| `--scheduler` | `none` | `none` (AdamW) or `onecycle` (SGD + warmup + cosine) |
| `--cpu` | off | Force CPU execution |

## Models

### Small CNN (~0.5M params)
- 6 conv layers in 3 blocks (32→32→64→64→128→128 channels)
- BatchNorm + residual skip connections per block
- MaxPool2d + Dropout(0.25) after each block
- FC head: 2048 → 128 → num_classes

### ResNet-18 (~11.2M params)
- Adapted for CIFAR 32x32 images (3x3 conv1, no maxpool)
- 4 layer groups: [64, 128, 256, 512] channels
- Global average pooling → linear classifier

## Key MLX Differences from PyTorch

- **NHWC format**: Tensors are (batch, height, width, channels), not NCHW
- **Lazy evaluation**: Must call `mx.eval()` after each optimizer update
- **Functional gradients**: Uses `nn.value_and_grad()` instead of `.backward()`
- **No autograd tape**: Gradient computation is explicit and functional

## Results

> All results on Apple M4 Max, MLX 0.30.6, seed 1111.

### CIFAR-10

| Model | Params | Accuracy | Epochs | Time | Optimizer |
|-------|--------|----------|--------|------|-----------|
| Small CNN | 0.57M | 71% | 2 | 13.61s | AdamW (lr=3e-4) |
| **ResNet-18** | **11.17M** | **94%** | **40** | **1423s** | SGD + OneCycleLR |

### CIFAR-100

| Model | Params | Accuracy | Epochs | Time | Optimizer |
|-------|--------|----------|--------|------|-----------|
| Small CNN | 0.58M | 28% | 2 | 13.25s | AdamW (lr=3e-4) |

### vs PyTorch (ResNet-18, CIFAR-10, 40 epochs)

| | MLX (Metal) | PyTorch (MPS) |
|---|---|---|
| **Accuracy** | **94%** | 92% |
| **Training Time** | 1423s | 1062s |
| **Final Loss** | 0.025 | 0.031 |

### Saved Weights

| File | Model | Dataset | Epochs | Accuracy |
|------|-------|---------|--------|----------|
| `cifar_mlx_20260215_092520.npz` | Small CNN | CIFAR-10 | 2 | 71% |
| `cifar_mlx_20260215_092821.npz` | ResNet-18 | CIFAR-10 | 2 | 74% |
| `cifar_mlx_20260215_092845.npz` | Small CNN | CIFAR-100 | 2 | 28% |
| `cifar_mlx_20260215_103051.npz` | ResNet-18 | CIFAR-10 | 40 | 94% |

Full per-class accuracy tables are in the root [Results.md](../Results.md).
