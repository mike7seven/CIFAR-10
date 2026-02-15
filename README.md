# CIFAR-10/100 Classifier on M-Series Macs

Image classifiers trained on CIFAR-10 and CIFAR-100 using two frameworks on Apple Silicon:

- **PyTorch** with MPS (Metal Performance Shaders) acceleration
- **Apple MLX** with native Metal — no MPS translation layer

Based on the official [PyTorch CIFAR-10 Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) ([source code](https://github.com/pytorch/tutorials/blob/d7a5681/beginner_source/blitz/cifar10_tutorial.py)).

## Best Results

| Framework | Model | Dataset | Accuracy | Time |
|-----------|-------|---------|----------|------|
| **MLX** | ResNet-18 | CIFAR-10 | **94%** | 1423s |
| **PyTorch** | ResNet-18 | CIFAR-10 | **92%** | 1062s |

## Prerequisites

- M-Series Mac (M1, M2, M3, M4, M5)
- [UV](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Clone the repo
git clone https://github.com/mike7seven/CIFAR-10.git
cd CIFAR-10

# Install PyTorch dependencies
uv sync

# Install MLX dependencies (separate venv)
cd cifar-mlx && uv sync
```

## Usage

### PyTorch

```bash
# Train on CIFAR-10 (default)
uv run python cifar-tutorial.py

# Train on CIFAR-100
uv run python cifar-tutorial.py --dataset cifar100

# All CLI flags: --dataset, --epochs, --batch-size, --seed
```

### MLX

```bash
cd cifar-mlx

# Small CNN, quick experiment
uv run python train.py

# ResNet-18 with OneCycleLR (best accuracy)
uv run python train.py --model resnet18 --scheduler onecycle --epochs 40

# CIFAR-100
uv run python train.py --dataset cifar100 --model resnet18 --scheduler onecycle

# All CLI flags: --dataset, --model, --epochs, --batch-size, --seed, --lr, --scheduler, --cpu, --memory-limit
```

> **Warning:** Never run PyTorch and MLX training jobs in parallel — both saturate the GPU. Always run sequentially.

## MPS Performance

| Device | Training Time (20 epochs) | Speedup |
|--------|---------------------------|---------|
| CPU    | ~270s | 1x |
| MPS    | ~99s | **2.7x faster** |

## PyTorch vs MLX (ResNet-18, CIFAR-10, 40 epochs)

| | PyTorch (MPS) | MLX (Metal) |
|---|---|---|
| **Accuracy** | 92% | **94%** |
| **Training Time** | 1062s | 1423s |
| **Final Loss** | 0.031 | 0.025 |
| **Framework Version** | PyTorch 2.10 | MLX 0.30.6 |

MLX achieved +2% higher accuracy but trained 34% slower. MLX is more optimized for transformer/LLM workloads; CNN performance is expected to improve in future releases.

## Model Progression (PyTorch)

### 1. Small Custom LeNet-style CNN (~62K params) — 54% accuracy

The original PyTorch tutorial architecture:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

### 2. Large Custom LeNet-style CNN (~1.98M params) — 85% accuracy

Widened conv layers + data augmentation + OneCycleLR scheduling:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
```

### 3. ResNet-18 (~11.2M params) — 92% accuracy

Switched to `torchvision.models.resnet18`, adapted for CIFAR's 32x32 images:

```python
def cifar_resnet18(num_classes=10):
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net
```

## MLX Implementation

The `cifar-mlx/` subdirectory contains a parallel implementation using Apple's [MLX framework](https://github.com/ml-explore/mlx), which runs natively on Apple Silicon via Metal.

### Models

- **SmallCNN** (~0.57M params) — 6-conv CNN with BatchNorm, residual connections, dropout
- **ResNet-18** (~11.2M params) — Same CIFAR adaptation as PyTorch version, in NHWC format

### Key Differences from PyTorch

- **NHWC format** — tensors are `(batch, height, width, channels)`, not NCHW
- **Lazy evaluation** — must call `mx.eval()` after each optimizer update
- **Functional gradients** — uses `nn.value_and_grad()` instead of `.backward()`
- **Unified memory** — no `.to(device)` calls needed

See [cifar-mlx/README.md](cifar-mlx/README.md) for full MLX documentation.

## Hardware DNA

Each training run logs a hardware "fingerprint" to `run_log.json`, capturing the OS, chip, framework version, and compute backend. This exists because floating-point math is non-associative — `(a + b) + c != a + (b + c)` — and GPU backends parallelize reductions in non-deterministic order. Over 40 epochs of backpropagation through ~11.2M parameters, these micro-differences compound into measurably distinct weight tensors on different hardware.

## Activation Visualization

`visualize_activations.py` uses PyTorch forward hooks to capture feature maps at four depths of the ResNet-18, saving PNG grids to `activations/`:

- **conv1** — edge/color detectors (64 channels, 32x32)
- **layer1** — first residual block (64 channels, 32x32)
- **layer2** — mid-level features (128 channels, 16x16)
- **layer3** — higher-level features (256 channels, 8x8)

```bash
uv run python visualize_activations.py                          # most recent checkpoint
uv run python visualize_activations.py cifar_net_20260208.pth   # specific checkpoint
```

## Detailed Results

See [Results.md](Results.md) for full training metrics, per-class accuracy tables, and framework comparisons.

See [Conclusion.md](Conclusion.md) for scientific findings and next steps.

## Future Improvements

- [x] Add data augmentation to improve accuracy
- [x] Increase network size (ResNet-18) and test with 40 epochs
- [x] Add learning rate scheduling (OneCycleLR) to improve convergence
- [x] Port to Apple MLX framework for native Metal comparison
- [x] Add CIFAR-100 support to both frameworks
- [ ] Full CIFAR-100 training runs (PyTorch + MLX, 40 epochs)
- [ ] Implement various seeds to find optimal accuracy
