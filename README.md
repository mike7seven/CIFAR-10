# CIFAR-10 Classifier on M-Series Macs

A PyTorch CNN trained on CIFAR-10 using **MPS (Metal Performance Shaders)** for GPU acceleration on Apple Silicon.

Based on the official [PyTorch CIFAR-10 Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) ([source code](https://github.com/pytorch/tutorials/blob/d7a5681/beginner_source/blitz/cifar10_tutorial.py)).

## Prerequisites

- M-Series Mac (M1, M2, M3, M4, M5)
- [UV](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Clone the repo
git clone https://github.com/mike7seven/CIFAR-10.git
cd CIFAR-10

# Install dependencies with UV
uv sync
```

## Usage

```bash
# Train with MPS (default)
uv run python cifar-tutorial.py

# Train with CPU (for comparison)
USE_CPU=1 uv run python cifar-tutorial.py
```

## MPS Performance

| Device | Training Time (20 epochs) | Speedup |
|--------|---------------------------|---------|
| CPU    | ~270s | 1x |
| MPS    | ~99s | **2.7x faster** |

## Results

- **Accuracy:** 92% (40 epochs, seed 1111, OneCycleLR + augmentation)
- **Network:** ~11.2M parameters
- **Architecture:** ResNet-18 adapted for CIFAR-10 (3x3 conv1, no maxpool)

See [Results.md](Results.md) for detailed training metrics and per-class accuracy.

See [Conclusion.md](Conclusion.md) for scientific findings and next steps.

## Model Progression

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

### 3. ResNet-18 (~11.2M params) — 92% accuracy (current)

Switched to `torchvision.models.resnet18`, adapted for CIFAR-10's 32x32 images:

```python
def cifar10_resnet18():
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net
```

## Key Code for MPS

```python
# Device detection for M-Series Macs
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Move model and data to MPS
net.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```

## Recommended Configuration

- **Device:** MPS (Apple Metal)
- **Minimum Epochs:** 20 (40+ for best results with augmentation)
- **Data Augmentation:** Enabled for improved generalization

## Future Improvements

- [x] Add data augmentation to improve accuracy
- [x] Increase network size (ResNet-18) and test with 40 epochs
- [x] Add learning rate scheduling (OneCycleLR) to improve convergence
- ~[ ] Compile PyTorch stable/release locally to test for speed and accuracy improvements~ (no longer pursuing CPU benchmarking)
- [ ] Implement various seeds to find optimal accuracy

> **Note:** CPU benchmarking has been retired. All future training and testing will use MPS exclusively.
