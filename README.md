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

- **Accuracy:** 71% (20 epochs, seed 1111)
- **Network:** 1.98M parameters
- **Architecture:** 2 conv layers (64, 128 channels) + 3 FC layers

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

## Future Improvements

- [ ] Add data augmentation to improve accuracy
