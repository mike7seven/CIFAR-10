# CIFAR-10 Classifier

PyTorch CNN trained on CIFAR-10 dataset using MPS (Apple Metal) acceleration.

## Usage

```bash
uv run python cifar-tutorial.py
```

## Current Results

- **Accuracy:** 71% (20 epochs, seed 1111)
- **Training time:** ~99 seconds on MPS
- **Network:** 1.98M parameters

## Future Improvements

- [ ] Add data augmentation to improve accuracy
