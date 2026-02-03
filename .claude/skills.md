# Skills

## train

Run CIFAR-10 training with MPS acceleration.

```bash
uv run python cifar-tutorial.py
```

## train-cpu

Run training on CPU for comparison.

```bash
USE_CPU=1 uv run python cifar-tutorial.py
```

## benchmark

Compare MPS vs CPU training performance by running both and comparing times.

## seed-search

Test different random seeds to find optimal accuracy:

```python
torch.manual_seed(SEED)
```

## add-augmentation

Add or modify data augmentation transforms:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip()
- ColorJitter (optional)
- RandomRotation (optional)
