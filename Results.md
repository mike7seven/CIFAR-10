# Training Results

## Small CNN (~62K params)

Original tutorial architecture with batch size 4.

### CPU, 2 Epochs

- **Overall accuracy:** 54%
- **Device:** cpu

| Class | Accuracy |
|-------|----------|
| frog  | 74.0%    |
| ship  | 67.6%    |
| car   | 65.8%    |
| truck | 65.2%    |
| horse | 55.1%    |
| bird  | 50.1%    |
| plane | 49.2%    |
| dog   | 43.5%    |
| deer  | 43.3%    |
| cat   | 35.1%    |

### MPS, 2 Epochs

- **Overall accuracy:** 52%
- **Device:** mps

---

## Large CNN (1.98M params)

Increased network width with batch size 64.

| Layer | Config |
|-------|--------|
| conv1 | 3→64 (5x5) |
| conv2 | 64→128 (5x5) |
| fc1   | 3200→512 |
| fc2   | 512→256 |
| fc3   | 256→10 |

### MPS vs CPU Comparison (2 Epochs)

| Device | Time | Accuracy | Speedup |
|--------|------|----------|---------|
| CPU    | 54.13s | 37% | 1x |
| MPS    | 12.01s | 39% | **4.5x faster** |

### MPS Training Progression

| Metric | 2 Epochs | 10 Epochs | 20 Epochs |
|--------|----------|-----------|-----------|
| **Time** | 12.01s | 48.21s | 112.43s |
| **Overall Accuracy** | 39% | 63% | 72% |
| **Final Loss** | ~1.72 | ~1.02 | ~0.60 |

### Per-Class Accuracy by Epoch

| Class | 10 Epochs | 20 Epochs |
|-------|-----------|-----------|
| plane | 68.5% | 82.2% |
| car   | 77.2% | 80.4% |
| bird  | 39.4% | 58.2% |
| cat   | 45.8% | 43.5% |
| deer  | 47.6% | 75.1% |
| dog   | 66.7% | 65.4% |
| frog  | 74.7% | 72.0% |
| horse | 65.3% | 80.5% |
| ship  | 80.7% | 80.8% |
| truck | 73.7% | 82.6% |

---

## Reproducibility (Seed 1111)

Large CNN, MPS, 20 Epochs

| Metric | Run 1 | Run 2 | Match |
|--------|-------|-------|-------|
| Loss [1, 100] | 2.300 | 2.300 | ✓ |
| Loss [10, 700] | 1.002 | 1.002 | ✓ |
| Loss [20, 700] | 0.573 | 0.573 | ✓ |
| **Overall Accuracy** | **71%** | **71%** | ✓ |

### Per-Class Accuracy (Seed 1111)

| Class | Accuracy |
|-------|----------|
| horse | 84.9% |
| plane | 84.4% |
| truck | 82.3% |
| ship  | 80.1% |
| car   | 77.8% |
| frog  | 72.7% |
| cat   | 65.9% |
| deer  | 57.6% |
| bird  | 55.7% |
| dog   | 53.7% |

---

## Data Augmentation (Seed 1111, 20 Epochs)

Large CNN, MPS with RandomCrop(32, padding=4) and RandomHorizontalFlip

| Metric | Without Augmentation | With Augmentation |
|--------|---------------------|-------------------|
| **Time** | 98.55s | 225.20s |
| **Overall Accuracy** | 71% | 70% |
| **Final Loss** | 0.573 | 0.905 |

### Per-Class Accuracy Comparison

| Class | Without Aug | With Aug | Δ |
|-------|-------------|----------|---|
| plane | 84.4% | 83.0% | -1.4 |
| car   | 77.8% | 71.5% | -6.3 |
| bird  | 55.7% | 68.0% | **+12.3** |
| cat   | 65.9% | 55.0% | -10.9 |
| deer  | 57.6% | 59.4% | +1.8 |
| dog   | 53.7% | 57.8% | +4.1 |
| frog  | 72.7% | 75.0% | +2.3 |
| horse | 84.9% | 70.3% | -14.6 |
| ship  | 80.1% | 79.7% | -0.4 |
| truck | 82.3% | 86.3% | **+4.0** |

Note: Data augmentation typically requires more epochs to show full benefit. The higher loss at epoch 20 indicates the model is still learning.

---

## Architecture Comparison

| Layer | Original Tutorial | Current |
|-------|-------------------|---------|
| conv1 | 3→6 (5x5) | 3→64 (5x5) |
| conv2 | 6→16 (5x5) | 64→128 (5x5) |
| fc1   | 400→120 | 3200→512 |
| fc2   | 120→84 | 512→256 |
| fc3   | 84→10 | 256→10 |
| **Total Params** | **~62K** | **~1.98M (32x larger)** |
