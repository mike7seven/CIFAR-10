# Training Results

## MPS vs CPU Performance

| Device | Training Time (20 epochs) | Speedup |
|--------|---------------------------|---------|
| CPU    | 54.13s (2 epochs) | 1x |
| MPS    | 12.01s (2 epochs) | **4.5x faster** |

## MPS Training Runs Comparison

| Metric | 2 Epochs | 10 Epochs | 20 Epochs |
|--------|----------|-----------|-----------|
| **CNN Size** | 1.98M params | 1.98M params | 1.98M params |
| **Batch Size** | 64 | 64 | 64 |
| **Time** | 12.01s | 48.21s | 112.43s |
| **Overall Accuracy** | 39% | 63% | 72% |
| **Final Loss** | ~1.72 | ~1.02 | ~0.60 |

## Per-Class Accuracy

| Class | 2 Epochs | 10 Epochs | 20 Epochs |
|-------|----------|-----------|-----------|
| plane | 45.7% | 68.5% | 82.2% |
| car | 70.4% | 77.2% | 80.4% |
| bird | 43.5% | 39.4% | 58.2% |
| cat | 22.5% | 45.8% | 43.5% |
| deer | 28.1% | 47.6% | 75.1% |
| dog | 73.8% | 66.7% | 65.4% |
| frog | 57.4% | 74.7% | 72.0% |
| horse | 51.7% | 65.3% | 80.5% |
| ship | 75.1% | 80.7% | 80.8% |
| truck | 55.8% | 73.7% | 82.6% |

## Reproducibility Test (Seed 1111)

| Metric | Run 1 | Run 2 | Match |
|--------|-------|-------|-------|
| Loss [1, 100] | 2.300 | 2.300 | ✓ |
| Loss [10, 700] | 1.002 | 1.002 | ✓ |
| Loss [20, 700] | 0.573 | 0.573 | ✓ |
| **Overall Accuracy** | **71%** | **71%** | ✓ |

### Per-Class Accuracy (Seed 1111, 20 Epochs)

| Class | Accuracy |
|-------|----------|
| plane | 84.4% |
| car | 77.8% |
| bird | 55.7% |
| cat | 65.9% |
| deer | 57.6% |
| dog | 53.7% |
| frog | 72.7% |
| horse | 84.9% |
| ship | 80.1% |
| truck | 82.3% |

## CNN Architecture

| Layer | Original Tutorial | Current |
|-------|-------------------|---------|
| conv1 | 3→6 (5x5) | 3→64 (5x5) |
| conv2 | 6→16 (5x5) | 64→128 (5x5) |
| fc1 | 400→120 | 3200→512 |
| fc2 | 120→84 | 512→256 |
| fc3 | 84→10 | 256→10 |
| **Total Params** | **~62K** | **~1.98M (32x larger)** |
