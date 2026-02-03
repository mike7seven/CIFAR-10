# Training Results

> **Hardware:** Tests performed on Apple M4 MacBook Pro with MPS acceleration.

## Table of Contents

- [Small CNN (~62K params)](#small-cnn-62k-params)
- [Large CNN (1.98M params)](#large-cnn-198m-params)
  - [MPS vs CPU Comparison](#mps-vs-cpu-comparison-2-epochs)
  - [MPS Training Progression](#mps-training-progression)
  - [Per-Class Accuracy by Epoch](#per-class-accuracy-by-epoch)
- [Reproducibility (Seed 1111)](#reproducibility-seed-1111)
- [Data Augmentation](#data-augmentation-seed-1111)
- [Architecture Comparison](#architecture-comparison)

---

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

## Data Augmentation (Seed 1111)

Large CNN, MPS with RandomCrop(32, padding=4) and RandomHorizontalFlip

| Metric | No Aug (20 epochs) | With Aug (20 epochs) | With Aug (40 epochs) |
|--------|-------------------|----------------------|----------------------|
| **Time** | 98.55s | 225.20s | 447.58s |
| **Overall Accuracy** | 71% | 70% | **78%** |
| **Final Loss** | 0.573 | 0.905 | 0.618 |

### Per-Class Accuracy Comparison

| Class | No Aug (20 ep) | Aug (20 ep) | Aug (40 ep) | Δ (40 vs No Aug) |
|-------|----------------|-------------|-------------|------------------|
| plane | 84.4% | 83.0% | 81.0% | -3.4 |
| car   | 77.8% | 71.5% | 89.8% | **+12.0** |
| bird  | 55.7% | 68.0% | 70.5% | **+14.8** |
| cat   | 65.9% | 55.0% | 62.1% | -3.8 |
| deer  | 57.6% | 59.4% | 68.5% | **+10.9** |
| dog   | 53.7% | 57.8% | 75.4% | **+21.7** |
| frog  | 72.7% | 75.0% | 87.0% | **+14.3** |
| horse | 84.9% | 70.3% | 78.0% | -6.9 |
| ship  | 80.1% | 79.7% | 90.2% | **+10.1** |
| truck | 82.3% | 86.3% | 80.2% | -2.1 |

**Key findings:** Data augmentation with 40 epochs achieved **78% accuracy** (+7% over baseline). Dog class improved most (+21.7%).

### Why Some Classes Decreased

Plane (-3.4%), cat (-3.8%), horse (-6.9%), and truck (-2.1%) dropped despite overall improvement. Contributing factors:

1. **Class confusion from augmentation:** Horizontal flips can make planes look like birds, horses like deer, and trucks like cars.

2. **Decision boundary shifts:** Large gains in dog (+21.7%), bird (+14.8%), and frog (+14.3%) shift decision boundaries, causing previously "easy" classes to lose edge cases.

3. **Feature generalization trade-off:** Without augmentation, the model memorizes specific patterns. With augmentation, it learns general features that improve overall accuracy but lose class-specific shortcuts.

4. **Test set mismatch:** Training sees augmented images, but test images are unaugmented. Orientation-sensitive classes may suffer.

**Bottom line:** Overall accuracy improved 71% → 78%. Gains in difficult classes outweigh small losses in easier ones.

### Hypothesis: Larger Model Should Improve All Classes

A larger network (e.g., ResNet-18) should mitigate the per-class accuracy drops because:

1. **More capacity** to learn both general and class-specific features simultaneously
2. **Deeper representations** can handle augmentation variations without losing orientation-sensitive patterns
3. **Residual connections** help preserve fine-grained features through the network

**Next experiment:** Increase network size and verify uniform accuracy improvement across all classes.

---

## Acceptable Accuracy Targets

| Model | Target Accuracy | Status |
|-------|-----------------|--------|
| Small CNN (~62K params) | ≥50% | ✓ Achieved (54%) |
| Large CNN (1.98M params) | ≥70% | ✓ Achieved (78%) |
| Large CNN + Augmentation | ≥75% | ✓ Achieved (78%) |
| ResNet-18 (~11M params) | ≥90% | Pending |

**Project goal:** Demonstrate MPS acceleration benefits while achieving competitive accuracy on CIFAR-10. Current results (78%) exceed baseline expectations for a simple CNN architecture.

---

## Epoch Training and Diminishing Returns

Our experiments showed accuracy improvements with additional epochs:

| Epochs | Accuracy | Δ from Previous |
|--------|----------|-----------------|
| 2      | 39%      | -               |
| 10     | 63%      | +24%            |
| 20     | 71%      | +8%             |
| 40 (aug) | 78%    | +7%             |

### Industry Research on Optimal Epochs

Research on CIFAR-10 training indicates diminishing returns beyond certain epoch thresholds:

| Epoch Range | Expected Accuracy | Notes |
|-------------|-------------------|-------|
| 20-30       | ~92-93%          | With learning rate scheduling |
| 40-50       | ~93-94%          | [FPT Software achieved 94% at 50 epochs](https://fptsoftware.com/resource-center/blogs/cifar10-94-of-accuracy-by-50-epochs-with-end-to-end-training) |
| 50-100      | ~94-95%          | Peak accuracy range |
| 100-200     | ~95%             | Marginal 0.5-1% improvement |
| 200+        | Diminishing      | Risk of overfitting |

**Key insight:** With modern techniques (OneCycleLR, proper augmentation), [94%+ accuracy is achievable in ~50 epochs](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html). Training beyond 100-200 epochs typically yields minimal gains relative to compute cost.

**Our gap:** Our 78% at 40 epochs suggests room for improvement through:
1. Learning rate scheduling (currently fixed)
2. Larger network architecture
3. Additional regularization (dropout, weight decay)

Sources: [ResearchGate - CIFAR-10 accuracy vs epochs](https://www.researchgate.net/figure/CIFAR-10-and-test-accuracies-over-100-epochs-SGD-with-a-fixed-step-size-and-Adam-for_fig4_327592152), [arXiv - 94% in 3.29s](https://arxiv.org/html/2404.00498v2)

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
