# Training Results

> **Hardware:** Tests performed on Apple M4 MacBook Pro with MPS acceleration.

## Table of Contents

- [Small Custom LeNet-style CNN (~62K params)](#small-custom-lenet-style-cnn-62k-params)
- [Large Custom LeNet-style CNN (1.98M params)](#large-custom-lenet-style-cnn-198m-params)
  - [MPS vs CPU Comparison](#mps-vs-cpu-comparison-2-epochs)
  - [MPS Training Progression](#mps-training-progression)
  - [Per-Class Accuracy by Epoch](#per-class-accuracy-by-epoch)
- [Reproducibility (Seed 1111)](#reproducibility-seed-1111)
- [Data Augmentation](#data-augmentation-seed-1111)
- [ResNet-18 (~11.2M params)](#resnet-18-112m-params)
- [Architecture Comparison](#architecture-comparison)
- [Hardware DNA](#hardware-dna)
- [Activation Visualization](#activation-visualization)

---

## Small Custom LeNet-style CNN (~62K params)

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

## Large Custom LeNet-style CNN (1.98M params)

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

Large Custom LeNet-style CNN, MPS, 20 Epochs

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

Large Custom LeNet-style CNN, MPS with RandomCrop(32, padding=4) and RandomHorizontalFlip

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

## Learning Rate Scheduling (Seed 1111, 40 Epochs)

Large Custom LeNet-style CNN with OneCycleLR (max_lr=0.1, cosine annealing, 30% warmup) + weight decay (5e-4)

| Metric | Fixed LR | OneCycleLR | Improvement |
|--------|----------|------------|-------------|
| **Overall Accuracy** | 78% | **85%** | **+7%** |
| **Final Loss** | 0.618 | 0.314 | -49% |
| **Time** | 447.58s | 450.70s | ~same |

### Per-Class Accuracy Comparison

| Class | Fixed LR | OneCycleLR | Δ |
|-------|----------|------------|---|
| plane | 81.0% | 89.0% | +8.0 |
| car   | 71.5% | 92.7% | **+21.2** |
| bird  | 68.0% | 80.7% | +12.7 |
| cat   | 55.0% | 71.9% | **+16.9** |
| deer  | 59.4% | 86.0% | **+26.6** |
| dog   | 57.8% | 76.9% | **+19.1** |
| frog  | 75.0% | 89.6% | +14.6 |
| horse | 70.3% | 89.7% | **+19.4** |
| ship  | 79.7% | 92.3% | +12.6 |
| truck | 86.3% | 90.8% | +4.5 |

**Key findings:** OneCycleLR improved all 10 classes. Deer showed largest gain (+26.6%). All classes now exceed 70% accuracy.

---

## Acceptable Accuracy Targets

| Model | Target Accuracy | Status |
|-------|-----------------|--------|
| Small LeNet-style CNN (~62K params) | ≥50% | ✓ Achieved (54%) |
| Large LeNet-style CNN (1.98M params) | ≥70% | ✓ Achieved (78%) |
| Large LeNet-style CNN + Augmentation | ≥75% | ✓ Achieved (78%) |
| Large LeNet-style CNN + Augmentation + LR Scheduling | ≥85% | ✓ Achieved (85%) |
| ResNet-18 (~11M params) | ≥90% | ✓ Achieved (92%) |

**Project goal:** Demonstrate MPS acceleration benefits while achieving competitive accuracy on CIFAR-10. Current results (92%) with ResNet-18 confirm the value of deeper architectures.

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

**Our gap closed:** Implementing OneCycleLR improved accuracy from 78% to **85%**. Upgrading to ResNet-18 further improved accuracy to **92%**.

Sources: [ResearchGate - CIFAR-10 accuracy vs epochs](https://www.researchgate.net/figure/CIFAR-10-and-test-accuracies-over-100-epochs-SGD-with-a-fixed-step-size-and-Adam-for_fig4_327592152), [arXiv - 94% in 3.29s](https://arxiv.org/html/2404.00498v2)

---

## ResNet-18 (~11.2M params)

ResNet-18 adapted for CIFAR-10 (3x3 conv1, no maxpool, 10-class FC) with OneCycleLR + augmentation.

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **92%** |
| **Training Time** | 1062.31s |
| **Final Loss** | 0.031 |
| **Device** | MPS |
| **Epochs** | 40 |
| **Seed** | 1111 |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| car   | 96.0% |
| horse | 95.8% |
| ship  | 95.6% |
| truck | 95.6% |
| frog  | 95.2% |
| deer  | 93.6% |
| plane | 92.9% |
| bird  | 89.8% |
| dog   | 87.4% |
| cat   | 85.6% |

### Comparison vs Large LeNet-style CNN + OneCycleLR

| Class | Large CNN (85%) | ResNet-18 (92%) | Δ |
|-------|----------------|-----------------|---|
| plane | 89.0% | 92.9% | +3.9 |
| car   | 92.7% | 96.0% | +3.3 |
| bird  | 80.7% | 89.8% | **+9.1** |
| cat   | 71.9% | 85.6% | **+13.7** |
| deer  | 86.0% | 93.6% | **+7.6** |
| dog   | 76.9% | 87.4% | **+10.5** |
| frog  | 89.6% | 95.2% | +5.6 |
| horse | 89.7% | 95.8% | +6.1 |
| ship  | 92.3% | 95.6% | +3.3 |
| truck | 90.8% | 95.6% | +4.8 |

**Key findings:** ResNet-18 improved all 10 classes. Cat showed the largest gain (+13.7%), confirming the hypothesis that a deeper network with residual connections mitigates per-class accuracy drops from augmentation. All classes now exceed 85%.

---

## Architecture Comparison

| Model | Params | Best Accuracy | Key Features |
|-------|--------|---------------|--------------|
| Small LeNet-style CNN (tutorial) | ~62K | 54% | Original architecture, batch size 4 |
| Large LeNet-style CNN | ~1.98M | 78% | Wider conv layers (64, 128), batch size 64 |
| Large LeNet-style CNN + OneCycleLR | ~1.98M | 85% | + LR scheduling, weight decay |
| **ResNet-18** | **~11.2M** | **92%** | Residual connections, adapted for 32x32 |

---

## Hardware DNA

Each training run captures a hardware "fingerprint" saved to `run_log.json`. Floating-point operations are non-associative: `(a + b) + c != a + (b + c)`. GPU backends (MPS, CUDA) parallelize reductions in non-deterministic order, so accumulated rounding differences across millions of backpropagation steps produce divergent final weights on different hardware — even with identical seeds.

### Apple M4 Max Run

| Field | Value |
|-------|-------|
| OS | Darwin |
| Architecture | arm64 |
| Chip | Apple M4 Max |
| PyTorch | 2.10.0 |
| Device | MPS |
| Seed | 1111 |
| Training Time | 1038.28s |
| Overall Accuracy | 92% |
| Model Params | 11,173,962 |

The `run_log.json` includes full per-class accuracy and training config for cross-machine comparison. Running the same seed on CUDA or CPU hardware will produce different weight tensors due to the non-associative floating-point accumulation order.

---

## Activation Visualization

`visualize_activations.py` hooks into four layers of the ResNet-18 during a forward pass, saving feature map grids as PNGs to `activations/`.

| Layer | Channels | Resolution | What It Detects |
|-------|----------|------------|-----------------|
| conv1 | 64 | 32x32 | Raw edges and color gradients |
| layer1 | 64 | 32x32 | Refined edges, basic shapes |
| layer2 | 128 | 16x16 | Mid-level features, object parts |
| layer3 | 256 | 8x8 | High-level abstract features, sparse and localized |

**Key observation:** As depth increases, activations become sparser and more spatially compressed — the network progressively distills raw pixel data into class-discriminative features. These feature maps are the hardware-specific "fingerprint" that would differ across MPS, CUDA, and CPU backends.

---

### LeNet-style CNN Layer Details (Small vs Large)

| Layer | Original Tutorial | Large CNN |
|-------|-------------------|---------|
| conv1 | 3→6 (5x5) | 3→64 (5x5) |
| conv2 | 6→16 (5x5) | 64→128 (5x5) |
| fc1   | 400→120 | 3200→512 |
| fc2   | 120→84 | 512→256 |
| fc3   | 84→10 | 256→10 |
| **Total Params** | **~62K** | **~1.98M (32x larger)** |
