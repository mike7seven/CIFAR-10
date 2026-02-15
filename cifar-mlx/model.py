"""
CNN model definitions for CIFAR-10/100 in MLX.

Provides two architectures:
  - SmallCNN: ~0.5M param CNN with residual connections (fast experiments)
  - ResNet18: ~11.2M param ResNet-18 adapted for 32x32 images (best accuracy)

All models expect NHWC input format: (batch, height, width, channels).
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


# ---------------------------------------------------------------------------
# Small CNN (~0.5M params)
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """6-conv CNN with residual connections, BatchNorm, and dropout."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: 32x32x3 → 16x16x32
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(32)
        self.skip1 = nn.Conv2d(32, 32, 1, stride=1, padding=0)

        # Block 2: 16x16x32 → 8x8x64
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm(64)
        self.skip2 = nn.Conv2d(64, 64, 1, stride=1, padding=0)

        # Block 3: 8x8x64 → 4x4x128
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm(128)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm(128)
        self.skip3 = nn.Conv2d(128, 128, 1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(0.25)

        # FC head: 4*4*128 = 2048
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x):
        # Block 1
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.skip1(x) + nn.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)

        # Block 2
        x = nn.relu(self.bn3(self.conv3(x)))
        x = self.skip2(x) + nn.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.drop(x)

        # Block 3
        x = nn.relu(self.bn5(self.conv5(x)))
        x = self.skip3(x) + nn.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.drop(x)

        # FC head
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# ResNet-18 for CIFAR (adapted for 32x32 images)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs with residual connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels)

        # Shortcut for dimension mismatch
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm(out_channels),
            )
        else:
            self.shortcut = lambda x: x

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = nn.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR 32x32 images.

    Uses 3x3 initial conv (not 7x7) and no initial maxpool,
    matching the PyTorch CIFAR adaptation.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Initial conv: 3x3 stride-1 (not 7x7 stride-2)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(64)

        # 4 layer groups: [64, 128, 256, 512]
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        # No maxpool (CIFAR adaptation)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global average pool: NHWC → average over H,W dims (axes 1,2)
        x = mx.mean(x, axis=(1, 2))
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(name="small-cnn", num_classes=10):
    """Create a model by name."""
    if name == "small-cnn":
        return SmallCNN(num_classes=num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_params(model):
    """Count total trainable parameters."""
    return sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
