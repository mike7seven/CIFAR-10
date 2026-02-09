# -*- coding: utf-8 -*-
"""
Visualize Activations
=====================

Uses PyTorch forward hooks to tap into the convolutional layers of a trained
CIFAR-10 ResNet-18 model during a forward pass.  Outputs a grid showing what
the M-series Mac is actually "seeing" at each stage of the network.

The key insight: because floating-point math is non-associative —
(a + b) + c != a + (b + c) — different hardware backends (MPS, CUDA, CPU)
accumulate rounding differences across millions of operations, producing
subtly different learned features.  This script lets you *see* those
hardware-specific representations.

Usage:
    uv run python visualize_activations.py                          # uses most recent .pth
    uv run python visualize_activations.py cifar_net_20260208.pth   # specific checkpoint
"""

import sys
import glob
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Model definition (must match the training script)
# ---------------------------------------------------------------------------

def cifar10_resnet18():
    """ResNet-18 adapted for CIFAR-10's 32x32 images."""
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net


# ---------------------------------------------------------------------------
# Activation capture via forward hooks
# ---------------------------------------------------------------------------

def visualize_model_activations(model, sample_image, device, class_name="",
                                output_dir="activations"):
    """
    Hooks into conv1, layer1, layer2, and layer3 to visualize feature maps
    at increasing depths of the ResNet-18. Saves figures as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks on key layers
    # conv1: raw edge/color detectors (64 channels, 32x32)
    # layer1: first residual block (64 channels, 32x32)
    # layer2: mid-level features (128 channels, 16x16)
    # layer3: higher-level features (256 channels, 8x8)
    hooks = [
        model.conv1.register_forward_hook(get_activation('conv1')),
        model.layer1.register_forward_hook(get_activation('layer1')),
        model.layer2.register_forward_hook(get_activation('layer2')),
        model.layer3.register_forward_hook(get_activation('layer3')),
    ]

    model.to(device)
    model.eval()

    input_tensor = sample_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    # Remove hooks
    for h in hooks:
        h.remove()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    pred_class = classes[predicted.item()]

    saved_files = []

    # Plot the original image first
    fig_orig, ax_orig = plt.subplots(1, 1, figsize=(3, 3))
    img = sample_image / 2 + 0.5  # unnormalize
    ax_orig.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    title = f'Predicted: {pred_class}'
    if class_name:
        title = f'Ground truth: {class_name} | {title}'
    ax_orig.set_title(title)
    ax_orig.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'original.png')
    fig_orig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig_orig)
    saved_files.append(path)
    print(f'  Saved {path}')

    # Plot activations for each hooked layer
    for layer_name, act in activations.items():
        channels = act.shape[1]
        num_show = min(channels, 16)  # show first 16 channels
        grid_size = int(np.ceil(np.sqrt(num_show)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.suptitle(
            f'{layer_name} activations ({channels} channels, '
            f'{act.shape[2]}x{act.shape[3]}) — {device}',
            fontsize=14
        )

        for i, ax in enumerate(axes.flat):
            if i < num_show:
                ax.imshow(act[0, i].cpu().numpy(), cmap='viridis')
                ax.set_title(f'Ch {i}', fontsize=8)
            ax.axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, f'{layer_name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)
        print(f'  Saved {path}')

    return saved_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_checkpoint():
    """Find the most recent .pth file in the project directory."""
    checkpoints = glob.glob('cifar_net_*.pth')
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def main():
    # Resolve checkpoint path
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    else:
        checkpoint = find_latest_checkpoint()

    if checkpoint is None or not os.path.exists(checkpoint):
        print('No checkpoint found. Train the model first:')
        print('  uv run python cifar-tutorial.py')
        sys.exit(1)

    print(f'Loading checkpoint: {checkpoint}')

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load model
    model = cifar10_resnet18()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))

    # Load a test image
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Pick a random test image
    idx = torch.randint(len(testset), (1,)).item()
    image, label = testset[idx]
    print(f'Sample #{idx}: ground truth = {classes[label]}')

    visualize_model_activations(model, image, device, class_name=classes[label])


if __name__ == '__main__':
    main()
