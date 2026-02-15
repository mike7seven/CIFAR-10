"""
Data loading for CIFAR-10/100 using mlx-data.

Training pipeline includes augmentation (random horizontal flip, random crop
with padding) matching the PyTorch implementation. Test pipeline applies
normalization only.
"""

import mlx.core as mx
from mlx.data.datasets import load_cifar10, load_cifar100


def get_cifar(batch_size, dataset="cifar10", root=None):
    """Load CIFAR-10 or CIFAR-100 and return (train_iter, test_iter)."""
    load_fn = load_cifar100 if dataset == "cifar100" else load_cifar10
    num_classes = 100 if dataset == "cifar100" else 10

    tr = load_fn(root=root, train=True)
    test = load_fn(root=root, train=False)

    def normalize(x):
        # Convert to float32 [0,1], then normalize to [-1,1]
        # Matches PyTorch: Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        return (x.astype("float32") / 255.0 - 0.5) / 0.5

    # Training iterator with augmentation
    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)        # pad height: 4px top, 4px bottom
        .pad("image", 1, 4, 4, 0.0)        # pad width: 4px left, 4px right
        .image_random_crop("image", 32, 32)  # random 32x32 crop
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    # Test iterator without augmentation
    test_iter = (
        test.to_stream()
        .key_transform("image", normalize)
        .batch(batch_size)
    )

    return tr_iter, test_iter, num_classes


# CIFAR-100 fine-grained class names (alphabetical order matching torchvision)
CIFAR100_CLASSES = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm',
)

CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)


def get_class_names(dataset="cifar10"):
    """Return class name tuple for the given dataset."""
    return CIFAR100_CLASSES if dataset == "cifar100" else CIFAR10_CLASSES
