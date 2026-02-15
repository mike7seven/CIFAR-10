"""
CIFAR-10/100 training script using Apple MLX.

Usage:
    uv run python train.py                                    # small CNN, CIFAR-10
    uv run python train.py --model resnet18 --scheduler onecycle  # ResNet-18 + scheduling
    uv run python train.py --dataset cifar100 --epochs 50     # CIFAR-100
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import dataset
import model


# ---------------------------------------------------------------------------
# Loss / eval functions
# ---------------------------------------------------------------------------

def loss_fn(net, X, y):
    logits = net(X)
    loss = mx.mean(nn.losses.cross_entropy(logits, y))
    acc = mx.mean(mx.argmax(logits, axis=1) == y)
    return loss, acc


def eval_fn(net, X, y):
    logits = net(X)
    return mx.argmax(logits, axis=1), y


# ---------------------------------------------------------------------------
# Hardware DNA
# ---------------------------------------------------------------------------

def get_hardware_context():
    context = {
        "OS": platform.system(),
        "OS_Version": platform.version(),
        "Architecture": platform.machine(),
        "Processor": platform.processor(),
        "MLX_Version": mx.__version__,
        "Device": "GPU" if mx.default_device() == mx.gpu else "CPU",
    }
    if platform.system() == "Darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
            context["Chip"] = brand
        except Exception:
            context["Chip"] = "Apple Silicon (Unknown)"
    return context


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # Seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Data
    tr_iter, test_iter, num_classes = dataset.get_cifar(
        args.batch_size, dataset=args.dataset
    )
    class_names = dataset.get_class_names(args.dataset)
    print(f"Dataset: {args.dataset.upper()} ({num_classes} classes)")

    # Model
    net = model.create_model(args.model, num_classes=num_classes)
    mx.eval(net.parameters())
    num_params = model.count_params(net)
    print(f"Model: {args.model} ({num_params / 1e6:.2f}M params)")

    # Optimizer + scheduler
    steps_per_epoch = 50000 // args.batch_size  # CIFAR train set size
    total_steps = args.epochs * steps_per_epoch

    if args.scheduler == "onecycle":
        warmup_steps = int(0.3 * total_steps)
        warmup = optim.linear_schedule(1e-3, 0.1, steps=warmup_steps)
        decay = optim.cosine_decay(0.1, total_steps - warmup_steps)
        lr_schedule = optim.join_schedules([warmup, decay], [warmup_steps])
        optimizer = optim.SGD(
            learning_rate=lr_schedule, momentum=0.9, weight_decay=5e-4
        )
        sched_desc = "OneCycleLR(max_lr=0.1, 30% warmup, cosine)"
    else:
        optimizer = optim.AdamW(learning_rate=args.lr)
        sched_desc = f"AdamW(lr={args.lr})"

    print(f"Scheduler: {sched_desc}")
    print(f"Training for {args.epochs} epochs...")

    loss_and_grad_fn = nn.value_and_grad(net, loss_fn)

    # --- Training loop ---
    start_time = time.time()

    for epoch in range(args.epochs):
        net.train()
        tr_iter.reset()
        running_loss = 0.0
        batch_count = 0

        for batch in tr_iter:
            X = mx.array(batch["image"])
            y = mx.array(batch["label"]).squeeze()

            (loss, acc), grads = loss_and_grad_fn(net, X, y)
            optimizer.update(net, grads)
            mx.eval(net.parameters(), optimizer.state)

            running_loss += loss.item()
            batch_count += 1

            if batch_count % 100 == 0:
                avg_loss = running_loss / batch_count
                print(f"  [{epoch + 1}, {batch_count:5d}] loss: {avg_loss:.3f}",
                      end="\r")

        avg_loss = running_loss / max(batch_count, 1)
        print(f"  Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    print(f"Finished training in {training_time:.2f}s")

    # --- Save weights ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_path = f"cifar_mlx_{timestamp}.npz"
    net.save_weights(weight_path)
    print(f"Weights saved to {weight_path}")

    # --- Evaluate ---
    net.eval()
    test_iter.reset()

    all_preds = []
    all_labels = []

    for batch in test_iter:
        X = mx.array(batch["image"])
        y = mx.array(batch["label"]).squeeze()
        preds, labels = eval_fn(net, X, y)
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = mx.concatenate(all_preds)
    all_labels = mx.concatenate(all_labels)
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size
    overall_acc = 100 * correct // total
    print(f"\nAccuracy on {total} test images: {overall_acc}%")

    # Per-class accuracy
    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        class_total = mask.sum().item()
        if class_total > 0:
            class_correct = ((all_preds == i) & mask).sum().item()
            acc = 100 * class_correct / class_total
            per_class_accuracy[name] = round(acc, 1)
            print(f"  {name:20s}: {acc:.1f}%")

    # --- Hardware DNA ---
    hardware = get_hardware_context()
    print(f"\n--- [ HARDWARE DNA ] ---")
    for k, v in hardware.items():
        print(f"  {k:16}: {v}")
    print("-" * 25)

    # --- Run log ---
    run_log = {
        "framework": "MLX",
        "hardware": hardware,
        "training": {
            "dataset": args.dataset,
            "num_classes": num_classes,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "model_params": num_params,
            "optimizer": sched_desc,
            "training_time_seconds": round(training_time, 2),
        },
        "results": {
            "overall_accuracy_pct": overall_acc,
            "per_class_accuracy_pct": per_class_accuracy,
        },
        "weights": weight_path,
        "timestamp": timestamp,
    }

    log_path = "run_log.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"Run log saved to {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR classifier with MLX")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"],
                        default="cifar10", help="Dataset (default: cifar10)")
    parser.add_argument("--model", choices=["small-cnn", "resnet18"],
                        default="small-cnn", help="Model architecture (default: small-cnn)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--seed", type=int, default=1111,
                        help="Random seed (default: 1111)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for AdamW (default: 3e-4)")
    parser.add_argument("--scheduler", choices=["none", "onecycle"],
                        default="none", help="LR scheduler (default: none)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU instead of GPU")
    parser.add_argument("--memory-limit", type=float, default=None,
                        help="Metal GPU memory limit in GB (e.g. 64 for 64GB)")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    if args.memory_limit is not None:
        limit_bytes = int(args.memory_limit * 1024**3)
        mx.set_memory_limit(limit_bytes)
        print(f"Metal memory limit set to {args.memory_limit:.0f}GB")

    print(f"Device: {mx.default_device()}")
    train(args)
