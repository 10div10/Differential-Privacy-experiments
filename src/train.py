"""
Differential Privacy experiments: accuracy vs. privacy budget (epsilon) on MNIST.

Trains:
  1. A non-private baseline CNN.
  2. Several DP-SGD models (via Opacus) targeting different epsilon values.

Outputs a CSV of results (results/accuracy_vs_epsilon.csv) and saves per-run
training curves for later plotting.

Usage:
    python train.py --epochs 10 --epsilons 0.5,1,3,8,15
    python train.py --baseline-only
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset choice: "mnist" (full 28x28 MNIST via torchvision, requires internet
# access to download on first run) or "digits" (scikit-learn's bundled 8x8
# digits dataset, ~1800 samples, no download needed -- useful for offline
# environments or a fast smoke test of the whole pipeline).
DATASET = os.environ.get("DP_DATASET", "mnist")


class SmallCNN(nn.Module):
    """Small CNN. Opacus requires GroupNorm instead of BatchNorm
    (BatchNorm mixes per-sample statistics across the batch, which breaks
    the per-sample gradient computation DP-SGD relies on)."""

    def __init__(self, in_hw=28):
        super().__init__()
        pooled = in_hw // 4
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * pooled * pooled, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def _fix_macos_ssl():
    """macOS's python.org / Homebrew Python builds don't automatically use
    the system CA trust store, which makes urllib SSL verification fail
    (CERTIFICATE_VERIFY_FAILED) even though the network connection itself
    is fine. Point Python at certifi's CA bundle if available."""
    import ssl
    try:
        import certifi
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass


def _mnist_loaders(batch_size):
    _fix_macos_ssl()
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader, 28


def _digits_loaders(batch_size):
    """Fallback dataset needing no internet: sklearn's bundled 8x8 digits."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    data = load_digits()
    X = data.images.astype("float32") / 16.0  # scale to [0, 1]
    X = (X - X.mean()) / (X.std() + 1e-8)
    y = data.target.astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    return train_loader, test_loader, 8


def get_loaders(batch_size):
    if DATASET == "digits":
        return _digits_loaders(batch_size)
    try:
        return _mnist_loaders(batch_size)
    except Exception as e:
        print(f"[warn] MNIST download failed ({e}); falling back to sklearn digits dataset.")
        return _digits_loaders(batch_size)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_baseline(epochs, batch_size, lr):
    print("\n=== Training non-private baseline ===")
    train_loader, test_loader, in_hw = get_loaders(batch_size)
    model = SmallCNN(in_hw).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        acc = evaluate(model, test_loader)
        print(f"  epoch {epoch+1}/{epochs}  test_acc={acc:.4f}")
    elapsed = time.time() - start

    final_acc = evaluate(model, test_loader)
    return {
        "run": "baseline",
        "epsilon": None,
        "delta": None,
        "noise_multiplier": None,
        "max_grad_norm": None,
        "accuracy": final_acc,
        "train_time_sec": round(elapsed, 1),
    }


def train_dp(target_epsilon, epochs, batch_size, lr, delta, max_grad_norm):
    print(f"\n=== Training DP-SGD model | target epsilon={target_epsilon} ===")
    train_loader, test_loader, in_hw = get_loaders(batch_size)
    model = SmallCNN(in_hw).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    start = time.time()
    for epoch in range(epochs):
        model.train()
        with BatchMemoryManager(
            data_loader=train_loader, max_physical_batch_size=128, optimizer=optimizer
        ) as memory_safe_loader:
            for x, y in memory_safe_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
        acc = evaluate(model, test_loader)
        eps_now = privacy_engine.get_epsilon(delta)
        print(f"  epoch {epoch+1}/{epochs}  test_acc={acc:.4f}  eps_so_far={eps_now:.2f}")
    elapsed = time.time() - start

    final_acc = evaluate(model, test_loader)
    final_epsilon = privacy_engine.get_epsilon(delta)
    return {
        "run": f"dp_eps_{target_epsilon}",
        "epsilon": round(final_epsilon, 3),
        "delta": delta,
        "noise_multiplier": round(optimizer.noise_multiplier, 4),
        "max_grad_norm": max_grad_norm,
        "accuracy": final_acc,
        "train_time_sec": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.2)
    parser.add_argument(
        "--epsilons", type=str, default="0.5,1,3,8,15",
        help="Comma-separated list of target epsilon values to sweep."
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--dp-only", action="store_true")
    args = parser.parse_args()

    results = []
    csv_path = os.path.join(RESULTS_DIR, "accuracy_vs_epsilon.csv")

    if not args.dp_only:
        results.append(train_baseline(args.epochs, args.batch_size, args.lr))

    if not args.baseline_only:
        for eps_str in args.epsilons.split(","):
            eps = float(eps_str)
            results.append(
                train_dp(eps, args.epochs, args.batch_size, args.lr, args.delta, args.max_grad_norm)
            )

    fieldnames = ["run", "epsilon", "delta", "noise_multiplier", "max_grad_norm", "accuracy", "train_time_sec"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults appended to {csv_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
