"""
Generates the accuracy-vs-epsilon plot and a training-time overhead plot
from results/accuracy_vs_epsilon.csv.

Usage:
    python plot_results.py
"""

import csv
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "accuracy_vs_epsilon.csv")


def load_results():
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    rows = load_results()
    baseline = next((r for r in rows if r["run"] == "baseline"), None)
    dp_rows = [r for r in rows if r["run"] != "baseline"]
    dp_rows.sort(key=lambda r: float(r["epsilon"]))

    epsilons = [float(r["epsilon"]) for r in dp_rows]
    accuracies = [float(r["accuracy"]) for r in dp_rows]
    times = [float(r["train_time_sec"]) for r in dp_rows]

    # --- Plot 1: Accuracy vs Epsilon ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilons, accuracies, marker="o", linewidth=2, label="DP-SGD")
    if baseline:
        ax.axhline(
            float(baseline["accuracy"]),
            color="gray",
            linestyle="--",
            label=f"Non-private baseline ({float(baseline['accuracy']):.3f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget (ε), log scale")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. Privacy Budget (MNIST, DP-SGD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_epsilon.png"), dpi=150)
    print("Saved results/accuracy_vs_epsilon.png")

    # --- Plot 2: Training time overhead ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(epsilons, times, marker="s", color="firebrick", label="DP-SGD train time")
    if baseline:
        ax2.axhline(
            float(baseline["train_time_sec"]),
            color="gray",
            linestyle="--",
            label=f"Baseline train time ({float(baseline['train_time_sec']):.1f}s)",
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Privacy budget (ε), log scale")
    ax2.set_ylabel("Training time (seconds)")
    ax2.set_title("Compute Overhead of DP-SGD vs. Epsilon")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "train_time_vs_epsilon.png"), dpi=150)
    print("Saved results/train_time_vs_epsilon.png")


if __name__ == "__main__":
    main()
