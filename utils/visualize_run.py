#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def read_binary_metrics(file_path):
    """Fast binary loader using NumPy structured arrays."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return np.array([]), np.array([])

    # <I = uint32 (4 bytes), <f = float32 (4 bytes) -> 8 bytes per record
    dt = np.dtype([("idx", "<u4"), ("val", "<f4")])
    data = np.fromfile(file_path, dtype=dt)
    return data["idx"], data["val"]


def downsample_for_plotting(x, y, max_points=20_000):
    """
    Min-max downsampler: preserves spikes and extremes while keeping
    the total point count small enough for instant rendering.
    """
    n = len(x)
    if n <= max_points:
        return x, y

    bucket_size = int(np.ceil(n / (max_points // 2)))
    # Trim to full buckets
    n_full = (n // bucket_size) * bucket_size
    x_view = x[:n_full].reshape(-1, bucket_size)
    y_view = y[:n_full].reshape(-1, bucket_size)

    min_indices = np.argmin(y_view, axis=1)
    max_indices = np.argmax(y_view, axis=1)
    row_idx = np.arange(x_view.shape[0])

    # Interleave min and max points in chronological order
    take_idx = np.column_stack([
        np.minimum(min_indices, max_indices),
        np.maximum(min_indices, max_indices),
    ])

    x_down = np.take_along_axis(x_view, take_idx, axis=1).ravel()
    y_down = np.take_along_axis(y_view, take_idx, axis=1).ravel()
    return x_down, y_down


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training metrics (loss and extracted cost) from run directory"
    )
    parser.add_argument("run_dir", type=str, help="Path to runs/N directory")
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.json")
    losses_path = os.path.join(args.run_dir, "losses.bin")
    costs_path = os.path.join(args.run_dir, "costs.bin")

    if not os.path.exists(losses_path) and not os.path.exists(costs_path):
        print(f"Error: Neither {losses_path} nor {costs_path} found.")
        return

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            print(f"Config: {json.dumps(config, indent=2)}")

    loss_batches, losses = read_binary_metrics(losses_path)
    cost_indices, costs = read_binary_metrics(costs_path)

    has_losses = len(loss_batches) > 0
    has_costs = len(cost_indices) > 0

    if not has_losses and not has_costs:
        print("No data to plot.")
        return

    num_plots = int(has_losses) + int(has_costs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), squeeze=False)
    plot_idx = 0

    if has_losses:
        x_plot, y_plot = downsample_for_plotting(loss_batches, losses)
        # Filter non-positive values to prevent warnings with log scale
        valid = y_plot > 0
        x_plot, y_plot = x_plot[valid], y_plot[valid]

        ax = axes[plot_idx, 0]
        ax.set_yscale("log")
        # rasterized=True keeps file size small and rendering fast
        ax.plot(
            x_plot,
            y_plot,
            color="red",
            linewidth=1.0,
            label="Total Loss",
            rasterized=True,
        )
        ax.set_title(
            f"Learner Loss over Optimization Batches ({len(loss_batches):,} points)"
        )
        ax.set_xlabel("Batch / Step")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
        plot_idx += 1

    if has_costs:
        x_plot, y_plot = downsample_for_plotting(cost_indices, costs)
        valid = y_plot > 0
        x_plot, y_plot = x_plot[valid], y_plot[valid]

        ax = axes[plot_idx, 0]
        ax.set_yscale("log")
        ax.plot(
            x_plot,
            y_plot,
            color="blue",
            linewidth=1.0,
            label="Extracted Cost (ms)",
            rasterized=True,
        )
        ax.set_title(
            f"Extracted Cost over Episodes ({len(cost_indices):,} points)"
        )
        ax.set_xlabel("Episode / Sample")
        ax.set_ylabel("Cost (ms)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
        plot_idx += 1

    plt.tight_layout()
    out_file = os.path.join(args.run_dir, "training_metrics.png")
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_file}")


if __name__ == "__main__":
    main()