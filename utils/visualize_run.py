#!/usr/bin/env python3
import argparse
import json
import os
import struct

import matplotlib.pyplot as plt


def read_binary_metrics(file_path):
    indices = []
    values = []
    record_size = struct.calcsize("<If")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            while chunk := f.read(record_size):
                if len(chunk) < record_size:
                    break
                idx, val = struct.unpack("<If", chunk)
                indices.append(idx)
                values.append(val)
    return indices, values


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

    if not loss_batches and not cost_indices:
        print("No data to plot.")
        return

    num_plots = (1 if loss_batches else 0) + (1 if cost_indices else 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), squeeze=False)
    plot_idx = 0

    if loss_batches:
        ax = axes[plot_idx, 0]
        ax.set_yscale("log")
        ax.plot(loss_batches, losses, color="red", linewidth=1.5, label="Total Loss")
        ax.scatter(loss_batches, losses, alpha=0.4, s=10, c="salmon")
        ax.set_title("Learner Loss over Optimization Batches")
        ax.set_xlabel("Batch / Step")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()
        plot_idx += 1

    if cost_indices:
        ax = axes[plot_idx, 0]
        ax.set_yscale("log")
        ax.plot(
            cost_indices,
            costs,
            color="blue",
            linewidth=1.5,
            label="Extracted Cost (ms)",
        )
        ax.scatter(cost_indices, costs, alpha=0.4, s=10, c="skyblue")
        ax.set_title("Extracted Cost over Episodes")
        ax.set_xlabel("Episode / Sample")
        ax.set_ylabel("Cost (ms)")
        ax.grid(True)
        ax.legend()
        plot_idx += 1

    plt.tight_layout()
    out_file = os.path.join(args.run_dir, "training_metrics.png")
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")


if __name__ == "__main__":
    main()
