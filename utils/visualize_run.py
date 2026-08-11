import argparse
import json
import os
import struct
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to runs/N directory")
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.json")
    losses_path = os.path.join(args.run_dir, "losses.bin")

    if not os.path.exists(losses_path):
        print(f"Error: {losses_path} not found.")
        return

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            print(f"Config: {json.dumps(config, indent=2)}")

    epochs = []
    costs = []
    losses = []
    
    epoch_costs = defaultdict(list)
    epoch_losses = defaultdict(list)
    
    # Read the binary data (uint32 epoch, uint32 worker, float cost, float loss)
    with open(losses_path, "rb") as f:
        while chunk := f.read(16):
            if len(chunk) < 16:
                break
            ep, wid, cost, loss = struct.unpack("<IIff", chunk)
            epochs.append(ep)
            costs.append(cost)
            losses.append(loss)
            epoch_costs[ep].append(cost)
            epoch_losses[ep].append(loss)

    if not epochs:
        print("No data to plot.")
        return

    # Compute worker averages per epoch
    unique_epochs = sorted(epoch_costs.keys())
    avg_costs = [sum(epoch_costs[ep]) / len(epoch_costs[ep]) for ep in unique_epochs]
    avg_losses = [sum(epoch_losses[ep]) / len(epoch_losses[ep]) for ep in unique_epochs]

    plt.figure(figsize=(12, 5))

    # Plot Cost
    plt.subplot(1, 2, 1)
    plt.scatter(epochs, costs, alpha=0.6, s=4, c='lightblue', label='Worker Samples')
    plt.plot(unique_epochs, avg_costs, color='blue', linewidth=2, label='Worker Average')
    plt.title("Execution Cost (Makespan) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cost (ms)")
    plt.grid(True)
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.scatter(epochs, losses, alpha=0.6, s=4, c='salmon', label='Worker Samples')
    plt.plot(unique_epochs, avg_losses, color='red', linewidth=2, label='Worker Average')
    plt.title("A2C Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    out_file = os.path.join(args.run_dir, "training_metrics.png")
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")
    plt.show()

if __name__ == "__main__":
    main()