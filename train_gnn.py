"""Train the LNS neighborhood-selection GNN.

Samples are serialized dictionaries with at least:

``node_features``
    ``[num_nodes, feature_dim]`` floating-point features.
``edge_index``
    Either ``[[sources], [targets]]`` or a list of ``[source, target]`` pairs.
``target_indices``
    Node indices that belonged to the successful repair neighborhood.

An optional ``reward`` scales the loss for samples that produced larger
objective improvements.
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from lns_selectors.gnn import NeighborhoodGnn


def loadSamples(path):
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            samples = json.load(handle)
    else:
        samples = torch.load(path, map_location="cpu")
    if not isinstance(samples, list) or not samples:
        raise ValueError("The training sample file must contain a non-empty list")
    return samples


def sampleTensors(sample, device):
    node_features = torch.tensor(
        sample["node_features"], dtype=torch.float32, device=device
    )
    edges = sample.get("edge_index", [[], []])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    elif edge_index.ndim == 2 and edge_index.shape[1] != 2 and edge_index.shape[0] == 2:
        edge_index = edge_index.contiguous()
    elif edge_index.ndim == 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    else:
        raise ValueError("edge_index must be a 2xE array or an Ex2 array")

    target = torch.zeros(
        node_features.shape[0], dtype=torch.float32, device=device
    )
    for index in sample.get("target_indices", []):
        if 0 <= int(index) < target.numel():
            target[int(index)] = 1.0
    return node_features, edge_index, target


def trainGnn(
    samples,
    output_path,
    epochs=20,
    learning_rate=1e-3,
    hidden_dim=64,
    layers=3,
    device=None,
    seed=0,
):
    if not samples:
        raise ValueError("At least one LNS training sample is required")
    if not samples[0].get("node_features"):
        raise ValueError("Training samples must contain node_features")

    random_generator = random.Random(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = len(samples[0]["node_features"][0])
    model = NeighborhoodGnn(feature_dim, hidden_dim, layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(max(1, int(epochs))):
        shuffled_samples = list(samples)
        random_generator.shuffle(shuffled_samples)
        model.train()
        total_loss = 0.0

        for sample in shuffled_samples:
            node_features, edge_index, target = sampleTensors(sample, device)
            if node_features.shape[1] != feature_dim:
                raise ValueError("All samples must use the same feature dimension")
            logits = model(node_features, edge_index)
            positive = target.sum().clamp_min(1.0)
            negative = (target.numel() - target.sum()).clamp_min(1.0)
            pos_weight = (negative / positive).detach()
            loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight
            )
            reward = float(sample.get("reward", 0.0))
            loss = loss * (1.0 + max(0.0, reward))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        mean_loss = total_loss / len(shuffled_samples)
        print(f"[train_gnn] epoch={epoch + 1:04d} loss={mean_loss:.6e}")

    output_directory = os.path.dirname(os.path.abspath(output_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "layers": layers,
        },
        output_path,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", help="JSON or torch-serialized LNS samples")
    parser.add_argument("--output", default="runs/lns_gnn.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trainGnn(
        loadSamples(args.samples),
        args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
