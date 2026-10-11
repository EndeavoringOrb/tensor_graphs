"""Train the LNS neighborhood-selection GNN."""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    # Permit ``python lns_selectors/gnn/train.py ...`` as well as ``-m``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lns_selectors.gnn.common import loadRun, metadataGraph
    from lns_selectors.gnn.gnn import NeighborhoodGnn
else:
    from .common import loadRun, metadataGraph
    from .gnn import NeighborhoodGnn


def loadSamples(path):
    """Load JSONL samples, a run directory, or a legacy JSON/torch file."""
    path = Path(path)
    if path.is_dir() or path.suffix.lower() in {".json", ".jsonl"}:
        samples, _ = loadRun(path)
    else:
        samples = torch.load(path, map_location="cpu")
    if isinstance(samples, dict) and isinstance(samples.get("problems"), dict):
        flattened = []
        for problem in samples["problems"].values():
            flattened.extend(
                problem if isinstance(problem, list) else problem.get("samples", [])
            )
        samples = flattened
    if not isinstance(samples, list) or not samples:
        raise ValueError("The training sample file must contain a non-empty list")
    return samples


def problemMetadata(problem_data):
    """Build metadata once so compact samples can share one graph."""
    from ortools_cp_model import OrtoolsSolver

    model = OrtoolsSolver(copy.deepcopy(problem_data))
    model.buildModel()
    return model.getNeighborhoodMetadata()


def sampleTensors(sample, device, metadata=None):
    if "node_features" in sample:
        node_features = torch.tensor(
            sample["node_features"], dtype=torch.float32, device=device
        )
        edges = sample.get("edge_index", [[], []])
    else:
        if metadata is None:
            raise ValueError("Compact samples require the run's problem.json")
        keys, node_features_data, edges = metadataGraph(metadata)
        if not keys or not node_features_data:
            raise ValueError("The problem metadata does not contain a usable graph")
        node_features = torch.tensor(
            node_features_data, dtype=torch.float32, device=device
        )

    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    elif edge_index.ndim == 2 and edge_index.shape[1] != 2 and edge_index.shape[0] == 2:
        edge_index = edge_index.contiguous()
    elif edge_index.ndim == 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    else:
        raise ValueError("edge_index must be a 2xE array or an Ex2 array")

    neighborhood = torch.zeros(
        node_features.shape[0], dtype=torch.float32, device=device
    )
    for index in sample.get(
        "neighborhood_indices",
        sample.get("target_indices", sample.get("neighborhood", [])),
    ):
        if 0 <= int(index) < neighborhood.numel():
            neighborhood[int(index)] = 1.0
    relative_runtime = float(
        sample.get(
            "relative_runtime",
            sample.get(
                "runtime_ratio",
                sample.get("target", 1.0 - float(sample.get("reward", 0.0))),
            ),
        )
    )
    target = torch.tensor(
        max(0.0, min(1.0, relative_runtime)), dtype=torch.float32, device=device
    )
    return node_features, edge_index, neighborhood, target


def trainGnn(
    samples,
    output_path,
    epochs=20,
    learning_rate=1e-3,
    hidden_dim=64,
    layers=3,
    device=None,
    seed=0,
    problem_data=None,
):
    if not samples:
        raise ValueError("At least one LNS training sample is required")
    if not samples[0].get("node_features") and problem_data is None:
        raise ValueError("Compact samples require the run's problem.json")

    has_compact_samples = any(not sample.get("node_features") for sample in samples)
    metadata = (
        problemMetadata(problem_data)
        if problem_data is not None and has_compact_samples
        else None
    )
    random_generator = random.Random(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if metadata is not None:
        feature_dim = int(metadata["feature_dim"])
    else:
        feature_dim = len(samples[0]["node_features"][0])
    model = NeighborhoodGnn(feature_dim, hidden_dim, layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(max(1, int(epochs))):
        shuffled_samples = list(samples)
        random_generator.shuffle(shuffled_samples)
        model.train()
        total_loss = 0.0

        for sample in shuffled_samples:
            node_features, edge_index, neighborhood, target = sampleTensors(
                sample, device, metadata
            )
            if node_features.shape[1] != feature_dim:
                raise ValueError("All samples must use the same feature dimension")
            prediction = model.predictRuntime(node_features, edge_index, neighborhood)
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        mean_loss = total_loss / len(shuffled_samples)
        print(f"[train] epoch={epoch + 1:04d} loss={mean_loss:.6e}")

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
    parser.add_argument(
        "samples",
        help="Sample run directory, samples.jsonl, JSON, or torch-serialized samples",
    )
    parser.add_argument("--output", default="runs/lns_gnn.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sample_path = Path(args.samples)
    if sample_path.is_dir() or sample_path.suffix.lower() in {".json", ".jsonl"}:
        sample_list, problem_data = loadRun(sample_path)
    else:
        sample_list, problem_data = loadSamples(sample_path), None
    trainGnn(
        sample_list,
        args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        device=args.device,
        seed=args.seed,
        problem_data=problem_data,
    )


if __name__ == "__main__":
    main()
