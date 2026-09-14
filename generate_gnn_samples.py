"""Generate supervised samples for the LNS neighborhood-selection GNN.

The input is the same problem JSON accepted by ``ortools_lns.py``.  Each
sample contains the complete CP model's primary-decision graph, the current
incumbent objective, and labels for neighborhoods that produced an improving
repair.  Run this script before ``train_gnn.py`` when a selector checkpoint is
not available yet.

Example::

    .venvx64\\Scripts\\python.exe generate_gnn_samples.py \
        problem.json runs/lns_samples.json --samples 500
"""

import argparse
import copy
import json
import os
import random

from lns_selectors.random import RandomNeighborhoodSelector
from ortools_cp_model import OrtoolsSolver


def selectorMetadata(model, incumbent):
    metadata = copy.deepcopy(model.getNeighborhoodMetadata())
    assignments = incumbent.get("primary_assignments", {})
    for key, group in metadata.get("groups", {}).items():
        group["incumbent_value"] = assignments.get(key)
    return metadata


def isChangeable(group):
    return bool(group.get("selectable")) and int(group.get("choice_count", 0)) > 1


def closeNeighborhood(seeds, metadata, target_size, random_generator):
    """Expand seeds while keeping only groups that can actually change."""
    groups = metadata.get("groups", {})
    adjacency = metadata.get("adjacency", {})
    selected = {
        key for key in seeds if key in groups and isChangeable(groups[key])
    }
    frontier = list(selected)
    while frontier and len(selected) < target_size:
        current = frontier.pop()
        neighbors = list(adjacency.get(current, ()))
        random_generator.shuffle(neighbors)
        for neighbor in neighbors:
            if (
                neighbor in selected
                or neighbor not in groups
                or not isChangeable(groups[neighbor])
            ):
                continue
            selected.add(neighbor)
            frontier.append(neighbor)
            if len(selected) >= target_size:
                break
    return selected


def metadataSample(metadata, neighborhood, reward):
    """Convert selector metadata into the training-script sample schema."""
    groups = metadata.get("groups", {})
    keys = sorted(groups)
    key_indices = {key: index for index, key in enumerate(keys)}
    feature_dim = int(metadata.get("feature_dim", 0))
    features = []
    for key in keys:
        values = list(groups[key].get("features", []))[:feature_dim]
        values.extend([0.0] * (feature_dim - len(values)))
        features.append(values)

    edges = []
    for source in keys:
        for target in metadata.get("adjacency", {}).get(source, ()):
            if target in key_indices:
                edges.append([key_indices[source], key_indices[target]])

    return {
        "node_features": features,
        "edge_index": edges,
        "target_indices": sorted(
            key_indices[key] for key in neighborhood if key in key_indices
        ),
        "reward": float(reward),
    }


def buildCandidate(problem_data, incumbent, neighborhood, timeout_sec):
    candidate_data = copy.deepcopy(problem_data)
    candidate_data["max_time_seconds"] = max(0.01, float(timeout_sec))
    candidate_data["print_progress"] = False
    candidate_data["include_primary_assignments"] = True
    assignments = incumbent.get("primary_assignments", {})
    fixings = {
        key: value for key, value in assignments.items() if key not in neighborhood
    }
    return OrtoolsSolver(
        candidate_data,
        primary_fixings=fixings,
        must_change_groups=neighborhood,
        incumbent_assignments=assignments,
    )


def generateSamples(
    problem_data,
    sample_count=500,
    neighborhood_size=15,
    max_neighborhood_size=64,
    repair_time_seconds=0.5,
    seed=0,
):
    """Collect random-repair samples from one problem instance."""
    if not problem_data.get("buckets"):
        raise ValueError("The problem must contain at least one bucket")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    random_generator = random.Random(seed)
    initial_data = copy.deepcopy(problem_data)
    initial_data.pop("max_time_seconds", None)
    initial_data["stop_after_first_solution"] = True
    initial_data["print_progress"] = False
    initial_data["include_primary_assignments"] = True

    incumbent_model = OrtoolsSolver(initial_data)
    incumbent = incumbent_model.solve()
    metadata = selectorMetadata(incumbent_model, incumbent)
    selector = RandomNeighborhoodSelector(
        target_size=neighborhood_size,
        max_hops=max(1, neighborhood_size),
        seed=seed,
    )
    samples = []
    current_objective = float(incumbent.get("objective", float("inf")))

    for _ in range(int(sample_count)):
        context = {
            "model": incumbent_model,
            "metadata": metadata,
            "incumbent": incumbent,
            "objective": current_objective,
        }
        seeds = selector.selectNeighborhood(context)
        neighborhood = closeNeighborhood(
            seeds,
            metadata,
            max(1, min(int(max_neighborhood_size), int(neighborhood_size))),
            random_generator,
        )
        if not neighborhood:
            continue

        try:
            candidate_model = buildCandidate(
                problem_data, incumbent, neighborhood, repair_time_seconds
            )
            candidate = candidate_model.solve()
        except RuntimeError:
            samples.append(metadataSample(metadata, set(), 0.0))
            continue

        candidate_objective = float(candidate.get("objective", float("inf")))
        improvement = current_objective - candidate_objective
        reward = max(0.0, improvement) / max(1.0, abs(current_objective))
        if improvement > 1e-6:
            samples.append(metadataSample(metadata, neighborhood, reward))
            incumbent_model = candidate_model
            incumbent = candidate
            metadata = selectorMetadata(incumbent_model, incumbent)
            current_objective = candidate_objective
        else:
            samples.append(metadataSample(metadata, set(), 0.0))

    return samples


def loadProblem(path):
    with open(path, "r", encoding="utf-8") as handle:
        problem_data = json.load(handle)
    if not isinstance(problem_data, dict):
        raise TypeError("The problem file must contain a JSON object")
    return problem_data


def saveSamples(samples, path):
    output_directory = os.path.dirname(os.path.abspath(path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(samples, handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", help="Problem JSON accepted by ortools_lns.py")
    parser.add_argument("output", help="Output JSON sample file")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--neighborhood-size", type=int, default=15)
    parser.add_argument("--max-neighborhood-size", type=int, default=64)
    parser.add_argument("--repair-time", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = generateSamples(
        loadProblem(args.problem),
        sample_count=args.samples,
        neighborhood_size=args.neighborhood_size,
        max_neighborhood_size=args.max_neighborhood_size,
        repair_time_seconds=args.repair_time,
        seed=args.seed,
    )
    saveSamples(samples, args.output)
    print(f"[generate_gnn_samples] wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
