"""Generate supervised samples for the LNS neighborhood-selection GNN.

The input is the same problem JSON accepted by ``ortools_lns.py``.  Each
sample contains the complete CP model's primary-decision graph, the current
incumbent runtime, the tested neighborhood, and the optimal repaired runtime
relative to the incumbent.  Run this script before ``train_gnn.py`` when a
selector checkpoint is not available yet.

Example::

    .venvx64\\Scripts\\python.exe generate_gnn_samples.py \
        problem.json runs/lns_samples.json --samples 500
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile

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


def metadataSample(
    metadata, neighborhood, current_runtime, solution_runtime, problem_id=None
):
    """Convert selector metadata into a runtime-regression sample.

    The target is the candidate plan runtime relative to the incumbent.  It is
    clipped because an optimal repair is allowed to be no better than the
    incumbent, and the training target must remain in ``[0, 1]``.
    """
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

    current_runtime = float(current_runtime)
    solution_runtime = float(solution_runtime)
    denominator = max(1.0, abs(current_runtime))
    relative_runtime = max(0.0, min(1.0, solution_runtime / denominator))
    sample = {
        "node_features": features,
        "edge_index": edges,
        "neighborhood_indices": sorted(
            key_indices[key] for key in neighborhood if key in key_indices
        ),
        "current_runtime": current_runtime,
        "solution_runtime": solution_runtime,
        "relative_runtime": relative_runtime,
        "runtime_ratio": relative_runtime,
        "target": relative_runtime,
    }
    # Keep the old name as a harmless compatibility alias for consumers that
    # use the neighborhood as a node mask.
    sample["target_indices"] = list(sample["neighborhood_indices"])
    if problem_id is not None:
        sample["problem_id"] = problem_id
    return sample


def buildCandidate(problem_data, incumbent, neighborhood, timeout_sec):
    candidate_data = copy.deepcopy(problem_data)
    candidate_data["max_time_seconds"] = max(0.01, float(timeout_sec))
    candidate_data["print_progress"] = False
    candidate_data["include_primary_assignments"] = True
    candidate_data["stop_after_first_solution"] = False
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
    repair_time_seconds=90.0,
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
    initial_data["print_progress"] = True
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

        candidate_model = buildCandidate(
            problem_data, incumbent, neighborhood, repair_time_seconds
        )
        candidate = candidate_model.solve()

        if candidate.get("status") != "OPTIMAL":
            print(f"Skipping non-OPTIMAL candidate. consider using simpler problem or raising timeout")
            # A FEASIBLE result at the time limit is deliberately excluded.
            continue

        candidate_objective = float(candidate.get("objective", float("inf")))
        if not math.isfinite(candidate_objective) or not math.isfinite(
            current_objective
        ):
            continue
        improvement = current_objective - candidate_objective
        samples.append(
            metadataSample(
                metadata,
                neighborhood,
                current_objective,
                candidate_objective,
            )
        )
        if improvement > 1e-6:
            incumbent_model = candidate_model
            incumbent = candidate
            metadata = selectorMetadata(incumbent_model, incumbent)
            current_objective = candidate_objective

    return samples


def loadProblem(path):
    with open(path, "r", encoding="utf-8") as handle:
        problem_data = json.load(handle)
    if not isinstance(problem_data, dict):
        raise TypeError("The problem file must contain a JSON object")
    print(f"Loaded problem")
    return problem_data


def problemId(problem_data):
    """Return a stable identity for one complete problem description."""
    encoded = json.dumps(
        problem_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sampleStore(existing):
    """Normalize current and legacy files to the appendable store schema."""
    if isinstance(existing, dict) and isinstance(existing.get("problems"), dict):
        return existing
    if isinstance(existing, list):
        # Do not throw away files created by the previous generator.  They do
        # not have a problem identity, so retain them in a legacy bucket.
        return {
            "format": "tensor_graphs_gnn_samples",
            "version": 2,
            "problems": {"legacy": {"samples": existing}},
        }
    if existing is None:
        return {
            "format": "tensor_graphs_gnn_samples",
            "version": 2,
            "problems": {},
        }
    raise ValueError("The existing training sample file must contain a list or store")


def saveSamples(samples, path, problem_id="default"):
    """Append samples under one problem key without discarding old samples."""
    output_directory = os.path.dirname(os.path.abspath(path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    existing = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    store = _sampleStore(existing)
    problems = store.setdefault("problems", {})
    problem_entry = problems.setdefault(problem_id, {"samples": []})
    if isinstance(problem_entry, list):
        problem_entry = {"samples": problem_entry}
        problems[problem_id] = problem_entry
    problem_entry.setdefault("samples", []).extend(samples)
    temporary_path = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=output_directory, prefix=".gnn_samples_", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(store, handle)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", help="Problem JSON accepted by ortools_lns.py")
    parser.add_argument("output", help="Output JSON sample file")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--neighborhood-size", type=int, default=15)
    parser.add_argument("--max-neighborhood-size", type=int, default=64)
    parser.add_argument(
        "--repair-time",
        type=float,
        default=90.0,
        help="Optimal-solve timeout for each repair neighborhood (seconds)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem_data = loadProblem(args.problem)
    samples = generateSamples(
        problem_data,
        sample_count=args.samples,
        neighborhood_size=args.neighborhood_size,
        max_neighborhood_size=args.max_neighborhood_size,
        repair_time_seconds=args.repair_time,
        seed=args.seed,
    )
    saveSamples(samples, args.output, problemId(problem_data))
    print(f"[generate_gnn_samples] wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
