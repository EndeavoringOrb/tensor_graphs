"""Generate supervised samples for the LNS neighborhood-selection GNN.

The input is the same problem JSON accepted by ``ortools_lns.py``.  The run
stores the complete problem once and appends compact samples containing the
tested neighborhood and the repaired runtime relative to the incumbent.  Run
this script before ``train.py`` when a selector checkpoint is not available.

Example::

    .venvx64\\Scripts\\python.exe -m lns_selectors.gnn.generate_samples \
        problem.json --samples 500
"""

import argparse
import copy
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Permit ``python lns_selectors/gnn/generate_samples.py ...`` as well as ``-m``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lns_selectors.gnn.common import (
        CONFIG_FILE,
        PROBLEM_FILE,
        SAMPLES_FILE,
        SAMPLES_ROOT,
        SampleGenerationConfig,
        SampleWriter,
        initializeRun,
        loadJsonl,
        loadProblem,
        problemHash,
        readJson,
        resolveRunDirectory,
        writeJson,
    )
    from lns_selectors.random import RandomNeighborhoodSelector
else:
    from ..random import RandomNeighborhoodSelector
    from .common import (
        CONFIG_FILE,
        PROBLEM_FILE,
        SAMPLES_FILE,
        SAMPLES_ROOT,
        SampleGenerationConfig,
        SampleWriter,
        initializeRun,
        loadJsonl,
        loadProblem,
        problemHash,
        readJson,
        resolveRunDirectory,
        writeJson,
    )
from ortools_cp_model import OrtoolsSolver


def selectorMetadata(model, incumbent):
    metadata = copy.deepcopy(model.getNeighborhoodMetadata())
    assignments = incumbent.get("primary_assignments", {})
    for key, group in metadata.get("groups", {}).items():
        group["incumbent_value"] = assignments.get(key)
    return metadata


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
        incumbent_assignments=assignments,
    )


def generateSamples(
    problem_data,
    sample_count=500,
    neighborhood_size=15,
    max_neighborhood_size=64,
    repair_time_seconds=90.0,
    seed=0,
    on_sample=None,
):
    """Collect random-repair samples from one problem instance."""
    if not problem_data.get("buckets"):
        raise ValueError("The problem must contain at least one bucket")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

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
        # The selector owns the neighborhood.  Keep max_neighborhood_size in
        # the API for compatibility with persisted sample-run configurations.
        neighborhood = set(seeds)
        if not neighborhood:
            continue

        candidate_model = buildCandidate(
            problem_data, incumbent, neighborhood, repair_time_seconds
        )
        try:
            candidate = candidate_model.solve()
        except RuntimeError as error:
            error_text = str(error)
            if error_text == "OR-Tools full found no feasible joint plan: INFEASIBLE":
                # Infeasibility is a definitive negative label for this
                # neighborhood.  Treat it like a repair that found no
                # improvement, while UNKNOWN remains an unresolved timeout.
                sample = metadataSample(
                    metadata,
                    neighborhood,
                    current_objective,
                    current_objective,
                )
                samples.append(sample)
                if on_sample is not None:
                    on_sample(sample)
                print(f"Recording infeasible candidate repair: {error}")
                continue
            if error_text == "OR-Tools full found no feasible joint plan: UNKNOWN":
                print(f"Skipping candidate repair: {error}")
                continue
            else:
                raise

        if candidate.get("status") != "OPTIMAL":
            print(
                "Skipping non-OPTIMAL candidate. consider using simpler problem or raising timeout"
            )
            # A FEASIBLE result at the time limit is deliberately excluded.
            continue

        candidate_objective = float(candidate.get("objective", float("inf")))
        if not math.isfinite(candidate_objective) or not math.isfinite(
            current_objective
        ):
            continue
        improvement = current_objective - candidate_objective
        sample = metadataSample(
            metadata,
            neighborhood,
            current_objective,
            candidate_objective,
        )
        samples.append(sample)
        if on_sample is not None:
            on_sample(sample)
        if improvement > 1e-6:
            incumbent_model = candidate_model
            incumbent = candidate
            metadata = selectorMetadata(incumbent_model, incumbent)
            current_objective = candidate_objective

    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "problem",
        nargs="?",
        help="Problem JSON accepted by ortools_lns.py (required for a new run)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="INDEX",
        help="Resume the latest run, or the numbered run INDEX",
    )
    parser.add_argument("--samples-root", default=str(SAMPLES_ROOT))
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--neighborhood-size", type=int, default=None)
    parser.add_argument("--max-neighborhood-size", type=int, default=None)
    parser.add_argument(
        "--repair-time",
        type=float,
        default=None,
        help="Optimal-solve timeout for each repair neighborhood (seconds)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.resume is None and not args.problem:
        parser.error("problem is required when starting a new run")
    run_directory = resolveRunDirectory(args.samples_root, args.resume)
    if args.resume is None:
        config = SampleGenerationConfig(
            sample_count=args.samples if args.samples is not None else 500,
            neighborhood_size=args.neighborhood_size
            if args.neighborhood_size is not None
            else 15,
            max_neighborhood_size=(
                args.max_neighborhood_size
                if args.max_neighborhood_size is not None
                else 64
            ),
            repair_time=args.repair_time if args.repair_time is not None else 90.0,
            seed=args.seed if args.seed is not None else 0,
        )
        problem_data = initializeRun(args.problem, run_directory, config)
    else:
        problem_data = loadProblem(run_directory / PROBLEM_FILE)
        saved_config = readJson(run_directory / CONFIG_FILE)
        config = SampleGenerationConfig(
            sample_count=args.samples
            if args.samples is not None
            else saved_config["sample_count"],
            neighborhood_size=(
                args.neighborhood_size
                if args.neighborhood_size is not None
                else saved_config["neighborhood_size"]
            ),
            max_neighborhood_size=(
                args.max_neighborhood_size
                if args.max_neighborhood_size is not None
                else saved_config["max_neighborhood_size"]
            ),
            repair_time=(
                args.repair_time
                if args.repair_time is not None
                else saved_config.get(
                    "repair_time", saved_config.get("repair_time_seconds", 90.0)
                )
            ),
            seed=args.seed if args.seed is not None else saved_config["seed"],
        )
        writeJson(run_directory / CONFIG_FILE, config.toDict())

    samples_path = run_directory / SAMPLES_FILE
    existing_count = len(loadJsonl(samples_path))
    if existing_count >= config.sample_count:
        print(
            f"[generate_samples] run already contains {existing_count} samples: {run_directory}"
        )
        return

    emitted_count = 0

    def appendNewSample(sample):
        nonlocal emitted_count
        if emitted_count >= existing_count:
            writer.append(sample)
        emitted_count += 1

    with SampleWriter(samples_path) as writer:
        generateSamples(
            problem_data,
            sample_count=config.sample_count,
            neighborhood_size=config.neighborhood_size,
            max_neighborhood_size=config.max_neighborhood_size,
            repair_time_seconds=config.repair_time,
            seed=config.seed,
            on_sample=appendNewSample,
        )
    print(
        f"[generate_samples] run={run_directory.name} "
        f"samples={max(existing_count, emitted_count)} "
        f"problem_hash={problemHash(problem_data)}"
    )


if __name__ == "__main__":
    main()
