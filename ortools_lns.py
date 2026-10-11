"""Exact fixed-complement large neighborhood search for Tensor Graphs."""

import copy
import json
import math
import os
import sys
import time
from collections import defaultdict

from ortools.sat.python import cp_model  # noqa: F401

from lns_selectors import (
    CacheNeighborhoodSelector,
    GnnNeighborhoodSelector,
    NeighborhoodSelector,  # noqa: F401
    RandomNeighborhoodSelector,
    RandomSubgraphSelector,  # noqa: F401
    StructuralNeighborhoodSelector,  # noqa: F401
)
from ortools_cp_model import OrtoolsSolver


class CriticalPathSelector(RandomNeighborhoodSelector):
    """Compatibility selector for callers of the previous API."""

    def selectUnfrozenNodes(
        self,
        bucket,
        selection_map,
        order,
        start_times,
        end_times,
        slack,
        critical_nodes,
        classes_by_id,
        iteration,
    ):
        critical = [cid for cid in critical_nodes if cid in selection_map]
        if not critical:
            return super().selectUnfrozenNodes(
                bucket,
                selection_map,
                order,
                start_times,
                end_times,
                slack,
                critical_nodes,
                classes_by_id,
                iteration,
            )
        seed = self.random.choice(critical)
        return {seed}


class MultiChoiceSelector(RandomNeighborhoodSelector):
    """Compatibility selector targeting multi-choice eclasses."""

    def selectUnfrozenNodes(
        self,
        bucket,
        selection_map,
        order,
        start_times,
        end_times,
        slack,
        critical_nodes,
        classes_by_id,
        iteration,
    ):
        choices = [
            cid
            for cid in order
            if cid in selection_map and len(classes_by_id[cid].get("enodes", [])) > 1
        ]
        return {self.random.choice(choices)} if choices else set()


class CompositeNeighborhoodSelector(RandomNeighborhoodSelector):
    """Compatibility alias; new searches use one explicit selector."""


NeuralNeighborhoodSelector = GnnNeighborhoodSelector


class OrtoolsLnsSolver:
    """Run LNS by repeatedly repairing the complete exact CP-SAT model."""

    def __init__(self, problem_data, selector=None):
        self.problem_data = problem_data
        configured_timeout = problem_data.get("max_time_seconds")
        self.timeout_sec = (
            None if configured_timeout is None else float(configured_timeout)
        )
        if self.timeout_sec is not None and (
            not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0
        ):
            raise ValueError("max_time_seconds must be positive")
        self.print_progress = bool(problem_data.get("print_progress", True))
        configured_subproblem_timeout = problem_data.get("lns_subproblem_time_seconds")
        self.subproblem_timeout_sec = (
            None if self.timeout_sec is None else 90.0
            if configured_subproblem_timeout is None
            else float(configured_subproblem_timeout)
        )
        configured_initial_timeout = problem_data.get("lns_initial_time_seconds")
        self.initial_timeout_sec = (
            None
            if configured_initial_timeout is None and self.timeout_sec is None
            else float(
                configured_initial_timeout
                if configured_initial_timeout is not None
                else max(0.1, min(2.0, self.timeout_sec * 0.2))
            )
        )
        for name, timeout in (
            ("lns_subproblem_time_seconds", self.subproblem_timeout_sec),
            ("lns_initial_time_seconds", self.initial_timeout_sec),
        ):
            if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self.selector = selector or CacheNeighborhoodSelector(
            target_size=int(problem_data.get("lns_neighborhood_size", 15)),
            seed=problem_data.get("random_seed"),
        )

    @staticmethod
    def _selectorMetadata(model, incumbent):
        metadata = copy.deepcopy(model.getNeighborhoodMetadata())
        assignments = incumbent.get("primary_assignments", {})
        for key, group in metadata.get("groups", {}).items():
            group["incumbent_value"] = assignments.get(key)
        return metadata

    def _legacySeeds(self, selector, incumbent, metadata):
        """Adapt the old eclass selector signature during migration."""
        seeds = set()
        extractions = incumbent.get("extractions", [])
        buckets = self.problem_data.get("buckets", [])
        for position, bucket in enumerate(buckets):
            extraction = extractions[position] if position < len(extractions) else {}
            selection_map = {
                int(key): int(value)
                for key, value in extraction.get("selection_map", {}).items()
            }
            order = [int(value) for value in extraction.get("order", [])]
            classes_by_id = {int(cls["id"]): cls for cls in bucket.get("classes", [])}
            result = selector.selectUnfrozenNodes(
                bucket,
                selection_map,
                order,
                {},
                {},
                {},
                order,
                classes_by_id,
                0,
            )
            for eclass_id in result or ():
                key = OrtoolsSolver.selectionKey(bucket["bucket_idx"], eclass_id)
                if key in metadata.get("groups", {}):
                    seeds.add(key)
        return seeds

    def _selectSeeds(self, context):
        selector = self.selector
        try:
            return set(selector.selectNeighborhood(context))
        except (AttributeError, NotImplementedError):
            return self._legacySeeds(
                selector, context["incumbent"], context["metadata"]
            )

    def _buildCandidate(self, incumbent, unfrozen, timeout_sec):
        candidate_data = copy.deepcopy(self.problem_data)
        if timeout_sec is None:
            candidate_data.pop("max_time_seconds", None)
        else:
            candidate_data["max_time_seconds"] = max(0.01, timeout_sec)
        candidate_data["print_progress"] = True
        candidate_data["include_primary_assignments"] = True
        assignments = incumbent.get("primary_assignments", {})
        fixings = {
            key: value for key, value in assignments.items() if key not in unfrozen
        }
        target_getter = getattr(self.selector, "getCandidateFixings", None)
        target_fixings = target_getter() if target_getter is not None else {}
        fixings.update(target_fixings or {})
        must_change_groups = set(target_fixings or ()) or set(unfrozen)
        return OrtoolsSolver(
            candidate_data,
            primary_fixings=fixings,
            must_change_groups=must_change_groups,
            incumbent_assignments=assignments,
        )

    def solve(self):
        if not self.problem_data.get("buckets"):
            raise ValueError("OR-Tools LNS requires at least one bucket")

        started = time.perf_counter()
        initial_data = copy.deepcopy(self.problem_data)
        if self.timeout_sec is None:
            initial_data.pop("max_time_seconds", None)
            initial_data["print_progress"] = self.print_progress
        else:
            initial_data["max_time_seconds"] = max(
                0.01, min(self.timeout_sec, self.initial_timeout_sec)
            )
            initial_data["print_progress"] = False
        initial_data["include_primary_assignments"] = True
        initial_data["stop_after_first_solution"] = True
        incumbent_model = OrtoolsSolver(initial_data)
        incumbent = incumbent_model.solve()
        metadata = self._selectorMetadata(incumbent_model, incumbent)
        improvements = 0
        iterations = 0
        outcomes = defaultdict(int)

        if self.print_progress:
            print(
                f"[OrtoolsLNS] Initial objective: {incumbent['objective']:.3f}; "
                f"groups: {len(metadata.get('groups', {}))}"
            )

        while self.timeout_sec is None or time.perf_counter() - started < self.timeout_sec:
            iterations += 1
            remaining_seconds = (
                None
                if self.timeout_sec is None
                else self.timeout_sec - (time.perf_counter() - started)
            )
            context = {
                "model": incumbent_model,
                "metadata": metadata,
                "incumbent": incumbent,
                "objective": incumbent.get("objective", float("inf")),
                "iteration": iterations,
                "remaining_seconds": remaining_seconds,
            }
            seeds = self._selectSeeds(context)
            unfrozen = set(seeds)
            if not unfrozen:
                outcomes["empty"] += 1
                if self.timeout_sec is None:
                    break
                continue

            if self.timeout_sec is None:
                solve_time = self.subproblem_timeout_sec
            else:
                remaining = self.timeout_sec - (time.perf_counter() - started)
                solve_time = max(0.01, remaining)
                if self.subproblem_timeout_sec is not None:
                    solve_time = min(self.subproblem_timeout_sec, solve_time)
            candidate_model = self._buildCandidate(incumbent, unfrozen, solve_time)
            try:
                candidate = candidate_model.solve()
            except RuntimeError:
                outcomes["infeasible"] += 1
                continue

            candidate_objective = float(candidate.get("objective", float("inf")))
            incumbent_objective = float(incumbent.get("objective", float("inf")))
            if candidate_objective + 1e-6 < incumbent_objective:
                improvement = incumbent_objective - candidate_objective
                incumbent = candidate
                incumbent_model = candidate_model
                metadata = self._selectorMetadata(incumbent_model, incumbent)
                improvements += 1
                outcomes["improved"] += 1
                if self.print_progress:
                    print(
                        f"[OrtoolsLNS] Iteration {iterations}: objective "
                        f"{candidate_objective:.3f} (-{improvement:.3f})"
                    )
            else:
                outcomes["no_improvement"] += 1

        incumbent["lns"] = {
            "iterations": iterations,
            "improvements": improvements,
            "outcomes": dict(outcomes),
        }
        return incumbent

# TODO: ortools_lns.py shouldn't call OrtoolsSolver. only OrtoolsLnsSolver. make ortools.py that routes to OrtoolsSolver/OrtoolsLnsSolver based on args. remove TENSOR_GRAPHS_USE_LNS from codebase.
def solveOrtools(problem_data):
    """Unified OR-Tools entry point."""
    use_lns = (
        os.environ.get("TENSOR_GRAPHS_USE_LNS") == "1"
        or problem_data.get("use_ortools_lns", False)
    )
    if use_lns or not problem_data.get("use_ortools_full", False):
        return OrtoolsLnsSolver(problem_data, selector=CacheNeighborhoodSelector()).solve()
    return OrtoolsSolver(problem_data).solve()


def main():
    if len(sys.argv) < 3:
        print("Usage: python ortools_lns.py <problem.json> <solution.json>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        problem_data = json.load(handle)
    solution = solveOrtools(problem_data)
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2)


if __name__ == "__main__":
    main()
