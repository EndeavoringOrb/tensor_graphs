"""Pure Scheduling CP-SAT Solver for Tensor Graphs.

Solves joint cache, eclass, enode, and dispatch scheduling across buckets
without bufferization, in-place aliasing, or physical memory allocation.
Finds the unconstrained lower bound on makespan.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import copy
import json
import math
import os
import sys
import time
from typing import Any

from ortools.sat.python import cp_model

try:
    from ortools_hints import compiledGraphsToHints
except ImportError:
    compiledGraphsToHints = None


def memSpaceKey(mem_space: dict[str, Any]) -> str:
    return f"{mem_space['type']}:{mem_space['idx']}"


class PureScheduleSolver:
    """Exact CP-SAT solver for kernel selection and dispatch scheduling.

    Omits 2D memory packing, in-place aliasing, and bufferization to isolate
    the minimum achievable compute makespan across hardware engines.
    """

    time_scale = 1000  # Milliseconds to microseconds
    integer_limit = 2**60

    def __init__(self, problem_data: dict[str, Any]):
        self.problem_data = problem_data
        self.model = cp_model.CpModel()
        self.buckets = problem_data.get("buckets", [])
        self.caching_enabled = not problem_data.get("disable_caching", False)
        self.candidates = (
            problem_data.get("candidates", []) if self.caching_enabled else []
        )
        self.candidates_by_base_id = {
            int(c["base_eclass_id"]): c for c in self.candidates
        }
        self.full_bucket_idx = problem_data.get("full_bucket_idx", 0)

        self.cache_choices: dict[int, cp_model.IntVar] = {}
        self.bucket_models: list[dict[str, Any]] = []
        self.objective_terms: list[cp_model.LinearExpr] = []
        self.external_hints: dict[str, Any] = {"cached": {}, "buckets": {}}

        if problem_data.get("cpu_hints") and compiledGraphsToHints is not None:
            try:
                self.external_hints = compiledGraphsToHints(
                    problem_data, problem_data["cpu_hints"]
                )
            except Exception as e:
                print(f"[Warning] Could not parse CPU hints: {e}", file=sys.stderr)

        self.model_built = False

    def addHint(self, var: Any, val: int | float) -> None:
        if var is not None and hasattr(var, "Index"):
            self.model.AddHint(var, int(val))

    def duration(self, enode: dict[str, Any]) -> int | None:
        try:
            cost = float(enode.get("cost", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(cost) or cost < 0:
            return None
        if any(enode.get(flag) for flag in ("is_input", "is_cache", "is_view")):
            return 0
        scaled_cost = cost * self.time_scale
        if not math.isfinite(scaled_cost) or scaled_cost >= self.integer_limit:
            raise ValueError("Kernel cost exceeds integer limit")
        return max(1, math.ceil(scaled_cost))

    def engineKeys(self, enode: dict[str, Any], mem_space: dict[str, Any]) -> set[str]:
        engines = enode.get("engines") or [
            {"type": 2, "idx": mem_space["idx"]}
            if mem_space.get("type") == 3
            else {"type": 0, "idx": 0}
        ]
        return {memSpaceKey(engine) for engine in engines}

    def createCacheVariables(self) -> None:
        if not self.caching_enabled:
            return

        for candidate in self.candidates:
            base_eclass_id = int(candidate["base_eclass_id"])
            if base_eclass_id in self.cache_choices:
                continue
            cached_var = self.model.NewBoolVar(f"cached_base_{base_eclass_id}")
            self.cache_choices[base_eclass_id] = cached_var

            hinted = self.external_hints.get("cached", {}).get(base_eclass_id)
            if hinted is not None:
                self.addHint(cached_var, int(hinted))

    def getReachableClasses(self, bucket: dict[str, Any]) -> set[int]:
        classes = {int(cls["id"]): cls for cls in bucket.get("classes", [])}
        root_id = int(bucket["root_eclass_id"])

        grounded: set[int] = set()
        parent_enodes: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
        remaining: dict[tuple[int, int], int] = {}

        for cid, cls in classes.items():
            for enode in cls.get("enodes", []):
                key = (cid, int(enode["enode_idx"]))
                children = [
                    int(c) for c in enode.get("children", []) if int(c) in classes
                ]
                u_children = set(children)
                remaining[key] = len(u_children)
                if len(u_children) == 0:
                    grounded.add(cid)
                for child in u_children:
                    parent_enodes[child].append((cid, key))

        queue = list(grounded)
        while queue:
            curr = queue.pop()
            for p_cid, p_key in parent_enodes[curr]:
                if remaining[p_key] > 0:
                    remaining[p_key] -= 1
                    if remaining[p_key] == 0:
                        if p_cid not in grounded:
                            grounded.add(p_cid)
                            queue.append(p_cid)

        reachable: set[int] = set()
        roots = [root_id]
        if self.caching_enabled:
            for cand in self.candidates:
                base_id = int(cand["base_eclass_id"])
                for cid, cls in classes.items():
                    if (
                        int(cls.get("base_eclass_id", -1)) == base_id
                        and cid in grounded
                    ):
                        roots.append(cid)

        q = [r for r in roots if r in grounded]
        while q:
            curr = q.pop()
            if curr in reachable:
                continue
            reachable.add(curr)
            cls = classes[curr]
            for enode in cls.get("enodes", []):
                children = [
                    int(c)
                    for c in enode.get("children", [])
                    if int(c) in classes and int(c) in grounded
                ]
                if len(set(children)) == len(set(enode.get("children", []))):
                    for child in children:
                        if child not in reachable:
                            q.append(child)

        return reachable

    def createBucket(self, bucket: dict[str, Any]) -> None:
        model = self.model
        b = int(bucket["bucket_idx"])

        reachable = self.getReachableClasses(bucket)
        classes = {
            int(cls["id"]): cls
            for cls in bucket.get("classes", [])
            if int(cls["id"]) in reachable
        }
        root_id = int(bucket["root_eclass_id"])
        if not classes or root_id not in classes:
            raise ValueError(f"No reachable classes or invalid root in bucket {b}")

        durations = {
            (cid, int(enode["enode_idx"])): self.duration(enode)
            for cid, cls in classes.items()
            for enode in cls.get("enodes", [])
        }

        # Calculate scheduling time horizon
        max_durations = [
            max(
                (
                    durations.get((cid, int(enode["enode_idx"]))) or 0
                    for enode in cls.get("enodes", [])
                ),
                default=0,
            )
            for cid, cls in classes.items()
        ]
        horizon = 1 + len(classes) + sum(max_durations)

        plan_hint = self.external_hints.get("buckets", {}).get(b)
        if plan_hint is not None:
            horizon = max(horizon, int(plan_hint.get("makespan", 0)) + 1)

        clean = set(map(int, bucket.get("clean_eclasses", [])))
        engines: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)
        engine_work: dict[str, list[cp_model.LinearExpr]] = defaultdict(list)
        consumers: dict[int, list[cp_model.IntVar]] = defaultdict(list)
        cache_nodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        nodes: dict[int, dict[str, Any]] = {}

        for cid, cls in classes.items():
            name = f"b{b}_c{cid}"
            node: dict[str, Any] = {
                "cls": cls,
                "active": model.NewBoolVar(f"{name}_active"),
                "start": model.NewIntVar(0, horizon, f"{name}_start"),
                "end": model.NewIntVar(0, horizon, f"{name}_end"),
                "selections": {},
                "cached": None,
            }
            nodes[cid] = node

            # Inactive nodes stay at zero
            model.Add(node["start"] == 0).OnlyEnforceIf(node["active"].Not())
            model.Add(node["end"] == 0).OnlyEnforceIf(node["active"].Not())
            model.Add(node["end"] >= node["start"]).OnlyEnforceIf(node["active"])

        # Link cache variables and enforce cache activation requirements
        for cid, node in nodes.items():
            cls = node["cls"]
            base_eclass_id = int(cls.get("base_eclass_id", cid))
            if base_eclass_id in self.cache_choices:
                cached_var = self.cache_choices[base_eclass_id]
                node["cached"] = cached_var
                cache_nodes[base_eclass_id].append(node)
                # If cached, the full bucket MUST compute this eclass
                if b == self.full_bucket_idx:
                    model.AddImplication(cached_var, node["active"])

        # Enode options and precedence
        for cid, node in nodes.items():
            cls = node["cls"]
            base_eclass_id = int(cls.get("base_eclass_id", cid))
            cached_var = node["cached"]

            for enode in cls.get("enodes", []):
                e_idx = int(enode["enode_idx"])
                selected = model.NewBoolVar(f"b{b}_c{cid}_e{e_idx}_selected")
                node["selections"][e_idx] = (selected, enode)

                duration = durations.get((cid, e_idx))
                children = [
                    int(c)
                    for c in enode.get("children", [])
                    if int(c) in nodes
                ]

                # If missing children or duration is invalid, cannot select
                if (
                    duration is None
                    or len(children) != len(enode.get("children", []))
                ):
                    model.Add(selected == 0)
                    continue

                # Kernel duration
                model.Add(node["end"] == node["start"] + duration).OnlyEnforceIf(
                    selected
                )

                # Cache read restrictions
                if enode.get("is_cache"):
                    candidate = self.candidates_by_base_id.get(base_eclass_id, {})
                    clean_buckets = candidate.get("clean_buckets")
                    can_read = (
                        cid in clean
                        and b != self.full_bucket_idx
                        and (clean_buckets is None or b in clean_buckets)
                        and int(enode.get("base_eclass_id", base_eclass_id))
                        == base_eclass_id
                    )
                    if can_read and cached_var is not None:
                        model.AddImplication(selected, cached_var)
                    else:
                        model.Add(selected == 0)
                    if children:
                        model.Add(selected == 0)

                # Cache scatter restrictions
                if enode.get("is_scatter"):
                    if b != self.full_bucket_idx and cached_var is not None:
                        model.AddImplication(selected, cached_var)
                    else:
                        model.Add(selected == 0)
                    if not children:
                        model.Add(selected == 0)

                # Children precedence: all inputs must complete before this enode starts
                for child_id in set(children):
                    child_node = nodes[child_id]
                    consumers[child_id].append(selected)
                    model.AddImplication(selected, child_node["active"])
                    model.Add(child_node["end"] <= node["start"]).OnlyEnforceIf(
                        selected
                    )

                # Input nodes start and end at 0
                if enode.get("is_input"):
                    model.Add(node["start"] == 0).OnlyEnforceIf(selected)
                    model.Add(node["end"] == 0).OnlyEnforceIf(selected)

                # Engine interval for concurrency & makespan reasoning
                if duration > 0:
                    interval = model.NewOptionalIntervalVar(
                        node["start"],
                        duration,
                        node["end"],
                        selected,
                        f"b{b}_c{cid}_e{e_idx}_run",
                    )
                    for engine_key in self.engineKeys(enode, cls["mem_space"]):
                        engines[engine_key].append(interval)
                        engine_work[engine_key].append(duration * selected)

            # Exactly one enode is selected if active, none if inactive
            selection_vars = [item[0] for item in node["selections"].values()]
            model.AddExactlyOne(selection_vars + [node["active"].Not()])

            # Root eclass must always be active
            if cid == root_id:
                model.Add(node["active"] == 1)

        # Consumer requirement: active non-root nodes must be consumed or cached
        for cid, node in nodes.items():
            if cid != root_id:
                or_terms = consumers[cid][:]
                if node["cached"] is not None:
                    or_terms.append(node["cached"])
                if not or_terms:
                    model.Add(node["active"] == 0)
                else:
                    model.AddBoolOr(or_terms).OnlyEnforceIf(node["active"])

        # Cache candidate requires at least one active matching producer
        for base_eclass_id, cached_var in self.cache_choices.items():
            matches = cache_nodes.get(base_eclass_id, [])
            if not matches:
                model.Add(cached_var == 0)
            else:
                model.AddBoolOr([n["active"] for n in matches]).OnlyEnforceIf(
                    cached_var
                )

        # Engine non-overlap constraints (multi-engine concurrency)
        makespan = model.NewIntVar(0, horizon, f"b{b}_makespan")
        for node in nodes.values():
            model.Add(makespan >= node["end"])

        for engine_key, intervals in engines.items():
            model.AddNoOverlap(intervals)
            model.Add(makespan >= sum(engine_work[engine_key]))

        # Objective weighting
        include_in_objective = b != self.full_bucket_idx or (
            len(self.buckets) == 1 and "weight" not in bucket
        )
        if include_in_objective:
            weight = float(bucket.get("weight", 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid bucket weight {weight}")
            self.objective_terms.append(weight * makespan)

        # Apply external hints if available
        if plan_hint is not None:
            self.applyHints(plan_hint, nodes, makespan)

        self.bucket_models.append(
            {
                "bucket": bucket,
                "nodes": nodes,
                "makespan": makespan,
                "horizon": horizon,
            }
        )

    def applyHints(
        self,
        plan_hint: dict[str, Any],
        nodes: dict[int, dict[str, Any]],
        makespan: cp_model.IntVar,
    ) -> None:
        hinted_nodes = plan_hint.get("nodes", {})
        self.addHint(makespan, int(plan_hint.get("makespan", 0)))

        for cid, node in nodes.items():
            hint = hinted_nodes.get(cid)
            active = hint is not None and int(hint.get("active", 0)) != 0

            self.addHint(node["active"], int(active))
            self.addHint(node["start"], int(hint.get("start", 0)) if active else 0)
            self.addHint(node["end"], int(hint.get("end", 0)) if active else 0)

            selection = hint.get("selection") if active else None
            for e_idx, (selection_var, _) in node["selections"].items():
                self.addHint(selection_var, int(selection == e_idx))

    def buildModel(self) -> PureScheduleSolver:
        if not self.buckets:
            raise ValueError("Pure schedule solver requires at least one bucket")

        if not self.model_built:
            self.createCacheVariables()
            for bucket in self.buckets:
                self.createBucket(bucket)
            if self.objective_terms:
                self.model.Minimize(sum(self.objective_terms))
            self.model_built = True

        return self

    def extractTopologicalOrder(
        self,
        active_cids: set[int],
        selections: dict[int, int],
        classes: dict[int, dict[str, Any]],
        start_times: dict[int, float],
    ) -> list[int]:
        """Compute a valid topological order prioritizing scheduled start times."""
        in_degree: dict[int, int] = {cid: 0 for cid in active_cids}
        dependents: dict[int, list[int]] = defaultdict(list)

        for cid in active_cids:
            e_idx = selections[cid]
            enode = classes[cid]["enodes"][e_idx]
            for child in set(enode.get("children", [])):
                child_id = int(child)
                if child_id in active_cids:
                    in_degree[cid] += 1
                    dependents[child_id].append(cid)

        # Ready queue sorted by (start_time, cid)
        ready = [cid for cid in active_cids if in_degree[cid] == 0]
        ready.sort(key=lambda x: (start_times.get(x, 0.0), x))

        order: list[int] = []
        while ready:
            curr = ready.pop(0)
            order.append(curr)
            for dep in dependents[curr]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    ready.append(dep)
            ready.sort(key=lambda x: (start_times.get(x, 0.0), x))

        if len(order) != len(active_cids):
            # Fallback if any unexpected cycle
            remaining = [cid for cid in active_cids if cid not in order]
            remaining.sort(key=lambda x: (start_times.get(x, 0.0), x))
            order.extend(remaining)

        return order

    def decodeSolution(
        self, solver: cp_model.CpSolver, status: int
    ) -> dict[str, Any]:
        cached_nodes = [
            base_eclass_id
            for base_eclass_id, var in self.cache_choices.items()
            if solver.Value(var)
        ]

        extractions = []
        for bucket_model in self.bucket_models:
            bucket = bucket_model["bucket"]
            b = int(bucket["bucket_idx"])
            nodes = bucket_model["nodes"]
            classes = {int(c["id"]): c for c in bucket.get("classes", [])}

            active_cids = {
                cid for cid, node in nodes.items() if solver.Value(node["active"])
            }

            selections: dict[int, int] = {}
            start_times: dict[int, float] = {}
            end_times: dict[int, float] = {}
            schedule: dict[str, Any] = {}

            for cid in active_cids:
                node = nodes[cid]
                s_ms = solver.Value(node["start"]) / self.time_scale
                e_ms = solver.Value(node["end"]) / self.time_scale
                start_times[cid] = s_ms
                end_times[cid] = e_ms

                for e_idx, (sel_var, enode) in node["selections"].items():
                    if solver.Value(sel_var):
                        selections[cid] = e_idx
                        schedule[str(cid)] = {
                            "enode_idx": e_idx,
                            "op_type": enode.get("op_type"),
                            "kernel_id": enode.get("kernel_id"),
                            "cost_ms": float(enode.get("cost", 0.0) or 0.0),
                            "start_ms": s_ms,
                            "end_ms": e_ms,
                            "duration_ms": max(0.0, e_ms - s_ms),
                            "is_view": enode.get("is_view", False),
                            "is_cache": enode.get("is_cache", False),
                            "is_input": enode.get("is_input", False),
                            "children": enode.get("children", []),
                        }
                        break

            order = self.extractTopologicalOrder(
                active_cids, selections, classes, start_times
            )
            makespan_ms = solver.Value(bucket_model["makespan"]) / self.time_scale

            extractions.append(
                {
                    "bucket_idx": b,
                    "makespan_ms": makespan_ms,
                    "active_classes_count": len(active_cids),
                    "selection_map": {str(k): v for k, v in selections.items()},
                    "order": order,
                    "schedule": schedule,
                }
            )

        obj_val = solver.ObjectiveValue() / self.time_scale
        best_bound = solver.BestObjectiveBound() / self.time_scale

        return {
            "solver": "ortools_pure_schedule",
            "status": solver.StatusName(status),
            "objective_ms": obj_val,
            "best_bound_ms": best_bound,
            "cached_base_eclass_ids": cached_nodes,
            "extractions": extractions,
            "wall_time_sec": solver.WallTime(),
        }

    def solve(self) -> dict[str, Any]:
        self.buildModel()

        error = self.model.Validate()
        if error:
            raise ValueError(f"Invalid pure schedule CP model: {error}")

        solver = cp_model.CpSolver()

        max_time = self.problem_data.get("max_time_seconds")
        if max_time is not None:
            solver.parameters.max_time_in_seconds = float(max_time)

        num_workers = int(
            self.problem_data.get("num_workers", min(16, os.cpu_count() or 4))
        )
        solver.parameters.num_workers = max(1, num_workers)

        print_progress = bool(
            self.problem_data.get(
                "print_progress", self.problem_data.get("log_search_progress", True)
            )
        )
        solver.parameters.log_search_progress = print_progress

        # Enable complete search engines (LP, SAT/CDCL, Pseudo-cost, LNS)
        status = solver.Solve(self.model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"No feasible schedule found: {solver.StatusName(status)}"
            )

        return self.decodeSolution(solver, status)


def solveSchedule(problem_data: dict[str, Any]) -> dict[str, Any]:
    """Entry point for programmatic invocation."""
    return PureScheduleSolver(problem_data).solve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solve unconstrained kernel schedule & dispatch lower-bound."
    )
    parser.add_argument("problem_json", help="Path to exported problem.json")
    parser.add_argument(
        "solution_json",
        nargs="?",
        default=None,
        help="Optional path to save solution.json",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Max solver time in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of CP-SAT workers",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress CP-SAT search progress log",
    )
    args = parser.parse_args()

    with open(args.problem_json, "r", encoding="utf-8") as f:
        problem_data = json.load(f)

    if args.time is not None:
        problem_data["max_time_seconds"] = args.time
    if args.workers is not None:
        problem_data["num_workers"] = args.workers
    if args.quiet:
        problem_data["print_progress"] = False

    t0 = time.perf_counter()
    result = solveSchedule(problem_data)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"PURE SCHEDULE RESULT ({result['status']})")
    print(f"Total Objective:  {result['objective_ms']:.3f} ms")
    print(f"Best Lower Bound: {result['best_bound_ms']:.3f} ms")
    print(f"Wall Time:        {elapsed:.2f} s")
    print(f"Cached Candidates: {len(result['cached_base_eclass_ids'])}")
    for ext in result["extractions"]:
        print(
            f"  - Bucket {ext['bucket_idx']}: makespan = {ext['makespan_ms']:.3f} ms, "
            f"{ext['active_classes_count']} active nodes"
        )
    print("=" * 60 + "\n")

    if args.solution_json:
        with open(args.solution_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved solution to {args.solution_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
