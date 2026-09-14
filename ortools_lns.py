"""Iterative Subgraph Large Neighborhood Search (LNS) Solver for Tensor Graphs.

Provides a modular LNS framework where the strategy for selecting which
eclasses to unfreeze is cleanly separated into pluggable NeighborhoodSelectors.
This allows simple heuristics today (Random, Critical Path, Multi-Choice) and
neural / learned models in the future.
"""

import abc
import copy
import heapq
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None


# =============================================================================
# Modular Neighborhood Selectors
# =============================================================================

class NeighborhoodSelector(abc.ABC):
    """Abstract base class for LNS neighborhood selection.
    
    Subclasses decide which eclass IDs should be unfrozen in each iteration.
    """

    @abc.abstractmethod
    def selectUnfrozenNodes(
        self,
        bucket: Dict[str, Any],
        selection_map: Dict[int, int],
        order: List[int],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        slack: Dict[int, int],
        critical_nodes: List[int],
        classes_by_id: Dict[int, Dict[str, Any]],
        iteration: int,
    ) -> Set[int]:
        """Select a set of active eclass IDs to unfreeze for local optimization."""
        raise NotImplementedError


class RandomSubgraphSelector(NeighborhoodSelector):
    """Dead-simple baseline heuristic: picks a random seed node and expands a k-hop neighborhood."""

    def __init__(self, target_size: int = 15, max_hops: int = 2):
        self.target_size = target_size
        self.max_hops = max_hops

    def selectUnfrozenNodes(
        self,
        bucket: Dict[str, Any],
        selection_map: Dict[int, int],
        order: List[int],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        slack: Dict[int, int],
        critical_nodes: List[int],
        classes_by_id: Dict[int, Dict[str, Any]],
        iteration: int,
    ) -> Set[int]:
        active_set = set(selection_map.keys())
        # Prefer candidates that have more than 1 alternative enode
        multi_choice = [
            cid for cid in order
            if cid in active_set and len(classes_by_id[cid].get("enodes", [])) > 1
        ]
        pool = multi_choice if multi_choice else list(active_set)
        if not pool:
            return set()

        seed = random.choice(pool)
        unfrozen: Set[int] = {seed}
        frontier: List[int] = [seed]

        # Build adjacency over active nodes
        parents: Dict[int, List[int]] = defaultdict(list)
        children: Dict[int, List[int]] = defaultdict(list)
        for cid in active_set:
            e_idx = selection_map[cid]
            enode = classes_by_id[cid]["enodes"][e_idx]
            for ch in enode.get("children", []):
                if ch in active_set:
                    children[cid].append(ch)
                    parents[ch].append(cid)

        for _ in range(self.max_hops):
            next_frontier = []
            for curr in frontier:
                neighbors = parents.get(curr, []) + children.get(curr, [])
                for nxt in neighbors:
                    if nxt not in unfrozen:
                        unfrozen.add(nxt)
                        next_frontier.append(nxt)
                        if len(unfrozen) >= self.target_size:
                            return unfrozen
            frontier = next_frontier
            if not frontier:
                break

        return unfrozen


class CriticalPathSelector(NeighborhoodSelector):
    """Domain-aware heuristic: targets operations on the critical path (slack == 0).
    
    Shortening operations on the critical path is mathematically required to
    reduce overall makespan.
    """

    def __init__(self, target_size: int = 20, max_hops: int = 2):
        self.target_size = target_size
        self.max_hops = max_hops

    def selectUnfrozenNodes(
        self,
        bucket: Dict[str, Any],
        selection_map: Dict[int, int],
        order: List[int],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        slack: Dict[int, int],
        critical_nodes: List[int],
        classes_by_id: Dict[int, Dict[str, Any]],
        iteration: int,
    ) -> Set[int]:
        active_set = set(selection_map.keys())
        crit_active = [cid for cid in critical_nodes if cid in active_set]
        if not crit_active:
            crit_active = list(active_set)
        if not crit_active:
            return set()

        # Prioritize critical nodes with multiple choices or highest duration
        def nodeScore(cid: int) -> float:
            num_choices = len(classes_by_id[cid].get("enodes", []))
            dur = end_times.get(cid, 0) - start_times.get(cid, 0)
            return (2.0 if num_choices > 1 else 1.0) * float(dur + 1)

        crit_active.sort(key=nodeScore, reverse=True)
        # Choose among top candidates
        sample_pool = crit_active[:max(3, len(crit_active) // 4)]
        seed = random.choice(sample_pool)

        unfrozen: Set[int] = {seed}
        frontier: List[int] = [seed]

        parents: Dict[int, List[int]] = defaultdict(list)
        children: Dict[int, List[int]] = defaultdict(list)
        for cid in active_set:
            e_idx = selection_map[cid]
            enode = classes_by_id[cid]["enodes"][e_idx]
            for ch in enode.get("children", []):
                if ch in active_set:
                    children[cid].append(ch)
                    parents[ch].append(cid)

        for _ in range(self.max_hops):
            next_frontier = []
            for curr in frontier:
                neighbors = parents.get(curr, []) + children.get(curr, [])
                for nxt in neighbors:
                    if nxt not in unfrozen:
                        unfrozen.add(nxt)
                        next_frontier.append(nxt)
                        if len(unfrozen) >= self.target_size:
                            return unfrozen
            frontier = next_frontier
            if not frontier:
                break

        return unfrozen


class MultiChoiceSelector(NeighborhoodSelector):
    """Targets clusters of eclasses that contain multiple kernel alternatives (fusions/rewrites)."""

    def __init__(self, target_size: int = 15):
        self.target_size = target_size

    def selectUnfrozenNodes(
        self,
        bucket: Dict[str, Any],
        selection_map: Dict[int, int],
        order: List[int],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        slack: Dict[int, int],
        critical_nodes: List[int],
        classes_by_id: Dict[int, Dict[str, Any]],
        iteration: int,
    ) -> Set[int]:
        active_set = set(selection_map.keys())
        multi_nodes = [
            cid for cid in order
            if cid in active_set and len(classes_by_id[cid].get("enodes", [])) > 1
        ]
        if not multi_nodes:
            return set()

        seed = random.choice(multi_nodes)
        unfrozen: Set[int] = {seed}

        # Expand along dataflow edges to collect connected multi-choice nodes
        for cid in multi_nodes:
            if len(unfrozen) >= self.target_size:
                break
            enode = classes_by_id[cid]["enodes"][selection_map[cid]]
            if any(ch in unfrozen for ch in enode.get("children", [])):
                unfrozen.add(cid)

        return unfrozen


class CompositeNeighborhoodSelector(NeighborhoodSelector):
    """Interleaves multiple selectors with configurable probabilities."""

    def __init__(
        self,
        selectors: Optional[List[Tuple[float, NeighborhoodSelector]]] = None,
    ):
        if selectors is None:
            self.selectors = [
                (0.60, CriticalPathSelector(target_size=20, max_hops=2)),
                (0.25, MultiChoiceSelector(target_size=15)),
                (0.15, RandomSubgraphSelector(target_size=15, max_hops=2)),
            ]
        else:
            self.selectors = selectors

    def selectUnfrozenNodes(
        self,
        bucket: Dict[str, Any],
        selection_map: Dict[int, int],
        order: List[int],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        slack: Dict[int, int],
        critical_nodes: List[int],
        classes_by_id: Dict[int, Dict[str, Any]],
        iteration: int,
    ) -> Set[int]:
        weights = [w for w, _ in self.selectors]
        chosen_selector = random.choices(
            [s for _, s in self.selectors], weights=weights, k=1
        )[0]
        return chosen_selector.selectUnfrozenNodes(
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


class NeuralNeighborhoodSelector(NeighborhoodSelector):
    """Plug-in template for a learned Graph Neural Network (GNN) neighborhood selector.
    
    Can be instantiated with a trained model checkpoint in the future.
    Falls back gracefully to CriticalPathSelector if model is not loaded.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.fallback = CriticalPathSelector()
        self.model = None
        if model_path and os.path.exists(model_path):
            self._loadModel(model_path)

    def _loadModel(self, model_path: str):
        # Placeholder for PyTorch/ONNX GNN model loading
        pass

    def selectUnfrozenNodes(
        self,
        bucket: Dict[str, Any],
        selection_map: Dict[int, int],
        order: List[int],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        slack: Dict[int, int],
        critical_nodes: List[int],
        classes_by_id: Dict[int, Dict[str, Any]],
        iteration: int,
    ) -> Set[int]:
        if self.model is None:
            return self.fallback.selectUnfrozenNodes(
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
        # If model is loaded, run GNN forward pass to sample unfrozen nodes
        return self.fallback.selectUnfrozenNodes(
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


# =============================================================================
# Iterative Subgraph LNS Solver
# =============================================================================

class OrtoolsLnsSolver:
    """Iterative Subgraph Large Neighborhood Search (LNS) Solver.
    
    Starts from the fast native CPU witness as incumbent, then repeatedly:
    1. Identifies a promising subgraph using a pluggable NeighborhoodSelector.
    2. Freezes all nodes outside the subgraph.
    3. Solves a fast, localized CP-SAT subproblem on the neighborhood.
    4. Accepts improvements and updates the incumbent plan.
    """

    alignment = 4096
    time_scale = 1000

    def __init__(
        self,
        problem_data: Dict[str, Any],
        selector: Optional[NeighborhoodSelector] = None,
    ):
        self.problem_data = problem_data
        self.buckets = problem_data.get("buckets", [])
        self.candidates = problem_data.get("candidates", [])
        self.mem_caps = problem_data.get("mem_caps", {})
        self.preallocated_buffers = problem_data.get("preallocated_buffers", [])
        self.cpu_hints = problem_data.get("cpu_hints", [])
        self.selector = selector or CompositeNeighborhoodSelector()
        
        max_time = problem_data.get("max_time_seconds")
        self.timeout_sec = float(max_time) if max_time and max_time > 0 else 30.0
        self.print_progress = bool(problem_data.get("print_progress", True))
        self.subproblem_timeout_sec = 2.0

    def computeSchedule(
        self,
        order: List[int],
        selection_map: Dict[int, int],
        classes_by_id: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[int, int], Dict[int, int], int]:
        """Calculates start and end times in microseconds and overall makespan."""
        start_times: Dict[int, int] = {}
        end_times: Dict[int, int] = {}
        engine_finish: Dict[str, int] = defaultdict(int)

        for cid in order:
            if cid not in selection_map or cid not in classes_by_id:
                continue
            e_idx = selection_map[cid]
            enode = classes_by_id[cid]["enodes"][e_idx]
            is_input = bool(enode.get("is_input"))
            is_cache = bool(enode.get("is_cache"))
            is_view = bool(enode.get("is_view"))

            cost = float(enode.get("cost", 0.0))
            if not math.isfinite(cost) or cost < 0:
                cost = 0.0
            duration = 0 if (is_input or is_cache or is_view) else max(1, math.ceil(cost * self.time_scale))

            children = enode.get("children", [])
            child_ready = max((end_times.get(ch, 0) for ch in children), default=0)

            engines = enode.get("engines") or [{"type": 0, "idx": 0}]
            eng_keys = [f"{eng['type']}:{eng['idx']}" for eng in engines]
            eng_ready = max((engine_finish[k] for k in eng_keys), default=0) if not (is_input or is_cache or is_view) else 0

            t_start = max(child_ready, eng_ready) if not is_input else 0
            t_end = t_start + duration

            start_times[cid] = t_start
            end_times[cid] = t_end

            if not (is_input or is_cache or is_view):
                for k in eng_keys:
                    engine_finish[k] = t_end

        makespan = max(end_times.values(), default=0)
        return start_times, end_times, makespan

    def computeCriticalPathAndSlack(
        self,
        order: List[int],
        selection_map: Dict[int, int],
        classes_by_id: Dict[int, Dict[str, Any]],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        makespan: int,
    ) -> Tuple[Dict[int, int], List[int]]:
        """Computes topological slack and identifies the critical path."""
        consumers: Dict[int, List[int]] = defaultdict(list)
        for cid in order:
            if cid in selection_map and cid in classes_by_id:
                e_idx = selection_map[cid]
                for ch in classes_by_id[cid]["enodes"][e_idx].get("children", []):
                    consumers[ch].append(cid)

        latest_end: Dict[int, int] = {}
        for cid in reversed(order):
            limits = [start_times[u] for u in consumers.get(cid, []) if u in start_times]
            latest_end[cid] = min(limits) if limits else makespan

        slack: Dict[int, int] = {}
        critical_nodes: List[int] = []
        for cid in order:
            if cid in end_times and cid in latest_end:
                s = latest_end[cid] - end_times[cid]
                slack[cid] = max(0, s)
                if s <= 0:
                    critical_nodes.append(cid)

        return slack, critical_nodes

    def allocateBuffers(
        self,
        order: List[int],
        selection_map: Dict[int, int],
        classes_by_id: Dict[int, Dict[str, Any]],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        preallocated_buffers: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Greedy First-Fit interval coloring for memory arenas."""
        prealloc_by_base = {
            item["base_eclass_id"]: item for item in preallocated_buffers
        }

        # Build view aliases
        def resolveBase(cid: int) -> int:
            curr = cid
            seen = set()
            while curr not in seen:
                seen.add(curr)
                if curr not in selection_map or curr not in classes_by_id:
                    break
                enode = classes_by_id[curr]["enodes"][selection_map[curr]]
                if enode.get("is_view") and enode.get("children"):
                    curr = enode["children"][0]
                else:
                    break
            return curr

        eclass_to_base = {cid: resolveBase(cid) for cid in order}

        # Track buffer lifetimes
        buffers_by_base: Dict[int, Dict[str, Any]] = {}
        next_buf_id = max((b.get("id", 0) for b in preallocated_buffers), default=0) + 1
        pos_map = {cid: idx for idx, cid in enumerate(order)}

        for cid in order:
            base = eclass_to_base[cid]
            base_cls = classes_by_id.get(base, classes_by_id[cid])
            base_id = base_cls["base_eclass_id"]

            if base not in buffers_by_base:
                if base_id in prealloc_by_base:
                    p = prealloc_by_base[base_id]
                    buf_id = p.get("buffer_id", next_buf_id)
                    off = int(p["offset"])
                    size = int(p["size"])
                    ms = p["mem_space"]
                    is_pre = True
                else:
                    buf_id = next_buf_id
                    next_buf_id += 1
                    off = 0
                    size = ((int(base_cls["size_bytes"]) + self.alignment - 1) // self.alignment) * self.alignment
                    ms = base_cls["mem_space"]
                    is_pre = (ms.get("type", 1) == 0)

                buffers_by_base[base] = {
                    "id": buf_id,
                    "base_cid": base,
                    "size": size,
                    "mem_space": ms,
                    "start": pos_map[cid],
                    "end": pos_map[cid] + 1,
                    "offset": off,
                    "is_preallocated": is_pre,
                }
            else:
                buffers_by_base[base]["end"] = max(
                    buffers_by_base[base]["end"], pos_map[cid] + 1
                )

        # Extend lifetimes to consumers
        for cid in order:
            e_idx = selection_map[cid]
            enode = classes_by_id[cid]["enodes"][e_idx]
            t_pos = pos_map[cid]
            for ch in enode.get("children", []):
                if ch in eclass_to_base:
                    ch_base = eclass_to_base[ch]
                    if ch_base in buffers_by_base:
                        buffers_by_base[ch_base]["end"] = max(
                            buffers_by_base[ch_base]["end"], t_pos + 1
                        )

        # First-fit allocation per memory space
        buf_list = list(buffers_by_base.values())
        by_ms: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for buf in buf_list:
            ms = buf["mem_space"]
            by_ms[f"{ms['type']}:{ms['idx']}"].append(buf)

        for ms_key, bufs in by_ms.items():
            prealloc_bufs = [b for b in bufs if b["is_preallocated"]]
            dyn_bufs = [b for b in bufs if not b["is_preallocated"]]
            placed = list(prealloc_bufs)
            sorted_dyn = sorted(dyn_bufs, key=lambda x: (x["start"], -x["size"]))

            for item in sorted_dyn:
                overlapping = [
                    p for p in placed
                    if max(p["start"], item["start"]) < min(p["end"], item["end"])
                ]
                forbidden = sorted(
                    [(p["offset"], p["offset"] + p["size"]) for p in overlapping],
                    key=lambda x: x[0],
                )
                cand_off = 0
                for f_start, f_end in forbidden:
                    if cand_off + item["size"] <= f_start:
                        break
                    if cand_off < f_end:
                        cand_off = ((f_end + self.alignment - 1) // self.alignment) * self.alignment
                item["offset"] = cand_off
                placed.append(item)

        eclass_to_buf = {str(cid): buffers_by_base[eclass_to_base[cid]]["id"] for cid in order}
        serialized_buffers = [
            {
                "id": b["id"],
                "mem_space": b["mem_space"],
                "size": b["size"],
                "start": b["start"],
                "end": b["end"],
                "offset": b["offset"],
            }
            for b in buf_list
        ]
        return serialized_buffers, eclass_to_buf

    def solveSubproblem(
        self,
        bucket: Dict[str, Any],
        unfrozen: Set[int],
        incumbent_selection: Dict[int, int],
        incumbent_order: List[int],
        start_times: Dict[int, int],
        end_times: Dict[int, int],
        classes_by_id: Dict[int, Dict[str, Any]],
        current_makespan: int,
    ) -> Optional[Dict[int, int]]:
        """Solves a fast CP-SAT subproblem on the unfrozen subgraph to improve local latency."""
        if not unfrozen:
            return None

        model = cp_model.CpModel()
        x: Dict[Tuple[int, int], Any] = {}
        t_start: Dict[int, Any] = {}
        t_end: Dict[int, Any] = {}
        u_list = list(unfrozen)

        # 1. Variables for unfrozen nodes
        for cid in u_list:
            cls = classes_by_id[cid]
            enodes = cls["enodes"]
            enode_vars = []
            for enode in enodes:
                e_idx = enode["enode_idx"]
                var = model.NewBoolVar(f"x_{cid}_{e_idx}")
                x[(cid, e_idx)] = var
                enode_vars.append(var)

            model.AddExactlyOne(enode_vars)
            t_start[cid] = model.NewIntVar(0, current_makespan, f"start_{cid}")
            t_end[cid] = model.NewIntVar(0, current_makespan, f"end_{cid}")

            # Link end time to selected duration
            for enode in enodes:
                e_idx = enode["enode_idx"]
                cost = float(enode.get("cost", 0.0))
                dur = 0 if (enode.get("is_input") or enode.get("is_cache") or enode.get("is_view")) else max(1, math.ceil(cost * self.time_scale))
                model.Add(t_end[cid] == t_start[cid] + dur).OnlyEnforceIf(x[(cid, e_idx)])

        # 2. Dependency constraints
        for cid in u_list:
            for enode in classes_by_id[cid]["enodes"]:
                e_idx = enode["enode_idx"]
                sel = x[(cid, e_idx)]
                for ch in enode.get("children", []):
                    if ch in unfrozen:
                        model.Add(t_end[ch] <= t_start[cid]).OnlyEnforceIf(sel)
                    elif ch in end_times:
                        # Boundary input: child is frozen outside U
                        model.Add(t_start[cid] >= end_times[ch]).OnlyEnforceIf(sel)

        # 3. Boundary outputs: consumers outside U must not be delayed
        for cid in u_list:
            # Check any consumer of cid outside U
            for other_cid in incumbent_order:
                if other_cid not in unfrozen and other_cid in incumbent_selection:
                    o_enode = classes_by_id[other_cid]["enodes"][incumbent_selection[other_cid]]
                    if cid in o_enode.get("children", []):
                        if other_cid in start_times:
                            model.Add(t_end[cid] <= start_times[other_cid])

        # 4. Engine non-overlap within unfrozen subgraph
        engines_intervals: Dict[str, List[Any]] = defaultdict(list)
        for cid in u_list:
            for enode in classes_by_id[cid]["enodes"]:
                if enode.get("is_input") or enode.get("is_cache") or enode.get("is_view"):
                    continue
                e_idx = enode["enode_idx"]
                cost = float(enode.get("cost", 0.0))
                dur = max(1, math.ceil(cost * self.time_scale))
                engines = enode.get("engines") or [{"type": 0, "idx": 0}]
                for eng in engines:
                    k = f"{eng['type']}:{eng['idx']}"
                    opt_interval = model.NewOptionalIntervalVar(
                        t_start[cid], dur, t_end[cid], x[(cid, e_idx)], f"eng_{k}_{cid}_{e_idx}"
                    )
                    engines_intervals[k].append(opt_interval)

        for k, intervals in engines_intervals.items():
            if len(intervals) > 1:
                model.AddNoOverlap(intervals)

        # 5. Objective: minimize latest completion of unfrozen nodes
        sub_makespan = model.NewIntVar(0, current_makespan, "sub_makespan")
        for cid in u_list:
            model.Add(sub_makespan >= t_end[cid])
        model.Minimize(sub_makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.subproblem_timeout_sec
        solver.parameters.num_workers = 4
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            new_selection = {}
            for cid in u_list:
                for enode in classes_by_id[cid]["enodes"]:
                    e_idx = enode["enode_idx"]
                    if solver.Value(x[(cid, e_idx)]) == 1:
                        new_selection[cid] = e_idx
                        break
            return new_selection

        return None

    def solve(self) -> Dict[str, Any]:
        if not self.buckets:
            raise ValueError("OR-Tools LNS requires at least one bucket")

        start_solve_time = time.time()
        extractions_res = []
        cached_nodes_res = []

        if self.candidates and not self.problem_data.get("disable_caching", False):
            # Maintain persistent candidates
            for cand in self.candidates:
                cached_nodes_res.append(
                    {"base_eclass_id": cand["base_eclass_id"], "mem_space": cand["mem_space"]}
                )

        for b_idx, bucket in enumerate(self.buckets):
            classes = bucket["classes"]
            classes_by_id = {cls["id"]: cls for cls in classes}

            # 1. Initialize Incumbent from CPU Witness Hint if available
            hint = self.cpu_hints[b_idx] if b_idx < len(self.cpu_hints) else None
            if hint and hint.get("selection_map") and hint.get("order"):
                incumbent_selection: Dict[int, int] = {
                    int(k): int(v) for k, v in hint["selection_map"].items()
                }
                incumbent_order: List[int] = [int(c) for c in hint["order"] if int(c) in incumbent_selection]
            else:
                # Fallback: topological sort with minimum-cost enode
                incumbent_selection = {
                    cls["id"]: 0 for cls in classes if cls.get("enodes")
                }
                incumbent_order = [cls["id"] for cls in classes]

            start_times, end_times, best_makespan = self.computeSchedule(
                incumbent_order, incumbent_selection, classes_by_id
            )
            initial_cost_ms = best_makespan / self.time_scale

            if self.print_progress:
                print(
                    f"[OrtoolsLNS] Bucket {bucket.get('bucket_idx', b_idx)}: "
                    f"Initial witness cost: {initial_cost_ms:.2f} ms ({len(incumbent_order)} active nodes)"
                )

            # 2. Iterative Subgraph LNS Loop
            iteration = 0
            improvements = 0

            while (time.time() - start_solve_time) < self.timeout_sec:
                iteration += 1
                slack, critical_nodes = self.computeCriticalPathAndSlack(
                    incumbent_order,
                    incumbent_selection,
                    classes_by_id,
                    start_times,
                    end_times,
                    best_makespan,
                )

                unfrozen = self.selector.selectUnfrozenNodes(
                    bucket,
                    incumbent_selection,
                    incumbent_order,
                    start_times,
                    end_times,
                    slack,
                    critical_nodes,
                    classes_by_id,
                    iteration,
                )

                if not unfrozen:
                    continue

                sub_res = self.solveSubproblem(
                    bucket,
                    unfrozen,
                    incumbent_selection,
                    incumbent_order,
                    start_times,
                    end_times,
                    classes_by_id,
                    best_makespan,
                )

                if sub_res is not None:
                    # Test candidate selection
                    cand_selection = dict(incumbent_selection)
                    cand_selection.update(sub_res)

                    # Re-toposort
                    adj: Dict[int, List[int]] = defaultdict(list)
                    in_degree: Dict[int, int] = {cid: 0 for cid in cand_selection}
                    for cid in cand_selection:
                        e_idx = cand_selection[cid]
                        for ch in classes_by_id[cid]["enodes"][e_idx].get("children", []):
                            if ch in cand_selection:
                                adj[ch].append(cid)
                                in_degree[cid] += 1

                    q = deque([cid for cid, deg in in_degree.items() if deg == 0])
                    cand_order = []
                    while q:
                        curr = q.popleft()
                        cand_order.append(curr)
                        for nxt in adj[curr]:
                            in_degree[nxt] -= 1
                            if in_degree[nxt] == 0:
                                q.append(nxt)

                    if len(cand_order) == len(cand_selection):
                        cand_starts, cand_ends, cand_makespan = self.computeSchedule(
                            cand_order, cand_selection, classes_by_id
                        )
                        if cand_makespan < best_makespan:
                            delta_pct = (best_makespan - cand_makespan) / best_makespan * 100.0
                            best_makespan = cand_makespan
                            incumbent_selection = cand_selection
                            incumbent_order = cand_order
                            start_times = cand_starts
                            end_times = cand_ends
                            improvements += 1
                            if self.print_progress:
                                print(
                                    f"[OrtoolsLNS] Iter {iteration}: Found improvement! "
                                    f"New cost: {best_makespan / self.time_scale:.2f} ms (-{delta_pct:.2f}%)"
                                )

            # 3. Final bufferization and packing
            buffers, eclass_to_buf = self.allocateBuffers(
                incumbent_order,
                incumbent_selection,
                classes_by_id,
                start_times,
                end_times,
                self.preallocated_buffers,
            )

            eclass_to_cost = {
                str(cid): float(classes_by_id[cid]["enodes"][incumbent_selection[cid]].get("cost", 0.0))
                for cid in incumbent_order
            }
            schedule_res = {
                str(cid): {"start": start_times.get(cid, 0), "end": end_times.get(cid, 0)}
                for cid in incumbent_order
            }

            final_cost_ms = best_makespan / self.time_scale
            if self.print_progress:
                print(
                    f"[OrtoolsLNS] Bucket {bucket.get('bucket_idx', b_idx)} complete. "
                    f"Ran {iteration} LNS iterations ({improvements} improvements). "
                    f"Final cost: {final_cost_ms:.2f} ms"
                )

            extractions_res.append(
                {
                    "cost": final_cost_ms,
                    "selection_map": {str(k): v for k, v in incumbent_selection.items()},
                    "order": incumbent_order,
                    "eclass_to_buf": eclass_to_buf,
                    "eclass_to_cost": eclass_to_cost,
                    "buffers": buffers,
                    "schedule": schedule_res,
                }
            )

        return {
            "solver": "ortools_full",  # Match session expectation for full solution
            "cached_nodes": cached_nodes_res,
            "extractions": extractions_res,
            "objective": sum(e["cost"] for e in extractions_res) * self.time_scale,
        }


def solveOrtools(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """Unified entry point for OR-Tools solving."""
    if cp_model is None:
        import subprocess
        import tempfile
        x64_candidates = [
            ".venvx64\\Scripts\\python.exe",
            ".venvx64/Scripts/python.exe",
            ".venvx64/bin/python",
        ]
        x64_py = None
        for c in x64_candidates:
            if os.path.exists(c):
                x64_py = c
                break
        with tempfile.TemporaryDirectory(prefix="ortools_lns_") as tmp_dir:
            prob_p = os.path.join(tmp_dir, "problem.json")
            sol_p = os.path.join(tmp_dir, "solution.json")
            with open(prob_p, "w", encoding="utf-8") as f:
                json.dump(problem_data, f)
            if x64_py:
                cmd = [x64_py, "ortools_lns.py", prob_p, sol_p]
            else:
                bin_path = os.path.join("tensor_graphs_cpp", "test_ortools_full.exe" if os.name == "nt" else "test_ortools_full")
                cmd = [bin_path, "--solve-json", prob_p, sol_p]
            subprocess.run(cmd, check=True, capture_output=True)
            with open(sol_p, "r", encoding="utf-8") as f:
                return json.load(f)

    use_lns = (
        os.environ.get("TENSOR_GRAPHS_USE_LNS") == "1"
        or problem_data.get("use_ortools_lns", False)
    )
    if use_lns:
        solver = OrtoolsLnsSolver(problem_data)
        return solver.solve()

    if problem_data.get("use_ortools_full", False):
        from ortools_full import OrtoolsSolver as FullOrtoolsSolver
        return FullOrtoolsSolver(problem_data).solve()

    # Default to LNS solver
    solver = OrtoolsLnsSolver(problem_data)
    return solver.solve()


def main():
    if len(sys.argv) < 3:
        print("Usage: python ortools_lns.py <problem.json> <solution.json>")
        sys.exit(1)

    problem_path = sys.argv[1]
    solution_path = sys.argv[2]

    with open(problem_path, "r", encoding="utf-8") as f:
        problem_data = json.load(f)

    solution_data = solveOrtools(problem_data)

    with open(solution_path, "w", encoding="utf-8") as f:
        json.dump(solution_data, f, indent=2)

    print(f"[ortools_lns] Solution successfully written to {solution_path}")


if __name__ == "__main__":
    main()
