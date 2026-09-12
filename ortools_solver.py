import json
import sys
from collections import deque
from typing import Any, Dict, List, Set, Tuple

from ortools.sat.python import cp_model


class OrtoolsSolver:
    """Unified CP-SAT optimization replacing CacheIterator, Extractor, DispatchIterator,
    BufferizeIterator, and MallocIterator.
    """

    def __init__(self, problem_data: Dict[str, Any]):
        self.problem_data = problem_data
        self.candidates = problem_data.get("candidates", [])
        self.buckets = problem_data.get("buckets", [])
        self.mem_caps = problem_data.get("mem_caps", {})
        self.preallocated_buffers = problem_data.get("preallocated_buffers", [])
        self.preallocated_lids = {
            item["logical_id"] for item in self.preallocated_buffers
        }
        self.preallocated_extents: Dict[str, int] = {}
        for item in self.preallocated_buffers:
            ms = item["mem_space"]
            ms_key = f"{ms['type']}{ms['idx']}"
            extent = int(item["offset"]) + int(item["size"])
            self.preallocated_extents[ms_key] = max(
                self.preallocated_extents.get(ms_key, 0), extent
            )

        min_compile_sec = problem_data.get("min_compile_seconds", 0.0)
        self.timeout_sec = max(30.0, float(min_compile_sec))
        self.print_progress = bool(problem_data.get("print_progress", True))

    def resolveViewAlias(
        self,
        cid: int,
        selection_map: Dict[int, int],
        classes: Dict[int, Dict[str, Any]],
    ) -> int:
        """Traces view alias chain back to the root tensor buffer."""
        curr = cid
        visited: Set[int] = set()
        while curr not in visited:
            visited.add(curr)
            if curr not in selection_map or curr not in classes:
                break
            enode = classes[curr]["enodes"][selection_map[curr]]
            if enode.get("is_view", False) and enode.get("children"):
                curr = enode["children"][0]
            else:
                break
        return curr

    def solve(self) -> Dict[str, Any]:
        # =========================================================================
        # Stage 1: Global Cache & Multi-Bucket E-Graph Extraction via CP-SAT
        # =========================================================================
        extract_model = cp_model.CpModel()

        # Cache candidate variables
        is_cached: Dict[int, Any] = {}
        for cand in self.candidates:
            lid = cand["logical_id"]
            is_cached[lid] = extract_model.NewBoolVar(f"cached_{lid}")

        # Enforce memory capacity for cache candidates per MemSpace
        cand_by_ms: Dict[str, List[Dict[str, Any]]] = {}
        for cand in self.candidates:
            ms = cand["mem_space"]
            ms_key = f"{ms['type']}{ms['idx']}"
            cand_by_ms.setdefault(ms_key, []).append(cand)

        for ms_key, cands_in_ms in cand_by_ms.items():
            cap = max(
                0,
                self.mem_caps.get(ms_key, 2**40)
                - self.preallocated_extents.get(ms_key, 0),
            )
            # Cache buffers are persistent across buckets, so they are not
            # interval-colored by extraction order.  Their only global
            # placement constraint is total reserved bytes in each memory
            # space.  Native finalization assigns the actual offsets together
            # with the input and intermediate buffers.
            cache_terms = [
                cand["size_bytes"] * is_cached[cand["logical_id"]]
                for cand in cands_in_ms
                if cand["logical_id"] not in self.preallocated_lids
            ]
            if cache_terms:
                extract_model.Add(sum(cache_terms) <= cap)

        # Per-bucket extraction variables and constraints
        bucket_active: Dict[Tuple[int, int], Any] = {}
        bucket_x: Dict[Tuple[int, int, int], Any] = {}
        total_cost_terms: List[Any] = []

        for b_dict in self.buckets:
            b = b_dict["bucket_idx"]
            b_weight = float(b_dict.get("weight", 1.0))
            root_id = b_dict["root_eclass_id"]
            classes_list = b_dict["classes"]
            clean_eclasses: Set[int] = set(b_dict.get("clean_eclasses", []))
            eclass_to_logical: Dict[int, int] = {
                int(k): int(v) for k, v in b_dict.get("eclass_to_logical", {}).items()
            }
            eclass_cache_lids: Dict[int, List[int]] = {}
            for cid, lid in eclass_to_logical.items():
                if lid in is_cached:
                    eclass_cache_lids.setdefault(cid, []).append(lid)

            # Map eclasses to enodes consuming them
            consumers: Dict[int, List[Tuple[int, int]]] = {}
            for cls in classes_list:
                cid = cls["id"]
                for enode in cls["enodes"]:
                    e_idx = enode["enode_idx"]
                    for ch in enode["children"]:
                        consumers.setdefault(ch, []).append((cid, e_idx))

            # Pass 1: create boolean variables
            for cls in classes_list:
                cid = cls["id"]
                bucket_active[(b, cid)] = extract_model.NewBoolVar(f"act_{b}_{cid}")
                for enode in cls["enodes"]:
                    e_idx = enode["enode_idx"]
                    bucket_x[(b, cid, e_idx)] = extract_model.NewBoolVar(
                        f"x_{b}_{cid}_{e_idx}"
                    )

            # Pass 2: add constraints
            for cls in classes_list:
                cid = cls["id"]
                act_var = bucket_active[(b, cid)]
                enode_vars = []

                for enode in cls["enodes"]:
                    e_idx = enode["enode_idx"]
                    cost = float(enode["cost"])
                    is_cache = enode.get("is_cache", False)
                    is_input = enode.get("is_input", False)
                    is_scatter = enode.get("is_scatter", False)
                    lid = enode.get("logical_id", -1)

                    x_var = bucket_x[(b, cid, e_idx)]
                    enode_vars.append(x_var)

                    if 0 < cost < 1e8:
                        scaled_cost = int(round(cost * b_weight * 1000.0))
                        total_cost_terms.append(scaled_cost * x_var)

                    if is_cache:
                        # A cache alternative is legal only for a selected
                        # candidate and only when this eclass is clean in the
                        # current bucket.
                        if cid not in clean_eclasses or lid not in is_cached:
                            extract_model.Add(x_var == 0)
                        else:
                            extract_model.Add(x_var <= is_cached[lid])
                    elif is_scatter:
                        # Scatter updates a cached backing eclass.  Requiring
                        # the eclass to be selected as cached prevents a
                        # fused path from silently bypassing the cache.
                        for ch in enode["children"]:
                            extract_model.Add(bucket_active[(b, ch)] >= x_var)
                        cache_vars = [is_cached[lid] for lid in eclass_cache_lids.get(cid, [])]
                        if cache_vars:
                            extract_model.Add(x_var <= sum(cache_vars))
                        else:
                            extract_model.Add(x_var == 0)
                    elif not is_input:
                        for ch in enode["children"]:
                            extract_model.Add(bucket_active[(b, ch)] >= x_var)

                extract_model.Add(sum(enode_vars) == act_var)

                if cid == root_id:
                    extract_model.Add(act_var == 1)
                else:
                    p_terms = [
                        bucket_x[(b, p, e)] for p, e in consumers.get(cid, [])
                    ]
                    if p_terms:
                        extract_model.Add(act_var <= sum(p_terms))
                    else:
                        extract_model.Add(act_var == 0)

            # If a logical node is selected for caching, every eclass carrying
            # that logical identity must be present in the extraction. This is
            # the CP-SAT equivalent of MissingCachedEClassRule.
            for cand in self.candidates:
                lid = cand["logical_id"]
                for cid, mapped_lid in eclass_to_logical.items():
                    if mapped_lid == lid:
                        extract_model.Add(bucket_active[(b, cid)] >= is_cached[lid])

        if total_cost_terms:
            extract_model.Minimize(sum(total_cost_terms))

        extract_solver = cp_model.CpSolver()
        extract_solver.parameters.max_time_in_seconds = self.timeout_sec
        extract_solver.parameters.num_workers = 12
        extract_solver.parameters.log_search_progress = self.print_progress
        extract_status = extract_solver.Solve(extract_model)

        if extract_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(
                f"[OrtoolsSolver] Extraction failed with status: {extract_solver.StatusName(extract_status)}"
            )
            return {"cached_nodes": [], "extractions": []}

        # Collect cached nodes
        cached_nodes_res = []
        for cand in self.candidates:
            lid = cand["logical_id"]
            if extract_solver.Value(is_cached[lid]) == 1:
                cached_nodes_res.append(
                    {"logical_id": lid, "mem_space": cand["mem_space"]}
                )

        # Build preallocated lookup: logical_id -> dict
        prealloc_by_lid: Dict[int, Dict[str, Any]] = {
            item["logical_id"]: item for item in self.preallocated_buffers
        }

        extractions_res = []

        # =========================================================================
        # Stage 2-4: Per-Bucket Dispatch, Bufferize, and Malloc
        # =========================================================================
        for b_dict in self.buckets:
            b = b_dict["bucket_idx"]
            root_id = b_dict["root_eclass_id"]
            classes_list = b_dict["classes"]
            classes_by_id = {cls["id"]: cls for cls in classes_list}
            eclass_to_logical = {
                int(k): v for k, v in b_dict.get("eclass_to_logical", {}).items()
            }

            # 1. Selection map
            selection_map: Dict[int, int] = {}
            eclass_to_cost: Dict[str, float] = {}
            for cid, cls in classes_by_id.items():
                if extract_solver.Value(bucket_active[(b, cid)]) == 1:
                    for enode in cls["enodes"]:
                        e_idx = enode["enode_idx"]
                        if extract_solver.Value(bucket_x[(b, cid, e_idx)]) == 1:
                            selection_map[cid] = e_idx
                            eclass_to_cost[str(cid)] = float(enode["cost"])
                            break

            # 2. Dispatch: Topological Sort (Kahn's Algorithm)
            in_degree: Dict[int, int] = {cid: 0 for cid in selection_map}
            adj: Dict[int, List[int]] = {cid: [] for cid in selection_map}

            for cid, e_idx in selection_map.items():
                enode = classes_by_id[cid]["enodes"][e_idx]
                for ch in enode["children"]:
                    if ch in selection_map:
                        adj[ch].append(cid)
                        in_degree[cid] += 1

            q = deque([cid for cid, deg in in_degree.items() if deg == 0])
            order: List[int] = []
            while q:
                curr = q.popleft()
                order.append(curr)
                for nxt in adj[curr]:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        q.append(nxt)

            if len(order) < len(selection_map):
                # Cycle fallback: append remaining
                remaining = set(selection_map.keys()) - set(order)
                order.extend(list(remaining))

            # 3. Bufferize: View Aliasing & Lifetime Computation
            # Map each eclass to its buffer base
            eclass_to_base: Dict[int, int] = {}
            for cid in order:
                eclass_to_base[cid] = self.resolveViewAlias(
                    cid, selection_map, classes_by_id
                )

            # Assign buffers with globally unique IDs
            buffers_by_base: Dict[int, Dict[str, Any]] = {}
            eclass_to_buf: Dict[str, int] = {}
            used_buf_ids: Set[int] = {
                item.get("buffer_id", 0) for item in self.preallocated_buffers
            }
            next_buf_id = max(used_buf_ids, default=0) + 1

            # Compute execution step for each node in order
            time_in_order = {cid: idx for idx, cid in enumerate(order)}

            for cid in order:
                base = eclass_to_base[cid]
                if base not in buffers_by_base:
                    # Check if preallocated
                    lid = eclass_to_logical.get(base, -1)
                    if lid != -1 and lid in prealloc_by_lid:
                        p_info = prealloc_by_lid[lid]
                        buf_id = p_info.get("buffer_id", next_buf_id)
                        offset = p_info["offset"]
                        size = p_info["size"]
                        ms = p_info["mem_space"]
                        is_prealloc = True
                    else:
                        buf_id = next_buf_id
                        next_buf_id += 1
                        offset = 0
                        size = classes_by_id[base]["size_bytes"]
                        ms = classes_by_id[base]["mem_space"]
                        is_prealloc = (ms.get("type", 1) == 0)  # STORAGE is preallocated

                    buffers_by_base[base] = {
                        "id": buf_id,
                        "base_cid": base,
                        "size": size,
                        "mem_space": ms,
                        "start": time_in_order[cid],
                        "end": time_in_order[cid] + 1,
                        "offset": offset,
                        "is_preallocated": is_prealloc,
                    }
                else:
                    buffers_by_base[base]["end"] = max(
                        buffers_by_base[base]["end"], time_in_order[cid] + 1
                    )

                eclass_to_buf[str(cid)] = buffers_by_base[base]["id"]

            # Extend buffer lifetimes to cover all consumers
            for cid in order:
                e_idx = selection_map[cid]
                enode = classes_by_id[cid]["enodes"][e_idx]
                t_cons = time_in_order[cid]
                for ch in enode["children"]:
                    if ch in eclass_to_base:
                        ch_base = eclass_to_base[ch]
                        if ch_base in buffers_by_base:
                            buffers_by_base[ch_base]["end"] = max(
                                buffers_by_base[ch_base]["end"], t_cons + 1
                            )

            buffer_list = list(buffers_by_base.values())

            # 4. Malloc: First-fit Interval Coloring & CP-SAT Strip Packing
            # Group buffers by MemSpace
            buffers_by_ms: Dict[str, List[Dict[str, Any]]] = {}
            for buf in buffer_list:
                ms = buf["mem_space"]
                ms_key = f"{ms['type']}{ms['idx']}"
                buffers_by_ms.setdefault(ms_key, []).append(buf)

            for ms_key, bufs in buffers_by_ms.items():
                cap = self.mem_caps.get(ms_key, 2**40)
                # First-fit interval coloring for non-preallocated buffers
                prealloc_bufs = [b for b in bufs if b["is_preallocated"]]
                dyn_bufs = [b for b in bufs if not b["is_preallocated"]]

                placed = list(prealloc_bufs)
                sorted_dyn = sorted(dyn_bufs, key=lambda x: (x["start"], -x["size"]))

                for b_item in sorted_dyn:
                    # Overlapping placed buffers in time
                    overlapping = [
                        p
                        for p in placed
                        if max(p["start"], b_item["start"]) < min(p["end"], b_item["end"])
                    ]
                    forbidden = sorted(
                        [(p["offset"], p["offset"] + p["size"]) for p in overlapping],
                        key=lambda x: x[0],
                    )
                    cand_off = 0
                    for f_start, f_end in forbidden:
                        if cand_off + b_item["size"] <= f_start:
                            break
                        if cand_off < f_end:
                            cand_off = (f_end + 4095) // 4096 * 4096
                    b_item["offset"] = cand_off
                    placed.append(b_item)

            # Clean buffer objects for JSON serialization
            serialized_buffers = []
            for buf in buffer_list:
                serialized_buffers.append(
                    {
                        "id": buf["id"],
                        "mem_space": buf["mem_space"],
                        "size": buf["size"],
                        "start": buf["start"],
                        "end": buf["end"],
                        "offset": buf["offset"],
                    }
                )

            total_bucket_cost = sum(eclass_to_cost.values())
            extractions_res.append(
                {
                    "cost": total_bucket_cost,
                    "selection_map": {str(k): v for k, v in selection_map.items()},
                    "order": order,
                    "eclass_to_buf": eclass_to_buf,
                    "eclass_to_cost": eclass_to_cost,
                    "buffers": serialized_buffers,
                }
            )

        return {
            "cached_nodes": cached_nodes_res,
            "extractions": extractions_res,
        }


def solveOrtools(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    if problem_data.get("use_ortools_full", False):
        from ortools_full import OrtoolsSolver as FullOrtoolsSolver

        return FullOrtoolsSolver(problem_data).solve()
    solver = OrtoolsSolver(problem_data)
    return solver.solve()


def main():
    if len(sys.argv) < 3:
        print("Usage: python ortools_solver.py <problem.json> <solution.json>")
        sys.exit(1)

    problem_path = sys.argv[1]
    solution_path = sys.argv[2]

    with open(problem_path, "r") as f:
        problem_data = json.load(f)

    solution_data = solveOrtools(problem_data)

    with open(solution_path, "w") as f:
        json.dump(solution_data, f, indent=2)

    print(f"[ortools_solver] Solution successfully written to {solution_path}")


if __name__ == "__main__":
    main()
