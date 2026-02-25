import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import math
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import get_buffer_size
from ..ir.graph import GraphEncoder
from ..ops import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from .memory import MemoryManager
from ..config import (
    DEBUG_EXECUTION,
    DEBUG_DETAILED,
    RECORD_KERNEL_LAUNCHES,
    RECORD_KERNEL_LAUNCHES_FOLDER,
)
from ..tools.timer import Timer
from tqdm import tqdm
import os
import json
from datetime import datetime
from line_profiler import profile


class Executor:
    def __init__(
        self,
        compiled_graph: CompiledGraph,
        memory_manager: MemoryManager,
        dirty_cache: Optional[Dict[str, Any]] = None,
    ):
        self.graph = compiled_graph
        self.mem = memory_manager
        self.dirty_cache = dirty_cache or {}
        self.last_inputs: Dict[str, Any] = {}
        # Removed _kernel_cache as we now rely on pre-computed kernels in dirty_cache

        output_dir = RECORD_KERNEL_LAUNCHES_FOLDER
        os.makedirs(output_dir, exist_ok=True)

        run_num = 0
        while os.path.exists(os.path.join(output_dir, f"{run_num}.jsonl")):
            run_num += 1

        self.kernel_launch_filename = os.path.join(output_dir, f"{run_num}.jsonl")

    def _calculate_dirty_ratio(
        self, regions: List[Tuple[Tuple[int, int], ...]], shape: Tuple[int, ...]
    ) -> float:
        if not regions or not shape:
            return 0.0

        clean_regions = []
        for reg in regions:
            if len(reg) != len(shape):
                continue

            norm_slices = []
            valid = True
            for i, (start, stop) in enumerate(reg):
                start = max(0, start)
                stop = min(shape[i], stop)

                if start >= stop:
                    valid = False
                    break
                norm_slices.append((start, stop))

            if valid:
                clean_regions.append(tuple(norm_slices))

        if not clean_regions:
            return 0.0

        total_vol = math.prod(shape)

        if len(clean_regions) == 1:
            r = clean_regions[0]
            vol = math.prod(stop - start for start, stop in r)
            return float(vol / total_vol)

        dim = len(shape)
        dim_coords = [set() for _ in range(dim)]

        for reg in clean_regions:
            for d, s in enumerate(reg):
                dim_coords[d].add(s[0])
                dim_coords[d].add(s[1])

        sorted_coords = [sorted(list(s)) for s in dim_coords]
        grid_shape = tuple(len(c) - 1 for c in sorted_coords)

        if any(g <= 0 for g in grid_shape):
            return 0.0

        coord_to_idx = [{v: i for i, v in enumerate(c)} for c in sorted_coords]

        occupied = np.zeros(grid_shape, dtype=bool)

        for reg in clean_regions:
            grid_slices = []
            for d, s in enumerate(reg):
                s_idx = coord_to_idx[d][s[0]]
                e_idx = coord_to_idx[d][s[1]]
                grid_slices.append(slice(s_idx, e_idx))
            occupied[tuple(grid_slices)] = True

        indices = np.argwhere(occupied)
        if indices.size == 0:
            return 0.0

        intervals = [np.diff(c) for c in sorted_coords]
        gathered_lengths = [intervals[d][indices[:, d]] for d in range(dim)]
        cell_volumes = np.prod(gathered_lengths, axis=0)
        dirty_vol = np.sum(cell_volumes)

        return dirty_vol / total_vol

    def _record_kernel_launch(
        self,
        node_name: str,
        op_type: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        compute_time_ms: float,
        is_partial: bool,
        backend: str,
        attrs: Dict[str, Any],
        is_inplace: bool,
    ):
        record = {
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
            "op_type": op_type,
            "mode": "PARTIAL" if is_partial else "FULL",
            "backend": backend,
            "input_shapes": input_shapes,
            "output_shape": output_shape,
            "compute_time_ms": round(compute_time_ms, 6),
            "attrs": attrs if attrs else {},
            "inplace": is_inplace,
        }

        with open(self.kernel_launch_filename, "a") as f:
            f.write(json.dumps(record, cls=GraphEncoder) + "\n")

    def _update_inputs(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, List[Tuple[Tuple[int, int], ...]]]:
        input_diffs = {}
        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node:
                continue

            if node.storage_type == StorageType.PERSISTENT:
                continue

            size = get_buffer_size(node.dtype, data=data)
            initial_refs = self.graph.ref_counts.get(node.name, 0)
            self.mem.prepare_allocation(node, size, initial_refs=initial_refs)

            old_data = self.last_inputs.get(name)

            dirty_region = DirtyPropagator.get_diff(old_data, data)
            node.dirty_region = dirty_region
            if dirty_region:
                input_diffs[name] = dirty_region

            self.mem.write(node, data)

            if hasattr(data, "clone"):
                self.last_inputs[name] = data.clone()
            elif isinstance(data, np.ndarray):
                self.last_inputs[name] = data.copy()
            else:
                self.last_inputs[name] = data

        return input_diffs

    def _next_power_of_2(self, x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def _find_best_bucket(
        self, input_diffs: Dict[str, List[Tuple[Tuple[int, int], ...]]]
    ):
        if not input_diffs:
            return None, None

        key_items = {}
        for name, region_list in input_diffs.items():
            node = self.graph.nodes_map.get(name)
            if not node or not region_list:
                continue

            box = region_list[0]
            canonical_box = []
            shape = node.shape or ()

            for d, (start, stop) in enumerate(box):
                dim_len = shape[d]

                target_len = self._next_power_of_2(stop - start)

                found = False
                while target_len <= self._next_power_of_2(dim_len):
                    bucket_start = (start // target_len) * target_len
                    bucket_end = min(bucket_start + target_len, dim_len)

                    if bucket_start <= start and bucket_end >= stop:
                        canonical_box.append((bucket_start, bucket_end))
                        found = True
                        break
                    target_len *= 2

                if not found:
                    canonical_box.append((0, dim_len))

            key_items[name] = (tuple(canonical_box),)

        sorted_items = sorted(key_items.items())
        key = tuple(sorted_items)
        return self.dirty_cache.get(key)

    @profile
    def run(self, inputs: Dict[str, Any]) -> Any:
        self.mem.step()
        with Timer("[Executor.run] update inputs"):
            input_diffs = self._update_inputs(inputs)

        bucket = None
        if self.dirty_cache:
            with Timer("[Executor.run] cache lookup"):
                bucket = self._find_best_bucket(input_diffs)
        if bucket is None:
            raise ValueError("[Executor.run] Could not find bucket")

        cached_regions = bucket["regions"]
        cached_input_slices = bucket["input_slices"]
        cached_kernels = bucket["kernels"]

        with Timer("[Executor.run] initializing"):
            static_refs = self.graph.ref_counts

            # --- Pass 1: Backward Liveness + in_cache ---
            needed_nodes = set()
            node_states = {}

            if self.graph.instructions:
                root_name = self.graph.instructions[-1].node_name
                needed_nodes.add(root_name)

            for inst in reversed(self.graph.instructions):
                name = inst.node_name
                if name not in node_states:
                    node_states[name] = {
                        "is_dirty": None,
                        "in_cache": None,
                    }

                state = node_states[name]

                if name in needed_nodes:
                    if state["in_cache"] is None:
                        node = self.graph.nodes_map[name]
                        dev_hint = node.backend.value if node.backend else "cpu"
                        state["in_cache"] = self.mem.has(name, dev_hint)

                    state["is_dirty"] = cached_regions[name] is not None
                    must_compute = state["is_dirty"] or not state["in_cache"]

                    if must_compute:
                        for p_name in inst.input_node_names:
                            needed_nodes.add(p_name)

            # --- Pass 2: Dry Run ---
            execution_plan = []
            counters = {
                "skip": 0,
                "full": 0,
                "part": 0,
                "pruned": 0,
                "kernel_switch": 0,
                "inplace": 0,
            }
            partial_ratio_sum = 0

            for inst in self.graph.instructions:
                node = self.graph.nodes_map[inst.node_name]

                if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                    continue

                name = node.name
                state = node_states[name]
                is_needed = name in needed_nodes

                plan = {
                    "inst": inst,
                    "node": node,
                    "is_needed": is_needed,
                    "mem_action": None,
                    "transfer_src": None,
                    "tasks": [],
                }

                if not is_needed:
                    counters["pruned"] += 1
                    execution_plan.append(plan)
                    continue

                dev_hint = node.backend.value if node.backend else "cpu"

                # In-place status for FULL compute (default)
                is_inplace = inst.inplace_input_index is not None
                compute_regions = []

                # Determine Computation Mode
                if state["is_dirty"] and state["in_cache"]:
                    counters["part"] += 1
                    if DEBUG_EXECUTION:
                        partial_ratio = (
                            self._calculate_dirty_ratio(
                                cached_regions[name], node.shape
                            )
                            if cached_regions[name] and node.shape
                            else 1.0
                        )
                        partial_ratio_sum += partial_ratio
                    compute_regions = [
                        r
                        for r in cached_regions[name]
                        if any(stop > start for start, stop in r)
                    ]
                    plan["mem_action"] = "prepare"

                elif not state["in_cache"]:
                    counters["full"] += 1
                    if is_inplace:
                        plan["mem_action"] = "transfer"
                        plan["transfer_src"] = inst.input_node_names[
                            inst.inplace_input_index
                        ]
                        counters["inplace"] += 1
                    else:
                        plan["mem_action"] = "prepare"
                    compute_regions = [tuple((0, dim) for dim in (node.shape or ()))]

                else:  # clean and in cache
                    counters["skip"] += 1
                    plan["mem_action"] = "prepare"

                # Lock Clean Cache Hits
                if not state["is_dirty"] and state["in_cache"]:
                    self.mem.prepare_allocation(
                        node, node.size_bytes, initial_refs=static_refs[name]
                    )

                # Execute Kernel Dry Run
                if compute_regions:
                    # Retrieve pre-cached data for this node
                    node_cached_input_slices = cached_input_slices.get(name)
                    node_cached_kernels = cached_kernels.get(name)

                    for i, region_slice in enumerate(compute_regions):
                        is_full_region = all(
                            start == 0 and stop == dim
                            for (start, stop), dim in zip(
                                region_slice, node.shape or ()
                            )
                        )
                        input_slice_regions = []

                        # Initialize with defaults from instruction
                        selected_kernel = inst.kernel

                        if is_full_region:
                            input_slice_regions = [None] * len(inst.input_node_names)
                        else:
                            # PARTIAL COMPUTE PATH
                            # 1. Get Input Slices
                            input_slice_regions = node_cached_input_slices[i]

                            # 2. Select Kernel
                            # Use pre-cached kernel if available
                            selected_kernel = node_cached_kernels[i]

                        plan["tasks"].append(
                            {
                                "region_slice": region_slice,
                                "is_full_region": is_full_region,
                                "input_slice_regions": input_slice_regions,
                                "selected_kernel": selected_kernel,
                                "current_inplace": is_inplace,
                                "backend": dev_hint,
                            }
                        )

                execution_plan.append(plan)

        # --- Pass 3: Execution ---
        with tqdm(
            execution_plan,
            desc="graph inst",
            disable=not DEBUG_EXECUTION,
        ) as pbar:
            for plan in pbar:
                inst = plan["inst"]
                node = plan["node"]
                name = node.name

                if DEBUG_EXECUTION and DEBUG_DETAILED:
                    print(f"[Executor.run] Executing {inst}")

                if not plan["is_needed"]:
                    for p_name in inst.input_node_names:
                        self.mem.release(p_name)
                    continue

                if not plan["tasks"]:
                    if plan["mem_action"] == "prepare":
                        self.mem.prepare_allocation(
                            node,
                            node.size_bytes,
                            initial_refs=static_refs[name],
                        )
                else:
                    pending_mem_action = plan["mem_action"]

                    for task in plan["tasks"]:
                        region_slice = task["region_slice"]
                        input_slice_regions = task["input_slice_regions"]
                        selected_kernel = task["selected_kernel"]

                        # get computed input views (must happen before transfer_ownership)
                        kernel_inputs = []
                        for inp_idx, p_name in enumerate(inst.input_node_names):
                            p_node = self.graph.nodes_map[p_name]
                            full_view = self.mem.get_view(p_node)
                            sl_list = input_slice_regions[inp_idx]

                            if sl_list is None or node.op_type in (OpType.RESHAPE,):
                                view = full_view
                            else:
                                sl = sl_list[0]
                                view_slices = tuple(slice(s, e) for s, e in sl)
                                view = full_view[view_slices]

                            kernel_inputs.append(view)

                        # perform allocation/transfer
                        if pending_mem_action:
                            if pending_mem_action == "transfer":
                                self.mem.transfer_ownership(plan["transfer_src"], name)
                            elif pending_mem_action == "prepare":
                                self.mem.prepare_allocation(
                                    node,
                                    node.size_bytes,
                                    initial_refs=static_refs[name],
                                )
                            pending_mem_action = None

                        # get output view
                        out_view = self.mem.get_view(node)
                        if node.op_type in (OpType.RESHAPE,):
                            out_slice = out_view
                        else:
                            out_view_slices = tuple(
                                slice(s, e) for s, e in region_slice
                            )
                            out_slice = out_view[out_view_slices]

                        # run kernel
                        start_time = time.perf_counter()
                        selected_kernel(kernel_inputs, [out_slice], inst.attrs)
                        node.compute_cost = (time.perf_counter() - start_time) * 1000

                        if RECORD_KERNEL_LAUNCHES:
                            self._record_kernel_launch(
                                node_name=name,
                                op_type=node.op_type,
                                input_shapes=[inp.shape for inp in kernel_inputs],
                                output_shape=node.shape or (0,),
                                compute_time_ms=node.compute_cost,
                                is_partial=not task["is_full_region"],
                                backend=task["backend"],
                                attrs=inst.attrs,
                                is_inplace=task["current_inplace"],
                            )

                # release parent
                for p_name in inst.input_node_names:
                    self.mem.release(p_name)

                # Update Progress Bar Stats
                if DEBUG_EXECUTION:
                    total_ops = counters["skip"] + counters["part"] + counters["full"]
                    counters["p_sum"] = (
                        f"{partial_ratio_sum / counters['part']:.2f}"
                        if counters["part"] > 0
                        else "N/A"
                    )
                    counters["p_cache"] = (
                        f"{(counters['skip'] + (counters['part'] - partial_ratio_sum)) / (total_ops if total_ops > 0 else 1):.2f}"
                    )
                    pbar.set_postfix(counters)

        root_name = self.graph.instructions[-1].node_name
        return self.mem.get_view(self.graph.nodes_map[root_name])
