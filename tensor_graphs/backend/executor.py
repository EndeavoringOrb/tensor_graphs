# tensor_graphs/backend/executor.py
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import math
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import TensorSignature, get_buffer_size
from ..ir.graph import GraphEncoder
from ..ops import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from .memory import MemoryManager
from .registry import KernelRegistry
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
        # Persistent cache across runs: Key = (node_name, shape_tuple) -> (Kernel, is_inplace)
        self._kernel_cache: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], Any] = {}

        # Determine output directory
        output_dir = RECORD_KERNEL_LAUNCHES_FOLDER
        os.makedirs(output_dir, exist_ok=True)

        # Find next available .jsonl file
        run_num = 0
        while os.path.exists(os.path.join(output_dir, f"{run_num}.jsonl")):
            run_num += 1

        self.kernel_launch_filename = os.path.join(output_dir, f"{run_num}.jsonl")

    def _calculate_dirty_ratio(
        self, regions: List[Tuple[slice, ...]], shape: Tuple[int, ...]
    ) -> float:
        """
        Calculates the ratio of the dirty volume to the total tensor volume,
        correctly handling overlapping regions using coordinate compression.
        """
        if not regions or not shape:
            return 0.0

        # 1. Normalize and Validate Regions
        clean_regions = []
        for reg in regions:
            if len(reg) != len(shape):
                continue

            norm_slices = []
            valid = True
            for i, s in enumerate(reg):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else shape[i]

                # Clamp to shape bounds
                start = max(0, start)
                stop = min(shape[i], stop)

                if start >= stop:
                    valid = False
                    break
                norm_slices.append(slice(start, stop))

            if valid:
                clean_regions.append(tuple(norm_slices))

        if not clean_regions:
            return 0.0

        total_vol = math.prod(shape)

        # Optimization: If single region, no overlap possible
        if len(clean_regions) == 1:
            r = clean_regions[0]
            vol = math.prod(s.stop - s.start for s in r)
            return vol / total_vol

        # 2. Coordinate Compression for N-Dimensional Overlap Handling
        # Collect all unique boundaries per dimension
        dim = len(shape)
        dim_coords = [set() for _ in range(dim)]

        for reg in clean_regions:
            for d, s in enumerate(reg):
                dim_coords[d].add(s.start)
                dim_coords[d].add(s.stop)

        # Sort coordinates to establish the compressed grid
        sorted_coords = [sorted(list(s)) for s in dim_coords]
        grid_shape = tuple(len(c) - 1 for c in sorted_coords)

        if any(g <= 0 for g in grid_shape):
            return 0.0

        # Map coordinate value to index in the compressed grid
        coord_to_idx = [{v: i for i, v in enumerate(c)} for c in sorted_coords]

        # Create a boolean grid representing the compressed space
        occupied = np.zeros(grid_shape, dtype=bool)

        # Map each region to the compressed grid and mark occupied cells
        for reg in clean_regions:
            grid_slices = []
            for d, s in enumerate(reg):
                s_idx = coord_to_idx[d][s.start]
                e_idx = coord_to_idx[d][s.stop]
                grid_slices.append(slice(s_idx, e_idx))
            occupied[tuple(grid_slices)] = True

        # 3. Calculate Union Volume
        # Find indices of all occupied cells in the compressed grid
        indices = np.argwhere(occupied)
        if indices.size == 0:
            return 0.0

        # Calculate the size of each segment in each dimension
        intervals = [np.diff(c) for c in sorted_coords]

        # Vectorized volume calculation:
        # For every occupied cell (index), look up the segment lengths and multiply them.
        # gathered_lengths[d] contains the lengths of the segments for that dimension
        # corresponding to the occupied cells.
        gathered_lengths = [intervals[d][indices[:, d]] for d in range(dim)]

        # Product of lengths across dimensions for each cell
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
        """Record kernel launch details to a .jsonl file."""

        # Create record
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

        # Append to .jsonl file
        with open(self.kernel_launch_filename, "a") as f:
            f.write(json.dumps(record, cls=GraphEncoder) + "\n")

    def _update_inputs(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, List[Tuple[slice, ...]]]:
        """
        Updates input buffers and returns the detected dirty regions (bounding boxes) for each input.
        """
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

            # Compute actual dirty regions
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

    def _find_best_bucket(self, input_diffs: Dict[str, List[Tuple[slice, ...]]]):
        if not input_diffs:
            return None, None

        key_items = {}
        for name, region_list in input_diffs.items():
            node = self.graph.nodes_map.get(name)
            if not node or not region_list:
                continue

            box = region_list[0]  # Bounding box
            canonical_box = []
            shape = node.shape or ()

            for d, sl in enumerate(box):
                start = sl.start if sl.start is not None else 0
                stop = sl.stop if sl.stop is not None else shape[d]
                dim_len = shape[d]

                # Start searching from the smallest possible power-of-2 length
                target_len = self._next_power_of_2(stop - start)

                found = False
                while target_len <= self._next_power_of_2(dim_len):
                    # Calculate the aligned start position for this zoom level
                    # This aligns with 0, target_len, 2*target_len...
                    bucket_start = (start // target_len) * target_len
                    bucket_end = min(bucket_start + target_len, dim_len)

                    # Check if the dirty region [start:stop] fits inside this aligned bucket
                    if bucket_start <= start and bucket_end >= stop:
                        canonical_box.append((bucket_start, bucket_end))
                        found = True
                        break

                    # If it doesn't fit (e.g. spans across a tile boundary),
                    # we must zoom out to the next power of 2.
                    target_len *= 2

                if not found:
                    # Fallback to full dimension if something went wrong
                    canonical_box.append((0, dim_len))

            key_items[name] = (tuple(canonical_box),)

        sorted_items = sorted(key_items.items())
        key = tuple(sorted_items)
        return self.dirty_cache.get(key)

    @profile
    def run(self, inputs: Dict[str, Any]) -> Any:
        self.mem.step()
        with Timer("[Executor.run] update inputs"):
            # Update inputs and get actual dirty diffs
            input_diffs = self._update_inputs(inputs)

        bucket = None
        if self.dirty_cache:
            with Timer("[Executor.run] cache lookup"):
                bucket = self._find_best_bucket(input_diffs)

        cached_regions = bucket["regions"] if bucket else {}
        cached_input_slices = bucket["input_slices"] if bucket else {}

        with Timer("[Executor.run] initializing"):
            # Initial reference counts calculated during compilation
            static_refs = self.graph.ref_counts

            # --- Pass 1: Forward Propagation & Cache Check ---
            # Determine the state of every node before we decide what to run
            node_states = {}  # name -> {'is_dirty': bool, 'in_cache': bool}

            for inst in self.graph.instructions:
                node = self.graph.nodes_map[inst.node_name]

                if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                    continue

                if bucket and node.name in cached_regions:
                    # Use cached dirty region
                    reg_data = cached_regions[node.name]
                    if reg_data:
                        # Deserialize: List[List[(s,e)]] -> List[Tuple[slice...]]
                        deserialized_reg = []
                        for box in reg_data:
                            slices = tuple(slice(s, e) for s, e in box)
                            deserialized_reg.append(slices)
                        node.dirty_region = deserialized_reg
                    else:
                        node.dirty_region = None
                else:
                    # Fallback if no bucket or node not in bucket (e.g. auto_copy)
                    # For auto_copy nodes, we can try to inherit from parent
                    if node.op_type == OpType.COPY_TO and node.name.startswith(
                        "auto_copy"
                    ):
                        p_name = inst.input_node_names[0]
                        p_node = self.graph.nodes_map.get(p_name)
                        if p_node:
                            node.dirty_region = p_node.dirty_region
                        else:
                            node.dirty_region = DirtyPropagator.propagate(node, inputs)
                    else:
                        node.dirty_region = DirtyPropagator.propagate(node, inputs)

                is_dirty = node.dirty_region is not None

                # Check if the buffer currently exists on the assigned device
                dev_hint = node.backend.value if node.backend else "cpu"
                in_cache = self.mem.has(node.name, dev_hint)

                node_states[node.name] = {"is_dirty": is_dirty, "in_cache": in_cache}

            # --- Pass 2: Backward Liveness Analysis ---
            # Determine which nodes are actually "needed" to produce the root output.
            # This allows us to "prune" nodes that don't contribute to the current result.
            needed_nodes = set()

            if self.graph.instructions:
                root_name = self.graph.instructions[-1].node_name
                needed_nodes.add(root_name)

            for inst in reversed(self.graph.instructions):
                name = inst.node_name
                if name not in node_states:
                    continue

                state = node_states[name]

                if name in needed_nodes:
                    # If a node is dirty or not in cache, we MUST compute it.
                    # To compute it, we need all of its parents.
                    must_compute = state["is_dirty"] or not state["in_cache"]

                    if must_compute:
                        for p_name in inst.input_node_names:
                            needed_nodes.add(p_name)

            # --- Pre-Pass: Lock Expected Cache Hits ---
            # Ensure that nodes we plan to Skip (reuse from cache) are not evicted
            # by intermediate allocations before we reach them.
            for inst in self.graph.instructions:
                name = inst.node_name
                if name not in needed_nodes:
                    continue

                node = self.graph.nodes_map[name]
                if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                    continue

                state = node_states[name]
                # Use same condition as main loop for Skip path
                if not state["is_dirty"] and state["in_cache"]:
                    self.mem.prepare_allocation(
                        node, node.size_bytes, initial_refs=static_refs[name]
                    )

        # --- Pass 3: Execution ---
        counters = {
            "skip": 0,
            "full": 0,
            "part": 0,
            "pruned": 0,
            "kernel_switch": 0,
            "inplace": 0,
        }
        partial_ratio_sum = 0

        with tqdm(
            self.graph.instructions,
            desc="graph inst",
            disable=not DEBUG_EXECUTION,
        ) as pbar:
            for inst in pbar:
                if DEBUG_EXECUTION and DEBUG_DETAILED:
                    print(f"[Executor.run] Executing {inst}")
                node = self.graph.nodes_map[inst.node_name]

                if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                    continue

                name = node.name
                state = node_states[name]
                is_needed = name in needed_nodes

                if not is_needed:
                    counters["pruned"] += 1
                    for p_name in inst.input_node_names:
                        self.mem.release(p_name)
                    continue

                dev_hint = node.backend.value if node.backend else "cpu"

                # Check for in-place signal from compiler
                is_inplace = inst.inplace_input_index is not None
                compute_regions: List[Tuple[slice, ...]] = []
                pending_output_allocation = False

                # Determine Computation Mode
                if state["is_dirty"] and state["in_cache"]:
                    counters["part"] += 1
                    if DEBUG_EXECUTION:
                        partial_ratio = (
                            self._calculate_dirty_ratio(node.dirty_region, node.shape)
                            if node.dirty_region and node.shape
                            else 1.0
                        )
                        partial_ratio_sum += partial_ratio
                    compute_regions = [
                        r for r in node.dirty_region if any(s.stop > s.start for s in r)
                    ]
                    self.mem.prepare_allocation(
                        node, node.size_bytes, initial_refs=static_refs[name]
                    )

                elif not state["in_cache"]:
                    counters["full"] += 1
                    # DEFER ALLOCATION: We don't know if we can do in-place or need new allocation
                    # until we resolve the kernel (which depends on concrete input shapes).
                    pending_output_allocation = True
                    compute_regions = [tuple(slice(None) for _ in (node.shape or ()))]

                else:  # clean and in cache
                    counters["skip"] += 1
                    self.mem.prepare_allocation(
                        node, node.size_bytes, initial_refs=static_refs[name]
                    )

                # Execute Kernel
                if compute_regions:
                    # Get cached input slices if available
                    node_cached_input_slices = cached_input_slices.get(name)

                    for i, region_slice in enumerate(compute_regions):
                        is_full_region = all(s == slice(None) for s in region_slice)
                        input_slice_regions = []

                        if is_full_region:
                            input_slice_regions = [None] * len(inst.input_node_names)
                        else:
                            # Try to use cached input requirements for this box index
                            if node_cached_input_slices and i < len(
                                node_cached_input_slices
                            ):
                                cached_reqs = node_cached_input_slices[i]
                                # Deserialize: List[List[(s,e)]] or None
                                if cached_reqs:
                                    for req in cached_reqs:
                                        if req:
                                            if len(req) > 0:
                                                box_data = req[0]
                                                slices = tuple(
                                                    slice(s, e) for s, e in box_data
                                                )
                                                input_slice_regions.append(slices)
                                            else:
                                                input_slice_regions.append(None)
                                        else:
                                            input_slice_regions.append(None)
                                else:
                                    # Fallback if cache structure mismatch (should not happen if key matched)
                                    query_region = [region_slice]
                                    reqs = DirtyPropagator.get_input_slices(
                                        node, query_region, inputs
                                    )
                                    for req in reqs:
                                        input_slice_regions.append(
                                            req[0] if req else None
                                        )
                            else:
                                # Fallback (e.g. auto_copy node not in cache)
                                query_region = [region_slice]
                                reqs = DirtyPropagator.get_input_slices(
                                    node, query_region, inputs
                                )
                                for req in reqs:
                                    input_slice_regions.append(req[0] if req else None)

                        kernel_inputs = []
                        concrete_shapes = []

                        for inp_idx, p_name in enumerate(inst.input_node_names):
                            p_node = self.graph.nodes_map[p_name]
                            full_view = self.mem.get_view(p_node)
                            sl = input_slice_regions[inp_idx]

                            if sl is None or node.op_type in (OpType.RESHAPE,):
                                view = full_view
                            else:
                                view = full_view[sl]

                            kernel_inputs.append(view)
                            concrete_shapes.append(view.shape)

                        # Kernel selection (Handle potential shape-specific kernels)
                        selected_kernel = inst.kernel
                        current_inplace = is_inplace

                        if not is_full_region or (
                            concrete_shapes and concrete_shapes[0] != node.shape
                        ):
                            cache_key = (name, tuple(concrete_shapes))
                            if cache_key in self._kernel_cache:
                                selected_kernel, current_inplace = self._kernel_cache[
                                    cache_key
                                ]
                            else:
                                input_sigs = []
                                for idx, shape in enumerate(concrete_shapes):
                                    p_node = self.graph.nodes_map[
                                        inst.input_node_names[idx]
                                    ]
                                    sig = TensorSignature(
                                        p_node.dtype, shape, node.backend
                                    )
                                    input_sigs.append(sig)

                                better_kernel = KernelRegistry.select_best_kernel(
                                    node.op_type,
                                    input_sigs,
                                    node.backend,
                                    node.dtype,
                                    inst.inplace_input_index is not None,
                                )
                                if better_kernel:
                                    selected_kernel, current_inplace = better_kernel
                                    counters["kernel_switch"] += 1
                                self._kernel_cache[cache_key] = (
                                    selected_kernel,
                                    current_inplace,
                                )

                        # LATE BINDING MEMORY ALLOCATION
                        # Now that we know the exact kernel and if it supports in-place
                        if pending_output_allocation:
                            if current_inplace:
                                src_name = inst.input_node_names[
                                    inst.inplace_input_index
                                ]
                                self.mem.transfer_ownership(src_name, name)
                            else:
                                self.mem.prepare_allocation(
                                    node,
                                    node.size_bytes,
                                    initial_refs=static_refs[name],
                                )

                            if not state["is_dirty"]:
                                node.dirty_region = None

                            pending_output_allocation = False

                        if current_inplace:
                            counters["inplace"] += 1

                        # Run Kernel
                        out_view = self.mem.get_view(node)
                        out_slice = (
                            out_view
                            if node.op_type in (OpType.RESHAPE,)
                            else out_view[region_slice]
                        )

                        start_time = time.perf_counter()
                        selected_kernel(kernel_inputs, [out_slice], inst.attrs)
                        node.compute_cost = (time.perf_counter() - start_time) * 1000

                        # Record kernel launch if enabled
                        if RECORD_KERNEL_LAUNCHES:
                            self._record_kernel_launch(
                                node_name=name,
                                op_type=node.op_type,
                                input_shapes=concrete_shapes,
                                output_shape=node.shape or (0,),
                                compute_time_ms=node.compute_cost,
                                is_partial=not is_full_region,
                                backend=dev_hint,
                                attrs=inst.attrs,
                                is_inplace=current_inplace,
                            )

                # Release Parents
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
