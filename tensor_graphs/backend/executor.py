import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import math
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType, TensorSignature
from ..ops import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from .memory import MemoryManager
from .registry import KernelRegistry
from ..config import DEBUG_EXECUTION, DEBUG_DETAILED
from tqdm import tqdm


class Executor:
    def __init__(self, compiled_graph: CompiledGraph, memory_manager: MemoryManager):
        self.graph = compiled_graph
        self.mem = memory_manager
        self.last_inputs: Dict[str, Any] = {}
        # Persistent cache across runs: Key = (node_name, shape_tuple) -> Kernel
        self._kernel_cache: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], Any] = {}

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

    def _update_inputs(self, inputs: Dict[str, Any]) -> None:
        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node:
                continue

            if node.storage_type == StorageType.PERSISTENT:
                continue

            size = (
                data.nbytes
                if hasattr(data, "nbytes")
                else ((data.numel() if hasattr(data, "numel") else 1) * 4)
            )
            self.mem.prepare_allocation(node, size)

            old_data = self.last_inputs.get(name)
            node.dirty_region = DirtyPropagator.get_diff(old_data, data)

            self.mem.write(node, data)

            if hasattr(data, "clone"):
                self.last_inputs[name] = data.clone()
            elif isinstance(data, np.ndarray):
                self.last_inputs[name] = data.copy()
            else:
                self.last_inputs[name] = data

    def run(self, inputs: Dict[str, Any]) -> Any:
        self.mem.step()
        self._update_inputs(inputs)
        current_refs = self.graph.ref_counts.copy()

        # --- Pass 1: Forward Propagation & Cache Check ---
        node_states = {}  # name -> {'is_dirty': bool, 'in_cache': bool}

        for inst in self.graph.instructions:
            node = self.graph.nodes_map[inst.node_name]

            if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                continue

            # Propagate dirty region from parents
            node.dirty_region = DirtyPropagator.propagate(node, inputs)
            is_dirty = node.dirty_region is not None

            # Check if exists in memory
            dev_hint = node.backend.value if node.backend else "cpu"
            in_cache = self.mem.has(node.name, dev_hint)

            node_states[node.name] = {"is_dirty": is_dirty, "in_cache": in_cache}

        # --- Pass 2: Backward Liveness Analysis ---
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
                must_compute = state["is_dirty"] or not state["in_cache"]

                if must_compute:
                    for p_name in inst.input_node_names:
                        needed_nodes.add(p_name)

        # --- Pass 3: Execution ---
        counters = {
            "no_op": 0,
            "skip": 0,
            "full": 0,
            "part": 0,
            "pruned": 0,
            "kernel_switch": 0,
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
                    counters["no_op"] += 1
                    continue

                name = node.name
                state = node_states[name]
                is_needed = name in needed_nodes

                compute_regions: List[Tuple[slice, ...]] = []

                if not is_needed:
                    counters["pruned"] += 1
                elif state["is_dirty"] and state["in_cache"]:
                    counters["part"] += 1
                    
                    if DEBUG_EXECUTION:
                        # Updated ratio calculation with overlap handling
                        if node.dirty_region and node.shape:
                            partial_ratio = self._calculate_dirty_ratio(
                                node.dirty_region, node.shape
                            )
                        else:
                            partial_ratio = 1.0
                        
                        partial_ratio_sum += partial_ratio
                    
                    # node.dirty_region is List[Tuple[slice, ...]]
                    # Filter out empty slices just in case
                    compute_regions = [
                        r for r in node.dirty_region if any(s.stop > s.start for s in r)
                    ]
                elif not state["in_cache"]:
                    counters["full"] += 1
                    # Full recompute: One big region
                    compute_regions = [tuple(slice(None) for _ in (node.shape or ()))]
                    if not state["is_dirty"]:
                        node.dirty_region = None
                else:
                    # Cached & Clean & Needed -> Skip
                    counters["skip"] += 1
                    self.mem.lock(node)

                # Update Stats
                if DEBUG_EXECUTION:
                    total_ops = counters["skip"] + counters["part"] + counters["full"]
                    counters["p_sum"] = (
                        f"{partial_ratio_sum / counters['part']:.2f}"
                        if counters["part"] > 0
                        else "N/A"
                    )
                    counters["p_cache"] = (
                        f"{(counters['skip'] + (counters['part'] - partial_ratio_sum)) / total_ops:.2f}"
                        if total_ops > 0
                        else "0.00"
                    )
                    pbar.set_postfix(counters)

                if compute_regions:
                    # --- PREPARE OUTPUT BUFFER ---
                    size_bytes = math.prod(node.shape or ()) * 4
                    if node.dtype == DType.FP16:
                        size_bytes //= 2
                    elif node.dtype == DType.BOOL:
                        size_bytes //= 4
                    self.mem.prepare_allocation(node, size_bytes)

                    # --- LOCK INPUTS (Once for all regions) ---
                    for p_name in inst.input_node_names:
                        p_node = self.graph.nodes_map[p_name]
                        self.mem.lock(p_node)

                    # --- EXECUTE REGIONS ---
                    for region_slice in compute_regions:
                        start_time = time.perf_counter()

                        # 1. Determine Input Requirements for this specific region
                        # Backward prop returns List[DirtyRegion], we take the first box (index 0)
                        # of the first DirtyRegion for each input.

                        # If region is "Full", inputs are "Full" (None)
                        is_full_region = all(s == slice(None) for s in region_slice)
                        input_slice_regions: List[Optional[Tuple[slice, ...]]] = []

                        if is_full_region:
                            input_slice_regions = [None] * len(inst.input_node_names)
                        else:
                            # Query backward propagation for this specific region
                            # query_region format: List[Tuple[slice,...]]
                            query_region = [region_slice]
                            reqs = DirtyPropagator.get_input_slices(
                                node, query_region, inputs
                            )

                            for req in reqs:
                                if req is None or not req:
                                    input_slice_regions.append(None)
                                else:
                                    # Take the first box of the requirement list
                                    input_slice_regions.append(req[0])

                        # 2. Prepare Views
                        kernel_inputs = []
                        concrete_shapes = []

                        for i, p_name in enumerate(inst.input_node_names):
                            p_node = self.graph.nodes_map[p_name]
                            full_view = self.mem.get_view(p_node)

                            sl = input_slice_regions[i]
                            if sl is None or node.op_type in (OpType.RESHAPE,):
                                view = full_view
                            else:
                                view = full_view[sl]

                            kernel_inputs.append(view)
                            concrete_shapes.append(view.shape)

                        # 3. Dynamic Kernel Selection
                        selected_kernel = inst.kernel

                        # Only check for partial/changed shapes if we want optimized kernels
                        # If region is full, we rely on the pre-compiled kernel unless shapes changed
                        if not is_full_region or (concrete_shapes[0] != node.shape):
                            cache_key = (name, tuple(concrete_shapes))

                            if cache_key in self._kernel_cache:
                                selected_kernel = self._kernel_cache[cache_key]
                            else:
                                # Build signatures for the specific shapes we have
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
                                    node.op_type, input_sigs, node.backend, node.dtype
                                )

                                if better_kernel:
                                    selected_kernel = better_kernel
                                    counters["kernel_switch"] += 1

                                self._kernel_cache[cache_key] = selected_kernel

                        # 4. Run Kernel
                        out_view = self.mem.get_view(node)
                        if node.op_type in (OpType.RESHAPE,):
                            out_slice = out_view
                        else:
                            out_slice = out_view[region_slice]

                        selected_kernel(kernel_inputs, [out_slice], inst.attrs)
                        node.compute_cost = (time.perf_counter() - start_time) * 1000

                    # --- RELEASE INPUTS (Once after all regions) ---
                    for p_name in inst.input_node_names:
                        current_refs[p_name] -= 1
                        if current_refs[p_name] == 0:
                            p_node = self.graph.nodes_map[p_name]
                            if p_node.storage_type == StorageType.TRANSIENT:
                                if DEBUG_EXECUTION and DEBUG_DETAILED:
                                    print(f"[Executor.run] unlocking {p_node}")
                                self.mem.unlock(p_node)

        root_name = self.graph.instructions[-1].node_name
        return self.mem.get_view(self.graph.nodes_map[root_name])