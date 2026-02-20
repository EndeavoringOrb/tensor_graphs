# tensor_graphs/backend/executor.py
import numpy as np
from typing import Dict, Any, List, Tuple
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import TensorSignature
from ..ops import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from .memory import MemoryManager
from .registry import KernelRegistry
from ..config import DEBUG_EXECUTION, DEBUG_DETAILED


class Executor:
    def __init__(self, compiled_graph: CompiledGraph, memory_manager: MemoryManager):
        self.graph = compiled_graph
        self.mem = memory_manager
        self.last_inputs: Dict[str, Any] = {}

        # AOT Bucket storage: (start, stop) -> list of (kernel, inputs, outputs, attrs)
        self.buckets: Dict[Tuple[int, int], List[Tuple[Any, List, List, Dict]]] = {}
        self._primary_input: str = ""  # Auto-detected primary input name
        self._bucket_dim: int = -1  # Auto-detected dimension to bucket on

    def compile(self):
        """
        Pre-compiles execution plans for power-of-2 intervals.
        Auto-detects the primary varying input by scanning the graph's transient
        INPUT nodes and selecting the one with the largest non-batch dimension.
        """
        # --- Auto-detect primary input and bucket dimension ---
        best_input_name = None
        best_dim_idx = -1
        best_dim_size = 0

        for node_name, node in self.graph.nodes_map.items():
            if node.op_type != OpType.INPUT:
                continue
            if node.storage_type == StorageType.PERSISTENT:
                continue
            if not node.shape or len(node.shape) < 2:
                continue

            # Find the largest non-batch dimension (skip dim 0 = batch)
            for d in range(1, len(node.shape)):
                dim_size = node.shape[d]
                if dim_size is not None and dim_size > best_dim_size:
                    best_dim_size = dim_size
                    best_dim_idx = d
                    best_input_name = node_name

        if best_input_name is None or best_dim_size <= 1:
            if DEBUG_EXECUTION:
                print("[Executor] No suitable input found for AOT bucketing, skipping.")
            return

        self._primary_input = best_input_name
        self._bucket_dim = best_dim_idx
        max_dim_size = best_dim_size

        if DEBUG_EXECUTION:
            print(f"[Executor] Auto-detected primary input: '{best_input_name}' dim={best_dim_idx} size={max_dim_size}")

        # 1. Lock in the memory arena
        self.mem.plan_static_memory(self.graph)

        # Generate intervals: power-of-2 chunks covering [0, max_dim_size)
        intervals = set()
        chunk_size = max_dim_size
        while chunk_size >= 1:
            for start in range(0, max_dim_size, chunk_size):
                stop = min(start + chunk_size, max_dim_size)
                intervals.add((start, stop))
            chunk_size //= 2

        if DEBUG_EXECUTION:
            print(f"[Executor] Compiling {len(intervals)} AOT buckets...")

        for start, stop in sorted(intervals):
            flat_plan = []

            # A. Reset dirty state on all graph nodes
            for node in self.graph.nodes_map.values():
                node.dirty_region = None

            # B. Mock dirty region on the primary input
            for node_name, node in self.graph.nodes_map.items():
                if node.op_type != OpType.INPUT:
                    continue
                if node.storage_type == StorageType.PERSISTENT:
                    continue
                if not node.shape:
                    continue

                ndim = len(node.shape)
                if node_name == best_input_name:
                    # Apply the interval to the bucket dimension
                    mock_slices = []
                    for d in range(ndim):
                        if d == best_dim_idx:
                            mock_slices.append(slice(start, stop))
                        else:
                            mock_slices.append(slice(0, node.shape[d]) if node.shape[d] else slice(None))
                    node.dirty_region = [tuple(mock_slices)]
                # Non-primary inputs: leave clean (None)

            # C. Forward Propagation: compute dirty regions for all nodes
            for inst in self.graph.instructions:
                node = self.graph.nodes_map[inst.node_name]
                if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                    continue
                node.dirty_region = DirtyPropagator.propagate(node, None)

            # D. Backward Liveness Pruning
            needed_nodes = set()
            if self.graph.instructions:
                needed_nodes.add(self.graph.instructions[-1].node_name)

            for inst in reversed(self.graph.instructions):
                if inst.node_name in needed_nodes:
                    node = self.graph.nodes_map[inst.node_name]
                    if node.dirty_region is not None:
                        for p_name in inst.input_node_names:
                            needed_nodes.add(p_name)

            # E. Build Flat Plan with pre-sliced static views
            for inst in self.graph.instructions:
                node = self.graph.nodes_map[inst.node_name]
                if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                    continue
                if inst.node_name not in needed_nodes or node.dirty_region is None:
                    continue

                # Get input slices from backward propagation
                req_in_slices = DirtyPropagator.get_input_slices(
                    node, node.dirty_region, None
                )

                kernel_inputs = []
                concrete_shapes = []
                for i, p_name in enumerate(inst.input_node_names):
                    p_node = self.graph.nodes_map[p_name]
                    dev = self.mem._resolve_device(p_node)
                    buf = self.mem.buffers[dev]

                    full_view = buf.static_views.get(p_name)
                    if full_view is None:
                        full_view = self.mem.get_view(p_node)

                    in_region = req_in_slices[i] if i < len(req_in_slices) else None

                    if in_region is None or node.op_type in (OpType.RESHAPE,):
                        kernel_inputs.append(full_view)
                    else:
                        sl = in_region[0] if in_region else None
                        if sl is not None:
                            kernel_inputs.append(full_view[sl])
                        else:
                            kernel_inputs.append(full_view)
                    concrete_shapes.append(kernel_inputs[-1].shape)

                # Slice Output View
                dev = self.mem._resolve_device(node)
                buf = self.mem.buffers[dev]
                out_full_view = buf.static_views.get(inst.node_name)
                if out_full_view is None:
                    out_full_view = self.mem.get_view(node)

                if node.op_type in (OpType.RESHAPE,):
                    kernel_outputs = [out_full_view]
                else:
                    out_sl = node.dirty_region[0] if node.dirty_region else None
                    if out_sl is not None:
                        kernel_outputs = [out_full_view[out_sl]]
                    else:
                        kernel_outputs = [out_full_view]

                # Select Kernel based on pre-sliced shapes
                input_sigs = [
                    TensorSignature(self.graph.nodes_map[n].dtype, inp.shape, node.backend)
                    for n, inp in zip(inst.input_node_names, kernel_inputs)
                ]

                result = KernelRegistry.select_best_kernel(
                    node.op_type, input_sigs, node.backend, node.dtype,
                    inst.inplace_input_index is not None,
                )
                kernel = result[0] if result else inst.kernel

                flat_plan.append((kernel, kernel_inputs, kernel_outputs, inst.attrs))

            self.buckets[(start, stop)] = flat_plan

        if DEBUG_EXECUTION:
            total_ops = sum(len(plan) for plan in self.buckets.values())
            print(f"[Executor] AOT compilation complete: {len(self.buckets)} buckets, {total_ops} total ops")

    def run(self, inputs: Dict[str, Any]) -> Any:
        """Execute with pre-compiled AOT buckets."""
        # 1. Write inputs to the static arena and find the dirty interval
        actual_start = float('inf')
        actual_stop = -1

        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node or node.storage_type == StorageType.PERSISTENT:
                continue

            # Diff against last input for bucket selection
            old_data = self.last_inputs.get(name)
            diff_regions = DirtyPropagator.get_diff(old_data, data)

            # Only the primary input's bucket dimension drives bucket selection
            if diff_regions and name == self._primary_input and self._bucket_dim >= 0:
                box = diff_regions[0]
                if self._bucket_dim < len(box):
                    sl = box[self._bucket_dim]
                    s = sl.start if sl.start is not None else 0
                    e = sl.stop if sl.stop is not None else (
                        data.shape[self._bucket_dim] if hasattr(data, 'shape') else 0
                    )
                    actual_start = min(actual_start, s)
                    actual_stop = max(actual_stop, e)

            # Write to static memory
            dev = self.mem._resolve_device(node)
            buf = self.mem.buffers.get(dev)
            if buf and name in buf.static_views:
                view = buf.static_views[name]
                if isinstance(view, np.ndarray):
                    src = data.cpu().numpy() if hasattr(data, 'cpu') else np.asarray(data)
                    if src.shape == view.shape:
                        view[...] = src
                    else:
                        view[...] = src.reshape(view.shape)
                elif hasattr(view, 'copy_'):
                    import torch
                    src = data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
                    view.copy_(src.to(view.device, dtype=view.dtype).reshape(view.shape))
            else:
                self.mem.write(node, data)

            # Cache for next diff
            if hasattr(data, 'clone'):
                self.last_inputs[name] = data.clone()
            elif isinstance(data, np.ndarray):
                self.last_inputs[name] = data.copy()
            else:
                self.last_inputs[name] = data

        # 2. Find Smallest Enclosing Bucket
        root_name = self.graph.instructions[-1].node_name
        root_node = self.graph.nodes_map[root_name]
        dev = self.mem._resolve_device(root_node)
        buf = self.mem.buffers[dev]

        if actual_start == float('inf'):
            # Inputs didn't change, return cached output
            if root_name in buf.static_views:
                return buf.static_views[root_name]
            return self.mem.get_view(root_node)

        best_plan = None
        min_size = float('inf')

        for (b_start, b_stop), plan in self.buckets.items():
            if b_start <= actual_start and b_stop >= actual_stop:
                size = b_stop - b_start
                if size < min_size:
                    min_size = size
                    best_plan = plan

        if best_plan is None:
            raise RuntimeError(
                f"No AOT bucket covers dirty region ({actual_start}, {actual_stop}). "
                f"Available buckets: {sorted(self.buckets.keys())}"
            )

        # 3. Execute Flat Plan
        for kernel, kernel_inputs, kernel_outputs, attrs in best_plan:
            kernel(kernel_inputs, kernel_outputs, attrs)

        # 4. Return Output
        if root_name in buf.static_views:
            return buf.static_views[root_name]
        return self.mem.get_view(root_node)
