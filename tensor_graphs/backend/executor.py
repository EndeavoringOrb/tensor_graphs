import numpy as np
import torch
import time
from typing import Dict, Any, Optional
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType
from ..ops.atomic_types import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from ..ir.node import CachePolicy
from ..config import *
from .cache import CacheManager


class Executor:
    """
    Unified Executor that runs a CompiledGraph with incremental execution support.
    Integrates DirtyRegion propagation and CacheManager to selectively execute
    or restore data.
    """

    def __init__(
        self,
        compiled_graph: CompiledGraph,
        cache_manager: Optional[CacheManager] = None,
    ):
        self.graph = compiled_graph
        self.cache_manager = cache_manager
        self.buffers: Dict[str, Any] = {}
        self.last_inputs: Dict[str, Any] = {}

        # 1. Allocate persistent buffers for the entire graph memory space
        self._allocate_buffers()

        # 2. Pre-calculate views
        self.input_views = {
            name: self._get_view(offset, name)
            for name, offset in self.graph.input_offsets.items()
        }
        self.output_views = {
            name: self._get_view(offset, name)
            for name, offset in self.graph.output_offsets.items()
        }

    def _allocate_buffers(self):
        device_sizes: Dict[str, int] = {}
        for alloc in self.graph.buffer_allocations.values():
            end = alloc.offset + alloc.size_bytes
            device_sizes[alloc.device] = max(device_sizes.get(alloc.device, 0), end)

        for device, size in device_sizes.items():
            if "cuda" in device or "gpu" in device:
                # Use standard tensor for GPU
                self.buffers[device] = torch.zeros(
                    size, dtype=torch.uint8, device=device
                )
            else:
                self.buffers[device] = np.zeros(size, dtype=np.uint8)

    def _map_dtype_np(self, dtype: DType) -> Any:
        if dtype == DType.FP32:
            return np.float32
        if dtype == DType.INT32:
            return np.int32
        if dtype == DType.FP16:
            return np.float16
        if dtype == DType.BOOL:
            return np.bool_
        return np.float32

    def _map_dtype_torch(self, dtype: DType) -> Any:
        if dtype == DType.FP32:
            return torch.float32
        if dtype == DType.INT32:
            return torch.int32
        if dtype == DType.FP16:
            return torch.float16
        if dtype == DType.BOOL:
            return torch.bool
        return torch.float32

    def _get_view(self, offset: int, name: str, device: Optional[str] = None) -> Any:
        meta = self.graph.node_metadata[name]
        alloc = self.graph.buffer_allocations[name]
        dev = device or alloc.device
        buf = self.buffers[dev]
        size_bytes = alloc.size_bytes

        if isinstance(buf, np.ndarray):
            np_dtype = self._map_dtype_np(meta.dtype)
            # Create a view into the buffer
            return np.ndarray(
                meta.shape, dtype=np_dtype, buffer=buf.data, offset=offset
            )
        elif isinstance(buf, torch.Tensor):
            torch_dtype = self._map_dtype_torch(meta.dtype)
            # Create a view
            slice_t = buf[offset : offset + size_bytes]
            return slice_t.view(torch_dtype).view(meta.shape)

        return None

    def copy_to_buffer(self, name: str, data: Any):
        """Writes data into the buffer at the node's allocated offset."""
        if name not in self.graph.buffer_allocations:
            return

        alloc = self.graph.buffer_allocations[name]
        offset = alloc.offset
        buf = self.buffers[alloc.device]
        meta = self.graph.node_metadata[name]

        if isinstance(buf, np.ndarray):
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=self._map_dtype_np(meta.dtype))

            # Flatten and copy bytes
            data_view = (
                data.astype(self._map_dtype_np(meta.dtype)).view(np.uint8).flatten()
            )
            buf[offset : offset + len(data_view)] = data_view

        elif isinstance(buf, torch.Tensor):
            t_dtype = self._map_dtype_torch(meta.dtype)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=t_dtype, device=alloc.device)
            else:
                data = data.to(dtype=t_dtype, device=alloc.device)

            data_view = data.reshape(-1).view(torch.uint8).flatten()
            buf[offset : offset + len(data_view)] = data_view

    def load_weights(self, weights: Dict[str, Any]):
        """Initializes persistent weights."""
        for name, data in weights.items():
            self.copy_to_buffer(name, data)
            # Weights are implicitly clean initially

    def _update_inputs(self, inputs: Dict[str, Any]):
        """Diffs inputs and updates persistent buffers."""
        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node:
                continue
            # TODO: We don't need to diff weights, they won't change. But we need to make sure they are copied to buffer the first time

            # Calculate diff
            old_data = self.last_inputs.get(name)
            node.dirty_region = DirtyPropagator.get_diff(old_data, data)

            # Update buffer
            self.copy_to_buffer(name, data)

            # Store history
            if hasattr(data, "clone"):
                self.last_inputs[name] = data.clone()
            elif isinstance(data, np.ndarray):
                self.last_inputs[name] = data.copy()
            else:
                self.last_inputs[name] = data

    def run(self, inputs: Dict[str, Any]) -> Any:
        # 1. Update Inputs & Set Input Dirty Flags
        self._update_inputs(inputs)

        executed_count = 0
        restored_count = 0
        skipped_count = 0

        # 2. Execute Instructions (Topological Order)
        for inst in self.graph.instructions:
            node_name = inst.node_name
            node = self.graph.nodes_map[node_name]

            # --- A. Propagate Dirty Region ---
            # Inputs already have dirty_region set.
            # Constants have None (Clean).
            if node.op_type not in (OpType.INPUT, OpType.CONSTANT):
                node.dirty_region = DirtyPropagator.propagate(node, inputs)

            # --- B. Determine Execution Strategy ---

            is_dirty = node.dirty_region is not None
            is_persistent = node.storage_type == StorageType.PERSISTENT

            # We treat Transient buffers as "Logically Lost" between runs unless restored.
            # However, Persistent buffers retain data.

            should_compute = False
            should_restore = False
            compute_region = node.dirty_region  # Default to propagating partial dirty

            # Check Cache Availability
            has_cache = self.cache_manager and self.cache_manager.has(node)

            if is_dirty:
                node.dirty_count += 1
                if is_persistent:
                    # Persistent buffer + Dirty -> Compute only the dirty slice (Patching)
                    should_compute = True
                else:
                    # Transient + Dirty
                    # We need the full buffer to be valid for children.
                    # Since transient buffer is recyclable, it's likely garbage.
                    # Option 1: Restore Full from Cache, then Compute Partial.
                    # Option 2: Compute Full.
                    if has_cache:
                        should_restore = True
                        should_compute = True  # Compute partial on top of restored
                    else:
                        should_compute = True
                        compute_region = None  # Force Full Recompute (None -> Full in get_input_slices logic usually, or explicit full)
                        if compute_region is None:
                            # Create explicit full slice for clarity if Propagator returns None for clean
                            # But here we want full dirty
                            compute_region = (
                                (slice(None),) * len(node.shape)
                                if node.shape
                                else (slice(None),)
                            )

            else:  # Clean
                if is_persistent:
                    # Persistent + Clean -> Skip
                    skipped_count += 1
                else:
                    # Transient + Clean
                    if has_cache:
                        # Restore from cache to make transient buffer valid
                        should_restore = True
                    else:
                        # Clean but lost buffer -> Force Full Recompute
                        should_compute = True
                        compute_region = (
                            (slice(None),) * len(node.shape)
                            if node.shape
                            else (slice(None),)
                        )
                        executed_count += 1  # Count as execution

            # --- C. Perform Actions ---

            # 1. Restore (if needed)
            if should_restore and self.cache_manager:
                restored_count += 1
                cached_data = self.cache_manager.get(node)
                if cached_data is not None:
                    self.copy_to_buffer(node_name, cached_data)
                else:
                    raise ValueError(
                        "cached_data is None even though cache_manager.has() returned True"
                    )

            # 2. Compute (if needed)
            if should_compute:
                start_time = time.perf_counter()

                # If compute_region is None here, it implies Full Dirty (from propagation)
                # Ensure we handle the "Full Dirty" tuple correctly
                if compute_region is None:
                    compute_region = (
                        (slice(None),) * len(node.shape)
                        if node.shape
                        else (slice(None),)
                    )

                # Get input slices required for this output region
                input_slice_regions = DirtyPropagator.get_input_slices(
                    node, compute_region, inputs
                )

                # Prepare views
                kernel_inputs = []
                for i, p_name in enumerate(inst.input_node_names):
                    p_view_full = self._get_view(inst.input_offsets[i], p_name)

                    if (
                        i < len(input_slice_regions)
                        and input_slice_regions[i] is not None
                    ):
                        kernel_inputs.append(p_view_full[input_slice_regions[i]])
                    else:
                        kernel_inputs.append(p_view_full)

                # Output view
                out_view_full = self._get_view(inst.output_offsets[0], node_name)
                out_view_slice = out_view_full[compute_region]

                # Execute
                inst.kernel(kernel_inputs, [out_view_slice], inst.attrs)

                end_time = time.perf_counter()
                node.compute_cost = (end_time - start_time) * 1000  # ms
                executed_count += 1

                # 3. Update Cache (Post-Compute)
                # Only cache if policy allows and we have a valid FULL buffer now.
                # If we did Restore+Partial, buffer is valid.
                # If we did Full Compute, buffer is valid.
                # If we did Partial on Persistent, buffer is valid.
                # If we did Partial on Transient (without restore), buffer is GARBAGE (mixed).
                # But our logic above prevents Partial on Transient without Restore.

                if self.cache_manager and node.cache_policy != CachePolicy.NEVER:
                    # Policy check logic could be more complex (AUTO)
                    # For now, put full buffer
                    full_out_view = self._get_view(inst.output_offsets[0], node_name)
                    self.cache_manager.put(node, full_out_view)

        if DEBUG_EXECUTION:
            print(
                f"[Executor] Executed: {executed_count}, Restored: {restored_count}, Skipped: {skipped_count}"
            )

        # 3. Return outputs
        if len(self.output_views) == 1:
            return next(iter(self.output_views.values()))
        return self.output_views
