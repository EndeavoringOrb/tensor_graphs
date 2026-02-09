# tensor_graphs/backend/executor.py
import numpy as np
import torch
import time
from typing import Dict, Any, Optional, Set
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType
from ..ops.atomic_types import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from ..ir.node import CachePolicy, TensorNode
from ..config import *
from .cache import CacheManager


class Executor:
    """
    Unified Executor that runs a CompiledGraph with incremental execution support.

    Key optimizations:
    - Persistent weights loaded once and never diffed again
    - All buffers managed in unified address space
    - CacheManager stores references, not copies (unless policy requires)
    - Dirty propagation only for transient/changed tensors
    """

    def __init__(
        self,
        compiled_graph: CompiledGraph,
        cache_manager: Optional[CacheManager] = None,
    ):
        self.graph = compiled_graph
        self.cache_manager = cache_manager or CacheManager(max_bytes=5 * 1024**3)

        # Unified buffer management (device -> buffer)
        self.buffers: Dict[str, Any] = {}
        self.last_inputs: Dict[str, Any] = {}

        # Track which persistent nodes have been loaded
        self.loaded_persistent: Set[str] = set()

        # Cache views to avoid repeated allocations
        self.views_cache: Dict[str, Any] = {}

        # 1. Allocate persistent buffers
        self._allocate_buffers()

        # 2. Pre-calculate and cache input/output views
        self.input_views = {
            name: self._get_or_create_view(offset, name)
            for name, offset in self.graph.input_offsets.items()
        }
        self.output_views = {
            name: self._get_or_create_view(offset, name)
            for name, offset in self.graph.output_offsets.items()
        }

    def _allocate_buffers(self):
        """Allocate unified buffers per device."""
        device_sizes: Dict[str, int] = {}
        for alloc in self.graph.buffer_allocations.values():
            end = alloc.offset + alloc.size_bytes
            device_sizes[alloc.device] = max(device_sizes.get(alloc.device, 0), end)

        for device, size in device_sizes.items():
            if "cuda" in device or "gpu" in device:
                self.buffers[device] = torch.zeros(
                    size, dtype=torch.uint8, device=device
                )
            else:
                self.buffers[device] = np.zeros(size, dtype=np.uint8)

    def _map_dtype_np(self, dtype: DType) -> Any:
        mapping = {
            DType.FP32: np.float32,
            DType.INT32: np.int32,
            DType.FP16: np.float16,
            DType.BOOL: np.bool_,
        }
        return mapping.get(dtype, np.float32)

    def _map_dtype_torch(self, dtype: DType) -> Any:
        mapping = {
            DType.FP32: torch.float32,
            DType.INT32: torch.int32,
            DType.FP16: torch.float16,
            DType.BOOL: torch.bool,
        }
        return mapping.get(dtype, torch.float32)

    def _get_or_create_view(
        self, offset: int, name: str, device: Optional[str] = None
    ) -> Any:
        """Get cached view or create and cache new one."""
        cache_key = f"{name}:{offset}"
        if cache_key in self.views_cache:
            return self.views_cache[cache_key]

        view = self._create_view(offset, name, device)
        self.views_cache[cache_key] = view
        return view

    def _create_view(self, offset: int, name: str, device: Optional[str] = None) -> Any:
        """Create a view into the buffer without copying data."""
        meta = self.graph.node_metadata[name]
        alloc = self.graph.buffer_allocations[name]
        dev = device or alloc.device
        buf = self.buffers[dev]
        size_bytes = alloc.size_bytes

        if isinstance(buf, np.ndarray):
            np_dtype = self._map_dtype_np(meta.dtype)
            # Create a zero-copy view
            return np.ndarray(
                meta.shape, dtype=np_dtype, buffer=buf.data, offset=offset
            )
        elif isinstance(buf, torch.Tensor):
            torch_dtype = self._map_dtype_torch(meta.dtype)
            # Slice and view without copying
            slice_t = buf[offset : offset + size_bytes]
            return slice_t.view(torch_dtype).reshape(meta.shape)

        return None

    def _write_to_buffer(
        self, name: str, data: Any, offset: Optional[int] = None
    ) -> None:
        """Write data to buffer with minimal copying."""
        if name not in self.graph.buffer_allocations:
            return

        alloc = self.graph.buffer_allocations[name]
        actual_offset = offset if offset is not None else alloc.offset
        buf = self.buffers[alloc.device]
        meta = self.graph.node_metadata[name]

        if isinstance(buf, np.ndarray):
            if isinstance(data, np.ndarray):
                data_to_write = data
            elif isinstance(data, torch.Tensor):
                data_to_write = data.cpu().numpy()
            else:
                data_to_write = np.array(data, dtype=self._map_dtype_np(meta.dtype))

            # Ensure correct dtype
            if data_to_write.dtype != self._map_dtype_np(meta.dtype):
                data_to_write = data_to_write.astype(self._map_dtype_np(meta.dtype))

            # Flatten and copy bytes (minimal operation)
            data_bytes = data_to_write.view(np.uint8).flatten()
            buf[actual_offset : actual_offset + len(data_bytes)] = data_bytes

        elif isinstance(buf, torch.Tensor):
            t_dtype = self._map_dtype_torch(meta.dtype)
            if isinstance(data, torch.Tensor):
                data_to_write = data
            else:
                data_to_write = torch.tensor(data, dtype=t_dtype, device=alloc.device)

            if data_to_write.dtype != t_dtype:
                data_to_write = data_to_write.to(dtype=t_dtype)
            if data_to_write.device != buf.device:
                data_to_write = data_to_write.to(device=buf.device)

            data_bytes = data_to_write.reshape(-1).view(torch.uint8).flatten()
            buf[actual_offset : actual_offset + len(data_bytes)] = data_bytes

    def load_weights(self, weights: Dict[str, Any]) -> None:
        """
        Load persistent weights once into buffers.
        Subsequent runs will skip reloading.
        """
        for name, data in weights.items():
            if name in self.loaded_persistent:
                continue  # Already loaded, skip

            self._write_to_buffer(name, data)
            self.loaded_persistent.add(name)

    def _update_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Update input buffers and track dirty regions.
        Skip persistent weights that were pre-loaded.
        """
        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node:
                continue

            # Skip persistent nodes that were already loaded
            if (
                node.storage_type == StorageType.PERSISTENT
                and name in self.loaded_persistent
            ):
                # Mark as clean (no changes to persistent weights)
                node.dirty_region = None
                continue

            # For transient inputs: compute dirty region
            old_data = self.last_inputs.get(name)
            node.dirty_region = DirtyPropagator.get_diff(old_data, data)

            # Update buffer
            self._write_to_buffer(name, data)

            # Store for next comparison (use reference for small inputs, copy for large)
            if hasattr(data, "shape") and np.prod(data.shape) > 1000000:
                # Large array: store reference (user responsible for not mutating)
                self.last_inputs[name] = data
            else:
                # Small input: clone for safety
                if hasattr(data, "clone"):
                    self.last_inputs[name] = data.clone()
                elif isinstance(data, np.ndarray):
                    self.last_inputs[name] = data.copy()
                else:
                    self.last_inputs[name] = data

    def _should_execute_node(
        self, node: TensorNode, is_dirty: bool
    ) -> tuple[bool, bool, Optional[tuple]]:
        """
        Determine execution strategy for a node.

        Returns:
            (should_compute, should_restore, compute_region)
        """
        is_persistent = node.storage_type == StorageType.PERSISTENT
        has_cache = self.cache_manager and self.cache_manager.has(node)

        if is_dirty:
            node.dirty_count += 1
            if is_persistent:
                # Persistent + Dirty -> Patch (compute only dirty region)
                return True, False, node.dirty_region
            else:
                # Transient + Dirty
                if has_cache:
                    # Restore full buffer, then patch
                    return True, True, node.dirty_region
                else:
                    # Recompute full
                    full_region = (
                        (slice(None),) * len(node.shape)
                        if node.shape
                        else (slice(None),)
                    )
                    return True, False, full_region
        else:  # Clean
            if is_persistent:
                # Persistent + Clean -> Skip
                return False, False, None
            else:
                # Transient + Clean
                if has_cache:
                    # Restore from cache
                    return False, True, None
                else:
                    # Force full recompute (buffer lost)
                    full_region = (
                        (slice(None),) * len(node.shape)
                        if node.shape
                        else (slice(None),)
                    )
                    return True, False, full_region

    def run(self, inputs: Dict[str, Any]) -> Any:
        """Execute graph with incremental computation and caching."""
        # 1. Update inputs and track changes
        self._update_inputs(inputs)

        executed_count = 0
        restored_count = 0
        skipped_count = 0

        # 2. Execute instructions in topological order
        for inst in self.graph.instructions:
            node_name = inst.node_name
            node = self.graph.nodes_map[node_name]

            # --- A. Propagate Dirty Region ---
            if node.op_type not in (OpType.INPUT, OpType.CONSTANT):
                node.dirty_region = DirtyPropagator.propagate(node, inputs)

            # --- B. Determine Strategy ---
            is_dirty = node.dirty_region is not None
            should_compute, should_restore, compute_region = self._should_execute_node(
                node, is_dirty
            )

            # --- C. Execute ---

            # 1. Restore if needed (before compute)
            if should_restore and self.cache_manager:
                restored_count += 1
                cached_data = self.cache_manager.get(node)
                if cached_data is not None:
                    self._write_to_buffer(node_name, cached_data)

            # 2. Compute if needed
            if should_compute:
                start_time = time.perf_counter()

                # Ensure compute_region is a tuple of slices
                if compute_region is None:
                    compute_region = (
                        (slice(None),) * len(node.shape)
                        if node.shape
                        else (slice(None),)
                    )

                # Get input slices required for output region
                input_slice_regions = DirtyPropagator.get_input_slices(
                    node, compute_region, inputs
                )

                # Prepare kernel inputs (views into buffer, no copy)
                kernel_inputs = []
                for i, p_name in enumerate(inst.input_node_names):
                    p_view = self._get_or_create_view(inst.input_offsets[i], p_name)

                    if (
                        i < len(input_slice_regions)
                        and input_slice_regions[i] is not None
                    ):
                        kernel_inputs.append(p_view[input_slice_regions[i]])
                    else:
                        kernel_inputs.append(p_view)

                # Output view (slice if partial compute)
                out_view = self._get_or_create_view(inst.output_offsets[0], node_name)
                out_view_slice = out_view[compute_region]

                # Execute kernel
                inst.kernel(kernel_inputs, [out_view_slice], inst.attrs)

                end_time = time.perf_counter()
                node.compute_cost = (end_time - start_time) * 1000  # ms
                executed_count += 1

                # 3. Cache result if policy allows
                if self.cache_manager and node.cache_policy != CachePolicy.NEVER:
                    full_out_view = self._get_or_create_view(
                        inst.output_offsets[0], node_name
                    )
                    # CacheManager will handle cloning if needed
                    self.cache_manager.put(node, full_out_view)
            else:
                skipped_count += 1

        if DEBUG_EXECUTION:
            print(
                f"[Executor] Executed: {executed_count}, Restored: {restored_count}, "
                f"Skipped: {skipped_count}"
            )

        # 3. Return outputs
        if len(self.output_views) == 1:
            return next(iter(self.output_views.values()))
        return self.output_views
