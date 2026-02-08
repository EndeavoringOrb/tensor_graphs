import numpy as np
import torch
from typing import Dict, Any, Optional
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType
from ..ops.atomic_types import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from ..config import *


class Executor:
    """
    Unified Executor that runs a CompiledGraph with incremental execution support.
    Uses DirtyRegion propagation to execute kernels only on changed data slices.
    """

    def __init__(
        self,
        compiled_graph: CompiledGraph,
        cache_manager: Optional[Any] = None,  # Kept for signature compatibility
    ):
        self.graph = compiled_graph
        self.buffers: Dict[str, Any] = {}

        # Store last inputs to calculate diffs between runs
        self.last_inputs: Dict[str, Any] = {}

        # 1. Allocate persistent buffers for the entire graph memory space
        self._allocate_buffers()

        # 2. Store instructions for linear execution
        self.prepared_instructions = self.graph.instructions

        # 3. Create full-buffer views for inputs and outputs
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
                self.buffers[device] = torch.empty(
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
            return np.ndarray(
                meta.shape, dtype=np_dtype, buffer=buf.data, offset=offset
            )
        elif isinstance(buf, torch.Tensor):
            torch_dtype = self._map_dtype_torch(meta.dtype)
            slice_t = buf[offset : offset + size_bytes]
            return slice_t.view(torch_dtype).view(meta.shape)

        return None

    def copy_to_buffer(self, name: str, data: Any):
        """Writes data into the persistent buffer at the node's allocated offset."""
        if name not in self.graph.buffer_allocations:
            return

        alloc = self.graph.buffer_allocations[name]
        offset = alloc.offset
        buf = self.buffers[alloc.device]
        meta = self.graph.node_metadata[name]

        if isinstance(buf, np.ndarray):
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=self._map_dtype_np(meta.dtype))

            data_u8 = (
                data.astype(self._map_dtype_np(meta.dtype)).view(np.uint8).flatten()
            )
            buf[offset : offset + len(data_u8)] = data_u8

        elif isinstance(buf, torch.Tensor):
            t_dtype = self._map_dtype_torch(meta.dtype)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=t_dtype, device=alloc.device)
            else:
                data = data.to(dtype=t_dtype, device=alloc.device)

            data_u8 = data.reshape(-1).view(torch.uint8).flatten()
            buf[offset : offset + len(data_u8)] = data_u8

    def load_weights(self, weights: Dict[str, Any]):
        """Initializes persistent weights and marks them as CLEAN."""
        for name, data in weights.items():
            self.copy_to_buffer(name, data)
            if name in self.graph.nodes_map:
                self.graph.nodes_map[name].dirty_region = None

    def _update_inputs(self, inputs: Dict[str, Any]):
        """Diffs new inputs against previous ones and updates buffer."""
        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node:
                continue

            # Calculate what region of the input actually changed
            old_data = self.last_inputs.get(name)
            node.dirty_region = DirtyPropagator.get_diff(old_data, data)

            # Overwrite buffer with new content
            self.copy_to_buffer(name, data)

            # Update history for next diff
            if hasattr(data, "clone"):
                self.last_inputs[name] = data.clone()
            elif isinstance(data, np.ndarray):
                self.last_inputs[name] = data.copy()
            else:
                self.last_inputs[name] = data

    def run(self, inputs: Dict[str, Any]) -> Any:
        self._update_inputs(inputs)
        executed_count = 0

        for inst in self.prepared_instructions:
            node_name = inst.node_name
            node = self.graph.nodes_map[node_name]

            if node.op_type != OpType.INPUT and node.op_type != OpType.CONSTANT:
                node.dirty_region = DirtyPropagator.propagate(node)

            # Only skip if CLEAN AND NOT RECYCLABLE.
            # Transient nodes must re-run to restore their buffers if they were clobbered.
            if node.dirty_region is None and node.storage_type != StorageType.TRANSIENT:
                continue

            # If it's Clean but forced to run (Transient), compute the FULL region
            compute_region = node.dirty_region
            if compute_region is None:
                compute_region = (
                    tuple(slice(None) for _ in range(len(node.shape)))
                    if node.shape
                    else (slice(None),)
                )

            input_regions = DirtyPropagator.get_input_slices(node, compute_region)

            # Generate sub-views for inputs
            input_views = []
            for i, p_name in enumerate(inst.input_node_names):
                p_view_full = self._get_view(inst.input_offsets[i], p_name)

                # Apply the slice calculated by the propagator
                if i < len(input_regions) and input_regions[i] is not None:
                    input_views.append(p_view_full[input_regions[i]])
                else:
                    # If no specific slice mapping exists, use full parent view
                    input_views.append(p_view_full)

            # D. Slice the output buffer to match the dirty region
            out_view_full = self._get_view(inst.output_offsets[0], node_name)
            out_view_slice = out_view_full[node.dirty_region]

            if DEBUG_EXECUTION and DEBUG_DETAILED:
                print(f"[Executor] EXEC: {node_name} | Region: {node.dirty_region}")

            # E. Execute the kernel on sub-slices
            inst.kernel(input_views, [out_view_slice], inst.attrs)
            executed_count += 1

        if DEBUG_EXECUTION:
            print(
                f"[Executor] Run complete. Executed: {executed_count}, Skipped: {len(self.prepared_instructions) - executed_count}"
            )

        # 3. Return output views
        if len(self.output_views) == 1:
            return next(iter(self.output_views.values()))
        return self.output_views
