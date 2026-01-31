import numpy as np
import torch
from typing import Dict, Any, List, Optional
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType
from ..ir.node import TensorNode
from ..compiler.planner import Planner
from ..compiler.compiler import Compiler


class Executor:
    """
    Unified Executor that runs a CompiledGraph using a pre-allocated memory buffer.
    All kernels must accept an 'outputs' argument (list of views).
    """

    def __init__(self, compiled_graph: CompiledGraph):
        self.graph = compiled_graph
        self.buffers: Dict[str, Any] = {}

        # Calculate size per device
        device_sizes: Dict[str, int] = {}
        for alloc in self.graph.buffer_allocations.values():
            end = alloc.offset + alloc.size_bytes
            device_sizes[alloc.device] = max(device_sizes.get(alloc.device, 0), end)

        for device, size in device_sizes.items():
            if "cuda" in device or "gpu" in device:
                # Align to 256 bytes or more
                self.buffers[device] = torch.empty(
                    size, dtype=torch.uint8, device=device
                )
            else:
                # CPU
                self.buffers[device] = np.zeros(size, dtype=np.uint8)

        # Pre-calculate views for instructions
        self.prepared_instructions = []
        for inst in self.graph.instructions:
            input_views = []
            for i, offset in enumerate(inst.input_offsets):
                input_name = inst.input_node_names[i]
                input_views.append(self._get_view(offset, input_name))

            output_views = []
            # Note: TensorNode metadata usually keyed by node_name.
            # For multi-output nodes, we might need refined metadata keys.
            # Assuming single-output nodes for now:
            for offset in inst.output_offsets:
                output_views.append(self._get_view(offset, inst.node_name))

            self.prepared_instructions.append(
                (inst.kernel, input_views, output_views, inst.attrs)
            )

        # Pre-calculate input/output views
        self.input_views = {
            name: self._get_view(offset, name)
            for name, offset in self.graph.input_offsets.items()
        }
        self.output_views = {
            name: self._get_view(offset, name)
            for name, offset in self.graph.output_offsets.items()
        }

    def _map_dtype_np(self, dtype: DType):
        if dtype == DType.FP32:
            return np.float32
        if dtype == DType.INT32:
            return np.int32
        if dtype == DType.FP16:
            return np.float16
        if dtype == DType.BOOL:
            return np.bool_
        return np.float32

    def _map_dtype_torch(self, dtype: DType):
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

        raise RuntimeError(f"Unknown buffer type: {type(buf)}")

    def copy_to_buffer(self, name: str, data: Any):
        if name not in self.graph.buffer_allocations:
            return

        alloc = self.graph.buffer_allocations[name]
        offset = alloc.offset
        buf = self.buffers[alloc.device]
        meta = self.graph.node_metadata[name]

        if isinstance(buf, np.ndarray):
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            np_dtype = self._map_dtype_np(meta.dtype)

            # Flatten and view as uint8 to copy
            data_casted = data.astype(np_dtype)
            data_u8 = data_casted.view(np.uint8).flatten()

            end = offset + len(data_u8)
            buf[offset:end] = data_u8

        elif isinstance(buf, torch.Tensor):
            t_dtype = self._map_dtype_torch(meta.dtype)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=t_dtype, device=alloc.device)
            else:
                data = data.to(dtype=t_dtype, device=alloc.device)

            data_u8 = data.view(torch.uint8).flatten()
            end = offset + len(data_u8)
            buf[offset:end] = data_u8

    def load_weights(self, weights: Dict[str, Any]):
        for name, data in weights.items():
            if name in self.graph.buffer_allocations:
                alloc = self.graph.buffer_allocations[name]
                if alloc.storage_type in (StorageType.PERSISTENT, StorageType.STATE):
                    self.copy_to_buffer(name, data)

    def run(self, inputs: Dict[str, Any]) -> Any:
        # 1. Copy inputs
        for name, data in inputs.items():
            self.copy_to_buffer(name, data)

        # 2. Execution Loop
        for kernel, input_views, output_views, attrs in self.prepared_instructions:
            # Unified Kernel API: (inputs, outputs, attrs)
            kernel(input_views, output_views, attrs)

        # 3. Return Output
        if len(self.output_views) == 1:
            return next(iter(self.output_views.values()))

        return self.output_views


def evaluate_graph(
    root: TensorNode, inputs: Dict[str, Any], db_path: str = "benchmarks.db"
) -> Any:
    """
    Helper function to maintain compatibility with tests.
    Compiles and runs the graph on the fly.
    """
    planner = Planner(db_path)
    recipe = planner.plan(root)

    compiler = Compiler()
    # Pass inputs as known_values to enable shape inference
    compiled_graph = compiler.compile(recipe, known_values=inputs)

    executor = Executor(compiled_graph)

    # Handle constants that might be in inputs or attributes
    return executor.run(inputs)
