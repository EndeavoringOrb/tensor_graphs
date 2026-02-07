import numpy as np
import torch
from typing import Dict, Any, Optional
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType
from ..ir.node import TensorNode
from ..compiler.planner import Planner
from ..compiler.compiler import Compiler
from ..ir.graph import topological_sort
from ..ops.atomic_types import OpType
from ..config import *
import time


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

        if DEBUG_EXECUTION:
            print(
                f"[DEBUG] Initializing Executor with {len(self.graph.instructions)} instructions."
            )

        for device, size in device_sizes.items():
            if DEBUG_EXECUTION:
                print(
                    f"[DEBUG] Pre-allocating {size / 1024 / 1024:.2f} MB on device: {device}"
                )

            if "cuda" in device or "gpu" in device:
                self.buffers[device] = torch.empty(
                    size, dtype=torch.uint8, device=device
                )
            else:
                self.buffers[device] = np.zeros(size, dtype=np.uint8)

        # Pre-calculate views for instructions
        self.prepared_instructions = []
        for inst in self.graph.instructions:
            input_views = []
            for i, offset in enumerate(inst.input_offsets):
                input_name = inst.input_node_names[i]
                input_views.append(self._get_view(offset, input_name))

            output_views = []
            for offset in inst.output_offsets:
                output_views.append(self._get_view(offset, inst.node_name))

            self.prepared_instructions.append(
                (inst.node_name, inst.kernel, input_views, output_views, inst.attrs)
            )

        self.input_views = {
            name: self._get_view(offset, name)
            for name, offset in self.graph.input_offsets.items()
        }
        self.output_views = {
            name: self._get_view(offset, name)
            for name, offset in self.graph.output_offsets.items()
        }

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

        raise RuntimeError(f"Unknown buffer type: {type(buf)}")

    def copy_to_buffer(self, name: str, data: Any):
        if name not in self.graph.buffer_allocations:
            return

        alloc = self.graph.buffer_allocations[name]
        offset = alloc.offset
        buf = self.buffers[alloc.device]
        meta = self.graph.node_metadata[name]

        # Add this safety check:
        data_size = int(np.prod(np.shape(data)))
        meta_size = int(np.prod(meta.shape))
        if data_size != meta_size:
            raise ValueError(
                f"Shape mismatch for node '{name}': "
                f"Buffer expects {meta_size} elements ({meta.shape}), "
                f"but provided data has {data_size} elements ({np.shape(data)})."
            )

        if isinstance(buf, np.ndarray):
            if not isinstance(data, np.ndarray):
                # Expand scalar to match node shape
                if np.isscalar(data):
                    data = np.full(
                        meta.shape, data, dtype=self._map_dtype_np(meta.dtype)
                    )
                else:
                    data = np.array(data)

            np_dtype = self._map_dtype_np(meta.dtype)

            # Flatten and view as uint8 to copy
            data_casted = data.astype(np_dtype).reshape(-1)
            data_u8 = data_casted.view(np.uint8).flatten()

            end = offset + len(data_u8)
            buf[offset:end] = data_u8

        elif isinstance(buf, torch.Tensor):
            t_dtype = self._map_dtype_torch(meta.dtype)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=t_dtype, device=alloc.device)
            else:
                data = data.to(dtype=t_dtype, device=alloc.device)

            data_u8 = data.reshape(-1).view(torch.uint8).flatten()
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
            if DEBUG_EXECUTION:
                print(f"[DEBUG] Loading input: '{name}' shape: {np.shape(data)}")
            self.copy_to_buffer(name, data)

        # 2. Execution Loop
        for (
            node_name,
            kernel,
            input_views,
            output_views,
            attrs,
        ) in self.prepared_instructions:
            if DEBUG_EXECUTION:
                print(f"[DEBUG] Executing: {node_name} ({kernel.__name__})")

            if DEBUG_EXECUTION and DEBUG_DETAILED:
                for i, v in enumerate(input_views):
                    view: Any = v
                    if isinstance(view, torch.Tensor):
                        val = (
                            f"{view.float().mean().item():.6f}"
                            if view.is_floating_point() and view.numel() > 0
                            else "N/A"
                        )
                    else:
                        val = (
                            f"{view.mean():.6f}"
                            if np.issubdtype(view.dtype, np.number) and view.size > 0
                            else "N/A"
                        )
                    print(f"  Input {i} shape: {view.shape}, mean: {val}")

            # Unified Kernel API: (inputs, outputs, attrs)
            start_k = time.perf_counter()
            kernel(input_views, output_views, attrs)
            end_k = time.perf_counter()

            if DEBUG_EXECUTION and DEBUG_DETAILED:
                duration = (end_k - start_k) * 1000
                print(f"  -> Finished in {duration:.4f} ms")
                for i, v in enumerate(output_views):
                    view: Any = v
                    if isinstance(view, torch.Tensor):
                        val = (
                            f"{view.float().mean().item():.6f}"
                            if view.is_floating_point() and view.numel() > 0
                            else "N/A"
                        )
                        nz = f"{(view != 0).sum().item()}"
                    else:
                        val = (
                            f"{view.mean():.6f}"
                            if np.issubdtype(view.dtype, np.number) and view.size > 0
                            else "N/A"
                        )
                        nz = f"{np.count_nonzero(view)}"
                    print(
                        f"  Output {i} shape: {view.shape}, mean: {val}, non-zeros: {nz}"
                    )

        # 3. Return Output
        if len(self.output_views) == 1:
            return next(iter(self.output_views.values()))

        return self.output_views


def evaluate_graph(
    root: TensorNode,
    inputs: Dict[str, Any],
    db_path: str = "benchmarks.db",
    greedy: bool = True,
) -> Any:
    """
    Compiles and runs the graph with enhanced stage logging.
    Args:
        root: The root node of the graph.
        inputs: Dictionary of input values.
        db_path: Path to benchmark database.
        greedy: If True, the Planner uses greedy optimization (fast compilation).
                If False, it performs exhaustive search (slow compilation, potentially faster execution).
    """
    total_start = time.perf_counter()

    if DEBUG_EXECUTION:
        print(
            f"\n{'=' * 60}\n[DEBUG] EVALUATING GRAPH: {root.name} (Op: {root.op_type})\n{'=' * 60}"
        )

    # 1. Planning Stage
    if DEBUG_EXECUTION:
        strategy_name = "Greedy" if greedy else "Exhaustive"
        print(f"[DEBUG] Stage 1: Planning ({strategy_name} optimization)...")

    planner = Planner(db_path, greedy=greedy)
    recipe = planner.plan(root, known_values=inputs)

    # 2. Compilation Stage
    if DEBUG_EXECUTION:
        print("[DEBUG] Stage 2: Compiling (Liveness analysis & memory planning)...")
    compiler = Compiler()
    compiled_graph = compiler.compile(recipe, known_values=inputs)

    if DEBUG_EXECUTION:
        print("[DEBUG] Compiled Graph Instructions:")
        for i, inst in enumerate(compiled_graph.instructions):
            print(
                f"  {i:03d} | {inst.node_name:<20} = {inst.kernel.__name__}({', '.join(inst.input_node_names)})"
            )
        print(
            f"[DEBUG] Total Static Memory Required: {compiled_graph.total_memory_bytes / 1024:.2f} KB"
        )

    # 3. Allocation Stage
    if DEBUG_EXECUTION:
        print("[DEBUG] Stage 3: Initializing Executor & Buffers...")
    executor = Executor(compiled_graph)

    # 4. Weight Loading Stage
    all_nodes = topological_sort(recipe.root)
    constants = {
        n.name: n.attrs["value"]
        for n in all_nodes
        if n.op_type == OpType.CONSTANT and "value" in n.attrs
    }

    if DEBUG_EXECUTION:
        print(
            f"[DEBUG] Stage 4: Loading {len(constants)} constants into persistent buffers..."
        )
    executor.load_weights(constants)

    # 5. Execution Stage
    if DEBUG_EXECUTION:
        print("[DEBUG] Stage 5: Running kernel execution loop...")
    result = executor.run(inputs)

    total_end = time.perf_counter()
    if DEBUG_EXECUTION:
        print(
            f"{'=' * 60}\n[DEBUG] GRAPH EVALUATION COMPLETE in {(total_end - total_start) * 1000:.2f} ms\n{'=' * 60}\n"
        )

    return result
