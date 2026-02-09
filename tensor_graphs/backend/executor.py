import numpy as np
import time
from typing import Dict, Any
import math
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType
from ..compiler.dirty_propagation import DirtyPropagator
from .memory import MemoryManager


class Executor:
    def __init__(
        self,
        compiled_graph: CompiledGraph,
        memory_manager: MemoryManager,
    ):
        self.graph = compiled_graph
        self.mem = memory_manager
        self.last_inputs: Dict[str, Any] = {}
        self.loaded_persistent = set()

    def load_weights(self, weights: Dict[str, Any]) -> None:
        """Load persistent weights into memory immediately."""
        for name, data in weights.items():
            if name in self.loaded_persistent:
                continue

            node = self.graph.nodes_map.get(name)
            if node and node.storage_type == StorageType.PERSISTENT:
                self.mem.allocate_persistent(node, data)
                self.loaded_persistent.add(name)

    def _update_inputs(self, inputs: Dict[str, Any]) -> None:
        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node:
                continue

            # If persistent (parameters)
            if node.storage_type == StorageType.PERSISTENT:
                if name not in self.loaded_persistent:
                    self.mem.allocate_persistent(node, data)
                    self.loaded_persistent.add(name)
                continue

            # Transient Inputs (Dynamic inputs like ids)
            # Allocation/Write
            size = (
                data.nbytes if hasattr(data, "nbytes") else (data.numel() * 4)
            )  # approx fallback
            self.mem.prepare_allocation(node, size)

            # Dirty Check
            old_data = self.last_inputs.get(name)
            node.dirty_region = DirtyPropagator.get_diff(old_data, data)

            self.mem.write(node, data)

            # Store copy/ref for next run diffing
            if hasattr(data, "clone"):
                self.last_inputs[name] = data.clone()
            elif isinstance(data, np.ndarray):
                self.last_inputs[name] = data.copy()
            else:
                self.last_inputs[name] = data

    def run(self, inputs: Dict[str, Any]) -> Any:
        self.mem.step()  # Advance LRU time
        self._update_inputs(inputs)

        # Dynamic ref counts for this run
        current_refs = self.graph.ref_counts.copy()

        executed_count = 0
        restored_count = 0  # "Restored" now means "Found in cache, didn't recompute"

        for inst in self.graph.instructions:
            node = self.graph.nodes_map[inst.node_name]

            # 1. Propagate Dirty
            node.dirty_region = DirtyPropagator.propagate(node, inputs)
            is_dirty = node.dirty_region is not None

            # 2. Check Cache Status
            # Device hint for checking existence
            dev_hint = node.backend.value if node.backend else "cpu"
            in_cache = self.mem.has(node.name, dev_hint)

            should_compute = False
            compute_region = None

            if is_dirty:
                should_compute = True
                compute_region = node.dirty_region
            elif not in_cache:
                should_compute = True
                compute_region = None  # Full recompute
                node.dirty_region = None  # Treat as clean but missing
            else:
                # Clean and In Cache -> Skip
                restored_count += 1
                # Mark as used (LRU)
                self.mem.lock(node)

            if should_compute:
                start_time = time.perf_counter()

                # 3. Allocation (Output)
                size_bytes = (
                    math.prod(node.shape or ()) * 4
                )  # Simplification for FP32/INT32
                if node.dtype == DType.FP16:
                    size_bytes //= 2
                elif node.dtype == DType.BOOL:
                    size_bytes //= 4

                # Ensure space exists (evict if needed)
                self.mem.prepare_allocation(node, size_bytes)

                # 4. Prepare Inputs (Views)
                kernel_inputs = []

                # Ensure compute_region is tuple of slices
                if compute_region is None:
                    compute_region = tuple(slice(None) for _ in (node.shape or ()))

                input_slice_regions = DirtyPropagator.get_input_slices(
                    node, compute_region, inputs
                )

                for i, p_name in enumerate(inst.input_node_names):
                    p_node = self.graph.nodes_map[p_name]
                    self.mem.lock(p_node)

                    full_view = self.mem.get_view(p_node)

                    if (
                        i < len(input_slice_regions)
                        and input_slice_regions[i] is not None
                    ):
                        kernel_inputs.append(full_view[input_slice_regions[i]])
                    else:
                        kernel_inputs.append(full_view)

                # 5. Execute
                out_view = self.mem.get_view(node)
                out_slice = out_view[compute_region]

                inst.kernel(kernel_inputs, [out_slice], inst.attrs)

                executed_count += 1
                node.compute_cost = (time.perf_counter() - start_time) * 1000

            # 6. Decrement Ref Counts & Unlock Inputs
            for p_name in inst.input_node_names:
                current_refs[p_name] -= 1
                if current_refs[p_name] == 0:
                    p_node = self.graph.nodes_map[p_name]
                    if p_node.storage_type == StorageType.TRANSIENT:
                        self.mem.unlock(p_node)

        # Output View
        root_name = self.graph.instructions[-1].node_name
        return self.mem.get_view(self.graph.nodes_map[root_name])
