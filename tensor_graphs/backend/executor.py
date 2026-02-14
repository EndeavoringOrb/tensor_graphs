import numpy as np
import time
from typing import Dict, Any
import math
from ..compiler.compiled_graph import CompiledGraph
from ..ir.buffer import StorageType
from ..ir.dtypes import DType
from ..ops import OpType
from ..compiler.dirty_propagation import DirtyPropagator
from .memory import MemoryManager
from ..config import DEBUG_EXECUTION, DEBUG_DETAILED
from tqdm import tqdm


class Executor:
    def __init__(
        self,
        compiled_graph: CompiledGraph,
        memory_manager: MemoryManager,
    ):
        self.graph = compiled_graph
        self.mem = memory_manager
        self.last_inputs: Dict[str, Any] = {}

    def _update_inputs(self, inputs: Dict[str, Any]) -> None:
        for name, data in inputs.items():
            node = self.graph.nodes_map.get(name)
            if not node:
                continue

            # If persistent (parameters), skip (loaded via load_weights)
            if node.storage_type == StorageType.PERSISTENT:
                continue

            # Transient Inputs (Dynamic inputs like ids)
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

        counters: dict[str, int | float | str] = {
            "no_op": 0,
            "skip": 0,
            "full": 0,
            "part": 0,
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

                # Skip Constants/Inputs as they have no instruction to execute
                if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                    counters["no_op"] += 1
                    continue

                node.dirty_region = DirtyPropagator.propagate(node, inputs)
                is_dirty = node.dirty_region is not None

                dev_hint = node.backend.value if node.backend else "cpu"
                in_cache = self.mem.has(node.name, dev_hint)

                should_compute = False
                compute_region = None

                if is_dirty and in_cache:
                    counters["part"] += 1
                    partial_ratio = 1
                    if len(node.shape) != len(node.dirty_region):
                        raise ValueError(
                            "[Executor.run] len(node.shape) != len(node.dirty_region)"
                        )
                    for i in range(len(node.shape)):
                        partial_ratio *= (
                            (node.dirty_region[i].stop or node.shape[i]) - (node.dirty_region[i].start or 0)
                        ) / node.shape[i]
                    partial_ratio_sum += partial_ratio
                    should_compute = True
                    compute_region = node.dirty_region
                elif not in_cache:
                    counters["full"] += 1
                    should_compute = True
                    compute_region = None
                    node.dirty_region = None
                else:
                    counters["skip"] += 1
                    self.mem.lock(node)

                counters["p_sum"] = (
                    (partial_ratio_sum / counters["part"])
                    if counters["part"] > 0
                    else "N/A"
                )
                counters["p_cache"] = (
                    counters["skip"] + (counters["part"] - partial_ratio_sum)
                ) / (counters["skip"] + counters["part"] + counters["full"])
                pbar.set_postfix(counters)

                if should_compute:
                    start_time = time.perf_counter()

                    size_bytes = math.prod(node.shape or ()) * 4
                    if node.dtype == DType.FP16:
                        size_bytes //= 2
                    elif node.dtype == DType.BOOL:
                        size_bytes //= 4

                    self.mem.prepare_allocation(node, size_bytes)

                    kernel_inputs = []
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
                            i >= len(input_slice_regions)
                            or input_slice_regions[i] is None
                        ):
                            kernel_inputs.append(full_view)
                        else:
                            kernel_inputs.append(full_view[input_slice_regions[i]])

                    out_view = self.mem.get_view(node)
                    out_slice = out_view[compute_region]

                    inst.kernel(kernel_inputs, [out_slice], inst.attrs)
                    node.compute_cost = (time.perf_counter() - start_time) * 1000

                for p_name in inst.input_node_names:
                    current_refs[p_name] -= 1
                    if current_refs[p_name] == 0:
                        p_node = self.graph.nodes_map[p_name]
                        if p_node.storage_type == StorageType.TRANSIENT:
                            self.mem.unlock(p_node)

        root_name = self.graph.instructions[-1].node_name
        return self.mem.get_view(self.graph.nodes_map[root_name])
