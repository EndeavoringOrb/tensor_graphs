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

        # --- Pass 1: Forward Propagation & Cache Check ---
        # Determine the state (Is Dirty? Is Cached?) for every node
        node_states = {}  # name -> {'is_dirty': bool, 'in_cache': bool}

        for inst in self.graph.instructions:
            node = self.graph.nodes_map[inst.node_name]

            # Skip Inputs/Constants as they are handled in _update_inputs or static
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
        # Determine which nodes MUST run.
        # A node is needed if:
        #   1. It is the graph output (root).
        #   2. It is an input to a node that IS needed AND MUST compute.
        #
        # A node MUST compute if:
        #   It is Needed AND (Dirty OR Not Cached).

        needed_nodes = set()

        # Assume the last instruction corresponds to the graph output
        if self.graph.instructions:
            root_name = self.graph.instructions[-1].node_name
            needed_nodes.add(root_name)

        # Iterate in reverse topological order
        for inst in reversed(self.graph.instructions):
            name = inst.node_name
            if name not in node_states:
                continue

            state = node_states[name]

            # If this node is not needed by any downstream consumer, skip logic
            if name in needed_nodes:
                # Do we need to actually execute this node?
                # Yes, if it's dirty OR if it's missing from cache.
                must_compute = state["is_dirty"] or not state["in_cache"]

                if must_compute:
                    # If we execute this node, we strictly need its inputs.
                    for p_name in inst.input_node_names:
                        needed_nodes.add(p_name)
                # Else: Node is needed, but it is Cached & Clean.
                # We can serve it from cache. We DO NOT mark its parents as needed
                # (unless they are needed by someone else).

        # --- Pass 3: Execution ---
        counters = {"no_op": 0, "skip": 0, "full": 0, "part": 0, "pruned": 0}
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

                should_compute = False
                compute_region = None

                if not is_needed:
                    # Pruned: Node is clean but missing, and no downstream node needs it to run.
                    counters["pruned"] += 1
                    # We skip execution entirely.
                elif state["is_dirty"] and state["in_cache"]:
                    # Partial Update
                    counters["part"] += 1
                    partial_ratio = 1
                    # Basic ratio calc for stats
                    if (
                        node.dirty_region
                        and node.shape
                        and len(node.shape) == len(node.dirty_region)
                    ):
                        for i, s in enumerate(node.dirty_region):
                            start = s.start if s.start is not None else 0
                            stop = s.stop if s.stop is not None else node.shape[i]
                            dim = node.shape[i] if node.shape[i] > 0 else 1
                            partial_ratio *= (stop - start) / dim
                    partial_ratio_sum += partial_ratio

                    should_compute = True
                    compute_region = node.dirty_region
                elif not state["in_cache"]:
                    # Full Recompute (Dirty or Clean but Missing & Needed)
                    counters["full"] += 1
                    should_compute = True
                    compute_region = None
                    # If clean but missing, ensure we don't mark it dirty for next run
                    if not state["is_dirty"]:
                        node.dirty_region = None
                else:
                    # Cached & Clean & Needed -> Skip
                    counters["skip"] += 1
                    self.mem.lock(node)

                # Update Stats
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
                            or node.op_type in (OpType.RESHAPE,)
                        ):
                            kernel_inputs.append(full_view)
                        else:
                            kernel_inputs.append(full_view[input_slice_regions[i]])

                    out_view = self.mem.get_view(node)
                    out_slice = out_view[compute_region] if node.op_type not in (OpType.RESHAPE,) else out_view

                    inst.kernel(kernel_inputs, [out_slice], inst.attrs)
                    node.compute_cost = (time.perf_counter() - start_time) * 1000

                # Release Transient Memory
                # We decrement ref counts even if we pruned the node, because
                # "skipping" conceptually means we are done with the inputs at this stage.
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
