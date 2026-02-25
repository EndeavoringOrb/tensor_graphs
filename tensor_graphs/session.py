from typing import Dict, Any, Optional, Union, List
from .ir.node import TensorNode
from .compiler.planner import Planner
from .compiler.compiled_graph import CompiledGraph
from .backend.executor import Executor
from .backend.memory import MemoryManager
from .ir.graph import topological_sort, GraphEncoder
from .ops.atomic_types import OpType
from .config import DEBUG_EXECUTION
from .ir.dtypes import Backend, TensorSignature
from .weights import SafetensorsSource, WeightSource
from .compiler.dirty_propagation import DirtyPropagator
from .compiler.propagation import _from_slices, _to_slices, GraphPropagator
from .backend.registry import KernelRegistry
import os
import json
import itertools
from tqdm import tqdm
import numpy as np


class GraphSession:
    def __init__(
        self,
        root: TensorNode,
        db_path: str = "benchmarks.db",
        max_memory_bytes: int = 5 * 1024**3,
        weights_path: str = "",
        cache_path: Optional[str] = None,
    ):
        self.root = root
        self.db_path = db_path
        self.mem_manager = MemoryManager(max_memory_bytes)
        self.executor: Optional[Executor] = None
        self.is_compiled = False
        self.weights_path = weights_path
        self.cache_path = cache_path
        self.dirty_cache: Dict[str, Any] = {}
        self.cached_compiled_graph: Optional[CompiledGraph] = None

        if self.cache_path and os.path.exists(self.cache_path):
            self._load_cache()

    def _load_cache(self):
        if DEBUG_EXECUTION:
            print(f"[Session] Loading dirty region cache from {self.cache_path}...")
        with open(self.cache_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)

                # Check if this is a compiled graph entry
                if entry.get("type") == "compiled_graph":
                    if DEBUG_EXECUTION:
                        print("[Session] Loading compiled graph from cache...")
                    self.cached_compiled_graph = CompiledGraph.from_dict(entry["data"])
                    continue

                # Reconstruct key tuple structure for dictionary lookup
                key_list = entry["key"]
                key_parts = []
                for k, regions in key_list:
                    region_parts = []
                    for box in regions:
                        box_tuple = tuple(tuple(item) for item in box)
                        region_parts.append(box_tuple)
                    key_parts.append((k, tuple(region_parts)))
                key = tuple(key_parts)

                # Pre-convert regions to tuple format at load time
                precomputed_regions = {}
                for node_name, reg_data in entry["regions"].items():
                    if reg_data:
                        converted = []
                        for box in reg_data:
                            slices = tuple(slice(s, e) for s, e in box)
                            converted.append(slices)
                        precomputed_regions[node_name] = converted
                    else:
                        precomputed_regions[node_name] = reg_data

                # Pre-convert input_slices
                precomputed_input_slices = {}
                for node_name, slices_data in entry["input_slices"].items():
                    if slices_data:
                        converted = []
                        for box_reqs in slices_data:
                            box_converted = []
                            for req in box_reqs:
                                if req is not None:
                                    req_slices = tuple(slice(s, e) for s, e in req[0])
                                    box_converted.append(req_slices)
                                else:
                                    box_converted.append(req)
                            converted.append(box_converted)
                        precomputed_input_slices[node_name] = converted
                    else:
                        precomputed_input_slices[node_name] = slices_data

                # Load concrete shapes (new)
                precomputed_shapes = entry.get("concrete_shapes", {})

                self.dirty_cache[key] = {
                    "regions": precomputed_regions,
                    "input_slices": precomputed_input_slices,
                    "concrete_shapes": precomputed_shapes,
                    # Kernels are not serialized, will be resolved in _ensure_cache_coverage or on demand
                    "kernels": {},
                }

    def _save_cache_entry(self, key, regions, input_slices, concrete_shapes):
        if not self.cache_path:
            return

        serialized_key = []
        for name, region_tuple in key:
            serialized_key.append([name, [list(box) for box in region_tuple]])

        entry = {
            "key": serialized_key,
            "regions": regions,
            "input_slices": input_slices,
            "concrete_shapes": concrete_shapes,
        }

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(entry, cls=GraphEncoder) + "\n")

    def _save_compiled_graph(self, compiled_graph: CompiledGraph):
        """Save the compiled graph to the cache file."""
        if not self.cache_path:
            return

        if DEBUG_EXECUTION:
            print("[Session] Saving compiled graph to cache...")

        entry = {
            "type": "compiled_graph",
            "data": compiled_graph.to_dict(),
        }

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(entry, cls=GraphEncoder) + "\n")

    def _next_power_of_2(self, x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def _generate_slices_for_dim(self, dim_len):
        """Generates non-overlapping aligned power-of-2 tiles."""
        slices = []
        size = 1
        max_size = self._next_power_of_2(dim_len)

        while size <= max_size:
            for i in range(0, dim_len, size):
                end = min(i + size, dim_len)
                slices.append((i, end))
            size *= 2

        return list(set(slices))

    def _resolve_kernels_for_cache_entry(self, entry, compiled_graph):
        """Resolve kernel objects for a cache entry based on stored concrete shapes."""
        if "kernels" not in entry:
            entry["kernels"] = {}

        # If kernels already resolved (in-memory) or shapes missing, return
        if entry["kernels"] or not entry.get("concrete_shapes"):
            return

        for node_name, shape_options in entry["concrete_shapes"].items():
            node = compiled_graph.nodes_map.get(node_name)
            instr = compiled_graph.get_instruction(node_name)
            if not node or not instr:
                continue

            # Determine inplace constraint from the full compute instruction
            allow_inplace = instr.inplace_input_index is not None

            kernels_for_node = []
            for concrete_shape_tuple in shape_options:
                # concrete_shape_tuple is a list of shapes (one per input), serialized as lists
                # Convert back to tuples
                input_shapes = [tuple(s) if s else None for s in concrete_shape_tuple]

                # Build signatures
                input_sigs = []
                for idx, shape in enumerate(input_shapes):
                    p_node = compiled_graph.nodes_map.get(instr.input_node_names[idx])
                    if p_node:
                        sig = TensorSignature(p_node.dtype, shape, node.backend)
                        input_sigs.append(sig)

                # Select best kernel
                kernel_result = KernelRegistry.select_best_kernel(
                    node.op_type,
                    input_sigs,
                    node.backend,
                    node.dtype,
                    allow_inplace=allow_inplace,
                )

                if kernel_result:
                    kernels_for_node.append(kernel_result[0])  # (kernel, is_inplace)
                else:
                    # Fallback to full kernel if specific partial kernel not found
                    kernels_for_node.append(instr.kernel)

            entry["kernels"][node_name] = kernels_for_node

    def _ensure_cache_coverage(
        self,
        input_nodes: List[TensorNode],
        sample_inputs: Dict[str, Any],
        compiled_graph,
    ):
        """
        Generates dirty region buckets for all permutations of input slice configurations.
        Pre-calculates concrete shapes and selects partial kernels.
        """
        if not self.cache_path:
            return

        if DEBUG_EXECUTION:
            print("[Session] verifying dirty region cache coverage...")

        # 1. Generate options per input
        input_options = []

        for node in input_nodes:
            if node.name not in sample_inputs:
                continue

            shape = sample_inputs[node.name].shape
            dim_options_list = []

            for d in range(len(shape)):
                dim_len = shape[d]
                dim_slices = self._generate_slices_for_dim(dim_len)
                dim_options_list.append(dim_slices)

            input_regions = list(itertools.product(*dim_options_list))
            input_regions_wrapped = [(r,) for r in input_regions]
            input_options.append((node.name, input_regions_wrapped))

        names = [x[0] for x in input_options]
        option_lists = [x[1] for x in input_options]

        all_permutations = list(itertools.product(*option_lists))
        missing_count = 0
        topo_nodes = list(compiled_graph.nodes_map.values())
        GraphPropagator.infer_shapes(topo_nodes, sample_inputs)

        for perm in tqdm(
            all_permutations,
            disable=not DEBUG_EXECUTION,
            desc="Ensuring cache coverage",
        ):
            key_map = {name: reg for name, reg in zip(names, perm)}
            sorted_items = sorted(key_map.items())
            key = tuple(sorted_items)

            # Check cache existence
            if key in self.dirty_cache:
                # Ensure kernels are resolved for loaded entries
                self._resolve_kernels_for_cache_entry(
                    self.dirty_cache[key], compiled_graph
                )
                continue

            missing_count += 1

            # Set inputs
            for name, region_tuple in key_map.items():
                node = next(n for n in input_nodes if n.name == name)
                node.dirty_region = _to_slices(region_tuple)

            # Reset others
            for node in topo_nodes:
                if node.op_type != OpType.INPUT:
                    node.dirty_region = None

            # Propagate Forward
            node_regions_map = {}
            node_input_slices_map = {}
            node_concrete_shapes_map = {}  # New: Stores concrete shapes for partial kernels
            node_kernels_map = {}  # New: Stores selected partial kernels

            for node in topo_nodes:
                if node.op_type != OpType.INPUT:
                    node.dirty_region = DirtyPropagator.propagate(node)

                # Serialize region
                if node.dirty_region:
                    ser_region = []
                    for box in node.dirty_region:
                        ser_box = []
                        for s in box:
                            ser_box.append((s.start, s.stop))
                        ser_region.append(ser_box)
                    node_regions_map[node.name] = ser_region

                    if node.op_type in (OpType.INPUT, OpType.CONSTANT):
                        continue

                    # Compute Backward Input Slices
                    input_slices_per_box = []
                    concrete_shapes_per_box = []  # New
                    partial_kernels_per_box = []  # New

                    instr = compiled_graph.get_instruction(node.name)
                    allow_inplace = (
                        instr.inplace_input_index is not None if instr else False
                    )

                    for box in node.dirty_region:
                        reqs = DirtyPropagator.get_input_slices(node, [box])

                        # Serialize requirements
                        ser_reqs = []
                        box_concrete_shapes = []
                        box_kernels = []

                        for req_idx, req in enumerate(reqs):
                            if req:
                                ser_req = []
                                for r_box in req:
                                    ser_req.append(list(r_box))
                                ser_reqs.append(
                                    _from_slices(ser_req, node.parents[req_idx].shape)
                                )

                                # --- NEW: Calculate Concrete Shape ---
                                # req is List[Tuple[slice...]]
                                p_node = node.parents[req_idx]
                                p_shape = p_node.shape or ()
                                # Use first box in req (usually one box in this logic)
                                sl = req[0]
                                # Calculate shape from slice
                                if sl:
                                    c_shape = []
                                    for s, dim in zip(sl, p_shape):
                                        start, stop, _ = s.indices(dim)
                                        c_shape.append(stop - start)
                                    box_concrete_shapes.append(tuple(c_shape))
                                else:
                                    box_concrete_shapes.append(p_shape)
                            else:
                                ser_reqs.append(None)
                                # Full shape if no slice
                                box_concrete_shapes.append(
                                    node.parents[req_idx].shape or ()
                                )

                        input_slices_per_box.append(ser_reqs)
                        concrete_shapes_per_box.append(box_concrete_shapes)

                        # --- NEW: Select Partial Kernel ---
                        if instr:
                            input_sigs = []
                            for idx, shape in enumerate(box_concrete_shapes):
                                p_node = node.parents[idx]
                                sig = TensorSignature(p_node.dtype, shape, node.backend)
                                input_sigs.append(sig)

                            kernel_result = KernelRegistry.select_best_kernel(
                                node.op_type,
                                input_sigs,
                                node.backend,
                                node.dtype,
                                allow_inplace=allow_inplace,
                            )

                            if kernel_result:
                                # Verify constraint
                                _, kernel_is_inplace = kernel_result
                                if allow_inplace != kernel_is_inplace:
                                    # Constraint violation: Fallback or raise error
                                    # For safety, we fallback to standard kernel instruction
                                    # Ideally, this shouldn't happen if registry is consistent
                                    box_kernels.append(instr.kernel)
                                else:
                                    box_kernels.append(kernel_result[0])
                            else:
                                # Fallback to full kernel
                                box_kernels.append(instr.kernel)

                        partial_kernels_per_box.append(box_kernels)

                    node_input_slices_map[node.name] = input_slices_per_box
                    node_concrete_shapes_map[node.name] = concrete_shapes_per_box
                    node_kernels_map[node.name] = partial_kernels_per_box
                else:
                    node_regions_map[node.name] = None
                    node_input_slices_map[node.name] = None
                    node_concrete_shapes_map[node.name] = None
                    node_kernels_map[node.name] = None

            # 4. Save
            entry = {
                "regions": node_regions_map,
                "input_slices": node_input_slices_map,
                "concrete_shapes": node_concrete_shapes_map,  # Save shapes
                "kernels": node_kernels_map,  # In-memory kernels
            }

            self.dirty_cache[key] = entry

            # Save to disk (kernels are not serialized, will be resolved on load)
            # We need to convert tuples in concrete_shapes to lists for JSON
            ser_shapes = {}
            for n, shapes in node_concrete_shapes_map.items():
                if shapes:
                    ser_shapes[n] = [list(s) for s in shapes]

            self._save_cache_entry(
                key, node_regions_map, node_input_slices_map, ser_shapes
            )

        # self._load_cache()  # Reload to ensure structure consistency (optional optimization: just update memory)

        # Clean up graph state
        for node in topo_nodes:
            node.dirty_region = None

        if DEBUG_EXECUTION and missing_count > 0:
            print(
                f"[Session] Computed and cached {missing_count} missing dirty configurations."
            )

    def compile(self, sample_inputs: Dict[str, Any]):
        if DEBUG_EXECUTION:
            print("[Session] Planning & Compiling graph...")

        if self.cached_compiled_graph is not None:
            if DEBUG_EXECUTION:
                print("[Session] Using cached compiled graph")
            compiled_graph = self.cached_compiled_graph

            input_nodes = [
                n
                for n in compiled_graph.nodes_map.values()
                if n.op_type == OpType.INPUT and n.storage_type.name == "TRANSIENT"
            ]
            self._ensure_cache_coverage(input_nodes, sample_inputs, compiled_graph)
        else:
            planner = Planner(self.db_path)
            compiled_graph = planner.plan(self.root, known_values=sample_inputs)

            input_nodes = [
                n
                for n in compiled_graph.nodes_map.values()
                if n.op_type == OpType.INPUT and n.storage_type.name == "TRANSIENT"
            ]

            self._ensure_cache_coverage(input_nodes, sample_inputs, compiled_graph)
            self._save_compiled_graph(compiled_graph)

        self.executor = Executor(
            compiled_graph,
            memory_manager=self.mem_manager,
            dirty_cache=self.dirty_cache,
        )
        self.is_compiled = True

        self._load_internal_constants()

        if DEBUG_EXECUTION:
            print("[Session] Compilation complete")

    def _load_internal_constants(self):
        """Loads constants (scalars/vectors) stored in graph attributes into memory."""
        if not self.is_compiled or not self.executor:
            return

        for node_name, node in self.executor.graph.nodes_map.items():
            if node.op_type == OpType.CONSTANT:
                val = node.attrs.get("value")
                if val is None:
                    continue

                if not isinstance(val, np.ndarray):
                    val = np.array(val)

                device = node.backend.value if node.backend else "cpu"
                if "numpy" in device:
                    device = "cpu"

                if not self.mem_manager.has(node_name, device):
                    self.mem_manager.allocate_persistent(node, val)

    def load_weights(
        self,
        source: Union[str, WeightSource, None] = None,
        backend_hint: Backend = Backend.CPU_NUMPY,
    ):
        """
        Loads weights from a file path or WeightSource object.
        Must be called after compile().
        """
        if not self.is_compiled:
            raise RuntimeError("Session must be compiled before loading weights.")

        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Weight file not found: {source}")

            if source.endswith(".safetensors"):
                source = SafetensorsSource(source)
            else:
                raise ValueError(f"Unsupported weight file format: {source}")

        if DEBUG_EXECUTION:
            print(f"[Session] Loading weights from {type(source).__name__}...")

        all_nodes = topological_sort(self.root)

        for node in all_nodes:
            if node.op_type != OpType.INPUT or node.storage_type.name != "PERSISTENT":
                continue

            device = node.backend.value if node.backend else "cpu"
            if "numpy" in device:
                device = "cpu"
            if self.mem_manager.has(node.name, device):
                continue

            if source is None:
                raise RuntimeError(
                    f"Node {node.name} is a persistent weight but no weight source provided."
                )

            if node.name not in source.keys():
                raise KeyError(
                    f"Weight '{node.name}' expected by graph but not found in source."
                )

            data = source.get_tensor(node.name)

            if self.executor and node.name in self.executor.graph.nodes_map:
                assigned_backend = self.executor.graph.nodes_map[node.name].backend
            else:
                assigned_backend = backend_hint

            if self.executor and node.name in self.executor.graph.nodes_map:
                exec_node = self.executor.graph.nodes_map[node.name]
                exec_node.backend = assigned_backend
                self.mem_manager.allocate_persistent(exec_node, data)
            else:
                node.backend = assigned_backend
                self.mem_manager.allocate_persistent(node, data)

        if DEBUG_EXECUTION:
            print("[Session] Weights loaded.")

    def run(self, inputs: Dict[str, Any]) -> Any:
        if not self.is_compiled:
            self.compile(inputs)
            self.load_weights(self.weights_path)

        return self.executor.run(inputs)
