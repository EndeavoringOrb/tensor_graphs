from typing import Dict, Any, Optional, Union, List
from .ir.node import TensorNode
from .compiler.planner import Planner
from .backend.executor import Executor
from .backend.memory import MemoryManager
from .ir.graph import topological_sort
from .ops.atomic_types import OpType
from .config import DEBUG_EXECUTION
from .ir.dtypes import Backend
from .weights import SafetensorsSource, WeightSource
from .compiler.dirty_propagation import DirtyPropagator
from .compiler.propagation import _from_slices, _to_slices, GraphPropagator
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
                # Reconstruct key tuple structure for dictionary lookup
                # Key format in JSON: [ [input_name, [ [start, stop], ... ]], ... ]
                key_list = entry["key"]
                # Convert lists back to tuples for hashability
                key = tuple(
                    (k, tuple(tuple(tuple(item) for item in box) for box in regions))
                    for k, regions in key_list
                )
                self.dirty_cache[key] = {
                    "regions": entry["regions"],
                    "input_slices": entry["input_slices"],
                }

    def _save_cache_entry(self, key, regions, input_slices):
        if not self.cache_path:
            return

        # Serialize structure
        # Key: tuple((name, tuple(tuple(box)))) -> list([name, list([list(box)])])
        serialized_key = []
        for name, region_tuple in key:
            serialized_key.append([name, [list(box) for box in region_tuple]])

        entry = {
            "key": serialized_key,
            "regions": regions,
            "input_slices": input_slices,
        }

        with open(self.cache_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _next_power_of_2(self, x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def _generate_slices_for_dim(self, dim_len):
        """Generates non-overlapping aligned power-of-2 tiles."""
        slices = []
        size = 1
        # Go up to the smallest power of 2 that covers the dimension
        max_size = self._next_power_of_2(dim_len)

        while size <= max_size:
            # Step by 'size' to ensure no overlaps within the same zoom level
            for i in range(0, dim_len, size):
                # i + size might exceed dim_len, so we clamp it
                end = min(i + size, dim_len)
                slices.append((i, end))
            size *= 2

        return list(set(slices))  # unique-ify in case dim_len is a power of 2

    def _ensure_cache_coverage(
        self, input_nodes: List[TensorNode], sample_inputs: Dict[str, Any]
    ):
        """
        Generates dirty region buckets for all permutations of input slice configurations.
        """
        if not self.cache_path:
            return

        if DEBUG_EXECUTION:
            print("[Session] verifying dirty region cache coverage...")

        # 1. Generate options per input
        input_options = []  # List of (input_name, list_of_region_tuples)

        for node in input_nodes:
            if node.name not in sample_inputs:
                continue

            shape = sample_inputs[node.name].shape
            dim_options_list = []

            for d in range(len(shape)):
                dim_len = shape[d]
                dim_slices = self._generate_slices_for_dim(dim_len)
                dim_options_list.append(dim_slices)

            # Cartesian product of dimensions for this single input -> List[RegionTuple]
            # RegionTuple is ((start, stop), (start, stop), ...)
            input_regions = list(itertools.product(*dim_options_list))

            # Filter? We treat each as a separate dirty configuration [box]
            # We assume 1 dirty box per input for the permutation basis
            # Wrap as a single-box list for consistency with Propagator: [(box)]
            input_regions_wrapped = [(r,) for r in input_regions]

            # Add "Clean" state (None)?
            # Prompt implies inputs are dirty. We stick to dirty permutations.

            input_options.append((node.name, input_regions_wrapped))

        # 2. Cartesian product across all inputs
        # names: [A, B], options: [ [rA1, rA2], [rB1, rB2] ]
        names = [x[0] for x in input_options]
        option_lists = [x[1] for x in input_options]

        # We need to temporarily set dirty regions on the graph nodes
        # To avoid messing up state, we can just set them, propagate, read, and clear.
        # Since we are before planner, it's safe.

        all_permutations = list(itertools.product(*option_lists))

        missing_count = 0
        topo_nodes = topological_sort(self.root)
        GraphPropagator.infer_shapes(topo_nodes, sample_inputs)

        for perm in tqdm(
            all_permutations,
            disable=not DEBUG_EXECUTION,
            desc="Ensuring cache coverage",
        ):
            # Construct Key
            # perm is ( region_tuple_A, region_tuple_B, ... )
            # Key must be sorted by name to be canonical
            key_map = {name: reg for name, reg in zip(names, perm)}
            # Sort by name
            sorted_items = sorted(key_map.items())
            key = tuple(sorted_items)

            if key in self.dirty_cache:
                continue

            missing_count += 1

            # 3. Propagate
            # Set inputs
            for name, region_tuple in key_map.items():
                # Find node
                node = next(n for n in input_nodes if n.name == name)
                # region_tuple is (box, ) where box is ((s,e), ...)
                # Convert to slices for propagator
                node.dirty_region = _to_slices(region_tuple)

            # Reset others
            for node in topo_nodes:
                if node.op_type != OpType.INPUT:
                    node.dirty_region = None

            # Propagate Forward
            node_regions_map = {}
            node_input_slices_map = {}

            for node in topo_nodes:
                if node.op_type != OpType.INPUT:
                    node.dirty_region = DirtyPropagator.propagate(node)

                # Store serialized region
                # node.dirty_region is List[Tuple[slice...]]
                # Serialize to List[List[(start, stop)]]
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

                    # Compute Backward Input Slices for each box
                    # This is needed to avoid calling get_input_slices at runtime
                    input_slices_per_box = []
                    for box in node.dirty_region:
                        reqs = DirtyPropagator.get_input_slices(node, [box])

                        # Serialize requirements (List of NumericRegions, one per parent)
                        ser_reqs = []
                        for req_idx, req in enumerate(reqs):
                            if req:
                                ser_req = []
                                for r_box in req:
                                    ser_req.append(list(r_box))  # tuple -> list
                                ser_reqs.append(
                                    _from_slices(ser_req, node.parents[req_idx].shape)
                                )
                            else:
                                ser_reqs.append(None)
                        input_slices_per_box.append(ser_reqs)

                    node_input_slices_map[node.name] = input_slices_per_box
                else:
                    node_regions_map[node.name] = None
                    node_input_slices_map[node.name] = None

            # 4. Save
            self.dirty_cache[key] = {
                "regions": node_regions_map,
                "input_slices": node_input_slices_map,
            }
            self._save_cache_entry(key, node_regions_map, node_input_slices_map)

        # Clean up graph state
        for node in topo_nodes:
            node.dirty_region = None

        if DEBUG_EXECUTION and missing_count > 0:
            print(
                f"[Session] Computed and cached {missing_count} missing dirty configurations."
            )

    def compile(self, sample_inputs: Dict[str, Any]):
        # Run pre-planner dirty region caching
        input_nodes = [
            n
            for n in topological_sort(self.root)
            if n.op_type == OpType.INPUT and n.storage_type.name == "TRANSIENT"
        ]
        # Ensure sample_inputs covers these inputs to determine shapes
        self._ensure_cache_coverage(input_nodes, sample_inputs)

        if DEBUG_EXECUTION:
            print("[Session] Planning & Compiling graph...")

        planner = Planner(self.db_path)
        compiled_graph = planner.plan(self.root, known_values=sample_inputs)

        self.executor = Executor(
            compiled_graph,
            memory_manager=self.mem_manager,
            dirty_cache=self.dirty_cache,
        )
        self.is_compiled = True

        # Automatically load internal constants defined in the graph
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

                # Ensure it's a numpy array
                if not isinstance(val, np.ndarray):
                    val = np.array(val)

                # Allocate if not already present
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
            # Only load persistent INPUT nodes (Weights/Parameters)
            # Constants are handled by _load_internal_constants
            if node.op_type != OpType.INPUT or node.storage_type.name != "PERSISTENT":
                continue

            # Check if already loaded (e.g. by constants loader or previous call)
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

            # Determine placement
            # We must use the node instance from the *compiled* graph (in executor)
            # because that's where the backend is finalized.
            if self.executor and node.name in self.executor.graph.nodes_map:
                assigned_backend = self.executor.graph.nodes_map[node.name].backend
            else:
                assigned_backend = backend_hint

            # Update the node backend if we found it in the compiled graph
            if self.executor and node.name in self.executor.graph.nodes_map:
                # The executor graph nodes are copies/modified, so we use them
                exec_node = self.executor.graph.nodes_map[node.name]
                exec_node.backend = assigned_backend
                self.mem_manager.allocate_persistent(exec_node, data)
            else:
                # Fallback if not compiled yet (shouldn't happen due to check above)
                node.backend = assigned_backend
                self.mem_manager.allocate_persistent(node, data)

        if DEBUG_EXECUTION:
            print("[Session] Weights loaded.")

    def run(self, inputs: Dict[str, Any]) -> Any:
        if not self.is_compiled:
            self.compile(inputs)
            self.load_weights(self.weights_path)

        return self.executor.run(inputs)
