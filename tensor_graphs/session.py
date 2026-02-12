from typing import Dict, Any, Optional, Union
from .ir.node import TensorNode
from .compiler.planner import Planner
from .compiler.compiler import Compiler
from .backend.executor import Executor
from .backend.memory import MemoryManager
from .ir.graph import topological_sort
from .ops.atomic_types import OpType
from .config import DEBUG_EXECUTION
from .ir.dtypes import Backend
from .weights import SafetensorsSource, WeightSource
import os
import numpy as np


class GraphSession:
    def __init__(
        self,
        root: TensorNode,
        db_path: str = "benchmarks.db",
        greedy: bool = True,
        max_memory_bytes: int = 5 * 1024**3,
    ):
        self.root = root
        self.db_path = db_path
        self.greedy = greedy
        self.mem_manager = MemoryManager(max_memory_bytes)
        self.executor: Optional[Executor] = None
        self.is_compiled = False

    def compile(self, sample_inputs: Dict[str, Any]):
        if DEBUG_EXECUTION:
            print("[Session] Planning graph...")

        planner = Planner(self.db_path, greedy=self.greedy)
        recipe = planner.plan(self.root, known_values=sample_inputs)

        if DEBUG_EXECUTION:
            print("[Session] Compiling graph...")

        compiler = Compiler()
        compiled_graph = compiler.compile(recipe, known_values=sample_inputs)

        self.executor = Executor(compiled_graph, memory_manager=self.mem_manager)
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
            assigned_backend = self.executor.graph.nodes_map[node.name].backend
            if assigned_backend is None:
                assigned_backend = backend_hint

            node.backend = assigned_backend

            self.mem_manager.allocate_persistent(node, data)

        if DEBUG_EXECUTION:
            print("[Session] Weights loaded.")

    def run(self, inputs: Dict[str, Any]) -> Any:
        if not self.is_compiled:
            self.compile(inputs)

        return self.executor.run(inputs)
