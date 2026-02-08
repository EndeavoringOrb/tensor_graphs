from typing import Dict, Any, Optional
from .ir.node import TensorNode
from .compiler.planner import Planner
from .compiler.compiler import Compiler
from .backend.executor import Executor
from .backend.cache import CacheManager
from .ir.graph import topological_sort
from .ops.atomic_types import OpType
from .config import DEBUG_EXECUTION


class GraphSession:
    """
    Manages a persistent execution session for a computational graph.
    Maintains the Executor, CacheManager, and execution state across multiple runs.
    """

    def __init__(
        self,
        root: TensorNode,
        db_path: str = "benchmarks.db",
        greedy: bool = True,
        max_cache_bytes: int = 1024**3,  # 1GB default
    ):
        self.root = root
        self.db_path = db_path
        self.greedy = greedy
        self.max_cache_bytes = max_cache_bytes

        self.executor: Optional[Executor] = None
        self.cache_manager = CacheManager(max_cache_bytes)

        # State
        self.is_compiled = False

    def compile(self, sample_inputs: Dict[str, Any]):
        """
        Compiles the graph using sample inputs for shape inference.
        """
        if DEBUG_EXECUTION:
            print("[Session] compiling graph...")

        planner = Planner(self.db_path, greedy=self.greedy)
        recipe = planner.plan(self.root, known_values=sample_inputs)

        compiler = Compiler()
        compiled_graph = compiler.compile(recipe, known_values=sample_inputs)

        # Pass CacheManager to Executor
        self.executor = Executor(compiled_graph, cache_manager=self.cache_manager)

        # Load constants once
        all_nodes = topological_sort(recipe.root)
        constants = {
            n.name: n.attrs["value"]
            for n in all_nodes
            if n.op_type == OpType.CONSTANT and "value" in n.attrs
        }
        self.executor.load_weights(constants)

        self.is_compiled = True

    def run(self, inputs: Dict[str, Any]) -> Any:
        """
        Executes the graph with the provided inputs.
        Compiles on first run if not already compiled.
        """
        if not self.is_compiled:
            self.compile(inputs)

        if self.executor is None:
            raise RuntimeError("Session executor is not initialized.")

        return self.executor.run(inputs)

    def invalidate_cache(self):
        """Clears the session cache."""
        self.cache_manager = CacheManager(self.max_cache_bytes)
        if self.executor:
            self.executor.cache_manager = self.cache_manager
