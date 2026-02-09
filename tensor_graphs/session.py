# tensor_graphs/session.py
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
    Persistent execution session for a computational graph.

    Key features:
    - Single CacheManager shared with Executor
    - Persistent weights loaded once
    - Incremental execution on subsequent runs
    - Minimal data copying via views and references
    """

    def __init__(
        self,
        root: TensorNode,
        db_path: str = "benchmarks.db",
        greedy: bool = True,
        max_cache_bytes: int = 5 * 1024**3,
    ):
        self.root = root
        self.db_path = db_path
        self.greedy = greedy
        self.max_cache_bytes = max_cache_bytes

        self.executor: Optional[Executor] = None
        self.cache_manager = CacheManager(max_cache_bytes)

        self.is_compiled = False

    def compile(self, sample_inputs: Dict[str, Any]):
        """Compile graph once with sample inputs."""
        if DEBUG_EXECUTION:
            print("[Session] Compiling graph...")

        planner = Planner(self.db_path, greedy=self.greedy)
        recipe = planner.plan(self.root, known_values=sample_inputs)

        compiler = Compiler()
        compiled_graph = compiler.compile(recipe, known_values=sample_inputs)

        # Create executor with shared cache manager
        self.executor = Executor(compiled_graph, cache_manager=self.cache_manager)

        # Load constants once (will be skipped on subsequent runs)
        all_nodes = topological_sort(recipe.root)
        constants = {
            n.name: n.attrs["value"]
            for n in all_nodes
            if n.op_type == OpType.CONSTANT and "value" in n.attrs
        }
        self.executor.load_weights(constants)

        self.is_compiled = True

        if DEBUG_EXECUTION:
            print("[Session] Compilation complete")

    def run(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute graph.

        On first run:
        - Compiles with sample inputs
        - Loads persistent weights

        On subsequent runs:
        - Skips weight reloading
        - Uses dirty propagation for incremental execution
        - Restores from cache when beneficial
        """
        if not self.is_compiled:
            self.compile(inputs)

        if self.executor is None:
            raise RuntimeError("Executor not initialized")

        return self.executor.run(inputs)

    def invalidate_cache(self):
        """Clear cache for next run (optional)."""
        self.cache_manager = CacheManager(self.max_cache_bytes)
        if self.executor:
            self.executor.cache_manager = self.cache_manager
