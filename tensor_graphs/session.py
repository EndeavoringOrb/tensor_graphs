from typing import Dict, Any, Optional
from .ir.node import TensorNode
from .compiler.planner import Planner
from .compiler.compiler import Compiler
from .backend.executor import Executor
from .backend.memory import MemoryManager
from .ir.graph import topological_sort
from .ops.atomic_types import OpType
from .config import DEBUG_EXECUTION


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

        # Unified Memory Manager
        self.mem_manager = MemoryManager(max_memory_bytes)
        self.executor: Optional[Executor] = None
        self.is_compiled = False

    def compile(self, sample_inputs: Dict[str, Any]):
        if DEBUG_EXECUTION:
            print("[Session] Compiling graph...")

        planner = Planner(self.db_path, greedy=self.greedy)
        recipe = planner.plan(self.root, known_values=sample_inputs)

        compiler = Compiler()
        compiled_graph = compiler.compile(recipe, known_values=sample_inputs)

        self.executor = Executor(compiled_graph, memory_manager=self.mem_manager)

        # Load constants
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
        if not self.is_compiled:
            self.compile(inputs)

        return self.executor.run(inputs)
