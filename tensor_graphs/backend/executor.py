import numpy as np
from typing import Dict, Any

from ..ir.node import TensorNode
from ..ops.atomic_types import OpType
from ..ir.dtypes import Backend
from ..backend.registry import KernelRegistry
from ..compiler.planner import ExecutionRecipe, Planner


class Executor:
    def __init__(self, recipe: ExecutionRecipe):
        self.recipe = recipe
        self.cache: Dict[TensorNode, Any] = {}

    def run(self, inputs: Dict[str, Any]) -> Any:
        self.cache = {}
        return self._eval(self.recipe.root, inputs)

    def _eval(self, node: TensorNode, inputs: Dict[str, Any]) -> Any:
        if node in self.cache:
            return self.cache[node]

        if node.op_type == OpType.INPUT:
            if node.name not in inputs:
                raise ValueError(f"Missing input: {node.name}")
            val = inputs[node.name]
            self.cache[node] = val
            return val

        elif node.op_type == OpType.CONSTANT:
            val = node.attrs.get("value")
            if val is not None and not isinstance(val, np.ndarray):
                val = np.array(val)
            self.cache[node] = val
            return val

        # Evaluate Parents
        parent_vals = [self._eval(p, inputs) for p in node.parents]
        input_sigs = [p.signature for p in node.parents]

        # Backend comes from recipe assignments OR node itself (since Planner rewrites nodes with backend)
        backend = self.recipe.assignments.get(node, node.backend)

        kernel = KernelRegistry.select_best_kernel(
            node.op_type, input_sigs, backend, target_dtype=node.dtype
        )

        if not kernel:
            raise RuntimeError(f"Kernel not found for {node.op_type} on {backend}")

        val = kernel(parent_vals, node.attrs)
        self.cache[node] = val
        return val


def evaluate_graph(
    root: TensorNode, inputs: Dict[str, Any], db_path: str = "benchmarks.db"
) -> Any:
    planner = Planner(db_path)
    recipe = planner.plan(root)
    executor = Executor(recipe)
    return executor.run(inputs)
