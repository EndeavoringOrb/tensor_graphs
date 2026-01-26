from typing import Dict, Any
from ..ir.node import TensorNode
from ..ir.dtypes import Backend
from ..ops.atomic_types import OpType
from ..backend.registry import KernelRegistry
from ..compiler.planner import ExecutionRecipe


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
                raise ValueError(f"Missing input data for node: {node.name}")
            val = inputs[node.name]
        elif node.op_type == OpType.CONSTANT:
            val = node.attrs.get("value")
        else:
            # 1. Evaluate Parents
            parent_vals = [self._eval(p, inputs) for p in node.parents]

            # 2. Get backend for this node
            backend = self.recipe.assignments.get(node, Backend.CPU_NUMPY)

            # 3. Select kernel for this backend
            input_sigs = [p.signature for p in node.parents]

            # Pass node.dtype as target_dtype to handle Cast specificity
            kernel = KernelRegistry.select_best_kernel(
                node.op_type, input_sigs, backend, target_dtype=node.dtype
            )

            if not kernel:
                raise RuntimeError(
                    f"No kernel found for {node.op_type} on backend {backend}"
                )

            val = kernel(parent_vals, node.attrs)

        self.cache[node] = val
        return val
