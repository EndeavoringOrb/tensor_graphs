import numpy as np
from typing import Dict
from ..ops.atomic import OpType
from ..ir.node import TensorNode
from ..backend.registry import KernelRegistry

# Ensure kernels are registered before we try to look them up
import tensor_graphs.backend.kernels


def evaluate_graph(root: TensorNode, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Evaluates the graph using the Kernel Registry with score-based dispatch.
    """
    cache: Dict[TensorNode, np.ndarray] = {}

    def _eval(node: TensorNode):
        if node in cache:
            return cache[node]

        if node.op_type == OpType.INPUT:
            if node.name not in inputs:
                raise ValueError(f"Missing input data for node: {node.name}")
            val = inputs[node.name]
            val = np.asarray(val)
        else:
            # Recursively evaluate parents
            parent_vals = [_eval(p) for p in node.parents]

            # Dispatch: Look up kernel based on PARENT signatures
            input_sigs = [p.signature for p in node.parents]

            kernel = KernelRegistry.select_best_kernel(node.op_type, input_sigs)

            if not kernel:
                raise NotImplementedError(
                    f"No valid kernel found for op '{node.op_type}' "
                    f"with input signatures {input_sigs}"
                )

            val = kernel(parent_vals)

        cache[node] = val
        return val

    return _eval(root)
