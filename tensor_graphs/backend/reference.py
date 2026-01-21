import numpy as np
from typing import Dict, List
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.backend.registry import KernelRegistry
# Ensure implementations are registered
import tensor_graphs.backend.ops.implementations

def evaluate_graph(root: TensorNode, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Evaluates the graph using the Kernel Registry.
    Dispatches based on the static signatures of the nodes.
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
            # In a real strict backend, we would verify val.dtype/shape match node.signature here
        else:
            # Recursively evaluate parents
            parent_vals = [_eval(p) for p in node.parents]
            
            # Dispatch: Look up kernel based on PARENT signatures
            # The kernel consumes parents and produces this node
            input_sigs = [p.signature for p in node.parents]
            
            kernel = KernelRegistry.get_kernel(node.op_type, input_sigs)
            if not kernel:
                raise NotImplementedError(
                    f"No kernel implementation found for op '{node.op_type}' "
                    f"with input signatures {input_sigs}"
                )
            
            val = kernel(parent_vals)

        cache[node] = val
        return val

    return _eval(root)