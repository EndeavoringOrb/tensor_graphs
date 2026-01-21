import numpy as np
from typing import Dict
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.ir.node import TensorNode

def run_atomic_op(op_type: str, input_vals: list):
    """Executes a single operation using NumPy."""
    if op_type == OpType.ADD:
        return input_vals[0] + input_vals[1]
    
    elif op_type == OpType.MUL:
        return input_vals[0] * input_vals[1]
    
    elif op_type == OpType.DOT:
        return np.matmul(input_vals[0], input_vals[1])
    
    elif op_type == OpType.SILU:
        return input_vals[0] * (1 / (1 + np.exp(-input_vals[0])))
    
    elif op_type == OpType.SUM:
        return np.sum(input_vals[0])

    # --- Fused Ops Support (For verifying optimizers) ---
    elif op_type == OpType.FUSED_MUL_ADD:
        # FusedMulAdd(A, B, C) -> (A * B) + C
        return (input_vals[0] * input_vals[1]) + input_vals[2]

    raise NotImplementedError(f"Op {op_type} not implemented in ref backend")

def evaluate_graph(root: TensorNode, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    recursively evaluates the graph.
    'inputs' is a dict mapping Node.name -> numpy array.
    """
    # Memoization cache to handle DAG structures efficiently
    cache: Dict[TensorNode, np.ndarray] = {}

    def _eval(node: TensorNode):
        if node in cache:
            return cache[node]

        if node.op_type == OpType.INPUT:
            if node.name not in inputs:
                raise ValueError(f"Missing input data for node: {node.name}")
            val = inputs[node.name]
        else:
            # Recursively evaluate parents
            parent_vals = [_eval(p) for p in node.parents]
            val = run_atomic_op(node.op_type, parent_vals)
        
        cache[node] = val
        return val

    return _eval(root)