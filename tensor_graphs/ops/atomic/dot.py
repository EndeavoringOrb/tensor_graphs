from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def dot_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Dot Product (Matrix Multiplication): A @ B
    inputs[0]: Matrix A
    inputs[1]: Matrix B
    """
    if len(inputs) != 2:
        raise ValueError("Dot requires exactly 2 inputs")

    a, b = inputs

    # Calculate output shape: (A.shape[0], B.shape[1])
    # Note: We assume inputs are 2D matrices for the reference graph.
    # If inputs are higher rank, broadcasting rules apply, but the graph structure
    # typically represents the result of matmul.
    out_shape = (a.shape[0], b.shape[1])

    return TensorNode(OpType.DOT, out_shape, a.dtype, [a, b], f"dot_{a.name}_{b.name}")
