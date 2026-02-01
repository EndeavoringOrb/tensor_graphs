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

    return TensorNode(OpType.DOT, a.dtype, [a, b], name=f"dot_{a.name}_{b.name}")
