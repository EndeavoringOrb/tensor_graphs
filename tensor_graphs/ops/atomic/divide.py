from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def divide_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Divide: A / B
    """
    if len(inputs) != 2:
        raise ValueError("Divide requires 2 inputs")

    a, b = inputs
    return TensorNode(OpType.DIVIDE, a.dtype, [a, b], name=f"divide_{a.name}_{b.name}")
