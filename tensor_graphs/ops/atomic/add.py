from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def add_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Add: A + B
    """
    if len(inputs) != 2:
        raise ValueError("Add requires 2 inputs")

    a, b = inputs
    return TensorNode(OpType.ADD, a.dtype, [a, b], name=f"add_{a.name}_{b.name}")
