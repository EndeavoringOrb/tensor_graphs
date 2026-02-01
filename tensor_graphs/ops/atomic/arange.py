from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def arange_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Arange: range(start, stop, step)
    """
    if len(inputs) != 3:
        raise ValueError("Arange requires 3 inputs: start, stop, step")

    start, stop, step = inputs
    return TensorNode(
        OpType.ARANGE,
        dtype=start.dtype,
        parents=[start, stop, step],
        name=f"arange_{start.name}_{stop.name}_{step.name}",
        attrs=attrs or {},
    )
