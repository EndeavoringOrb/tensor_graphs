from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def power_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Power: A ^ B
    inputs[0]: Base tensor
    inputs[1]: Exponent tensor
    """
    if len(inputs) != 2:
        raise ValueError("Power requires exactly 2 inputs")

    a, b = inputs

    return TensorNode(
        OpType.POWER,
        a.dtype,
        [a, b],
        name=f"power_{a.name}_{b.name}",
        backend=a.backend,
    )
