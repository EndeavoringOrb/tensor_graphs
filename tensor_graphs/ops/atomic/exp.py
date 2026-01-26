from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def exp_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Exponential: exp(x)
    inputs[0]: Input tensor
    """
    if len(inputs) != 1:
        raise ValueError("Exp requires exactly 1 input")

    x = inputs[0]
    return TensorNode(
        OpType.EXP,
        x.shape,
        x.dtype,
        [x],
        f"exp_{x.name}",
    )
