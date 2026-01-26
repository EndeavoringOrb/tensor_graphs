from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def sqrt_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Square Root: sqrt(x)
    inputs[0]: Input tensor
    """
    if len(inputs) != 1:
        raise ValueError("Sqrt requires exactly 1 input")

    x = inputs[0]

    return TensorNode(
        OpType.SQRT,
        x.shape,
        x.dtype,
        [x],
        f"sqrt_{x.name}",
        backend=x.backend,
    )
