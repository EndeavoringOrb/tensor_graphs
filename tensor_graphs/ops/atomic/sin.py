from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def sin_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Sine: sin(x)
    inputs[0]: Input tensor
    """
    if len(inputs) != 1:
        raise ValueError("Sin requires exactly 1 input")

    x = inputs[0]

    return TensorNode(
        OpType.SIN,
        x.shape,
        x.dtype,
        [x],
        f"sin_{x.name}",
        backend=x.backend,
    )
