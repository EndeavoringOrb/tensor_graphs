from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def negate_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Negate: -x
    inputs[0]: Input tensor
    """
    if len(inputs) != 1:
        raise ValueError("Negate requires exactly 1 input")

    x = inputs[0]

    return TensorNode(
        OpType.NEGATE,
        x.shape,
        x.dtype,
        [x],
        f"negate_{x.name}",
        backend=x.backend,
    )
