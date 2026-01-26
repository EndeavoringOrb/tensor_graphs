from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def cos_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Cosine: cos(x)
    inputs[0]: Input tensor
    """
    if len(inputs) != 1:
        raise ValueError("Cos requires exactly 1 input")

    x = inputs[0]
    return TensorNode(
        OpType.COS,
        x.shape,
        x.dtype,
        [x],
        f"cos_{x.name}",
    )
