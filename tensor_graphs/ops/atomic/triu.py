from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def triu_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Upper Triangle: Extract upper triangle of matrix
    inputs[0]: Data tensor (matrix)
    attrs["k"]: Diagonal offset (optional, default: 0)
    """
    if len(inputs) != 1:
        raise ValueError("Triu requires exactly 1 input: data tensor")

    data = inputs[0]

    k = attrs.get("k", 0) if attrs else 0

    return TensorNode(
        OpType.TRIU,
        data.shape,
        data.dtype,
        [data],
        f"triu_{data.name}",
        attrs={"k": k},
        backend=data.backend,
    )
