from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def triu_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Upper Triangle.
    inputs: [Data Tensor]
    attrs['k']: int (diagonal offset, default 0)
    """
    if len(inputs) != 1:
        raise ValueError("Triu requires exactly 1 data input")

    data = inputs[0]
    k_val = attrs.get("k", 0) if attrs else 0

    return TensorNode(
        OpType.TRIU,
        data.dtype,
        inputs,
        name=f"triu_{data.name}",
        attrs={"k": k_val},
        backend=data.backend,
    )
