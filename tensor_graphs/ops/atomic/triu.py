from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def triu_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Upper Triangle: Extract upper triangle of matrix
    inputs[0]: Data tensor (matrix)
    inputs[1]: Diagonal offset k (optional)
    attrs["k"]: Diagonal offset (optional, default: 0) if inputs[1] not present
    """
    if len(inputs) == 1:
        data = inputs[0]
        k_val = attrs.get("k", 0) if attrs else 0
        node_attrs = {"k": k_val}
        parents = [data]
    elif len(inputs) == 2:
        data = inputs[0]
        k_tensor = inputs[1]
        parents = [data, k_tensor]
        node_attrs = {}  # k is in the input
    else:
        raise ValueError("Triu requires 1 or 2 inputs: data tensor and optional k")

    return TensorNode(
        OpType.TRIU,
        data.shape,
        data.dtype,
        parents,
        f"triu_{data.name}",
        attrs=node_attrs,
        backend=data.backend,
    )
