from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def permute_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Permute/Transpose.
    inputs: [Data Tensor]
    attrs['dims']: List[int] (Permutation order)
    """
    if len(inputs) != 1:
        raise ValueError("Permute requires exactly 1 data input")

    if attrs is None or "dims" not in attrs:
        raise ValueError("Permute requires 'dims' (permutation order) in attributes")

    data = inputs[0]
    dims = attrs["dims"]

    # Calculate output shape
    out_shape = tuple(data.shape[i] for i in dims) if data.shape else None

    return TensorNode(
        OpType.PERMUTE,
        data.dtype,
        inputs,
        name=f"permute_{data.name}",
        attrs=attrs,
        backend=data.backend,
    )
