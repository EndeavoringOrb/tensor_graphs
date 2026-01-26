from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def permute_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Permute/Transpose: permute dimensions
    inputs[0]: Data tensor
    inputs[1]: Permutation axes (1D tensor of integers)
    """
    if len(inputs) != 2:
        raise ValueError("Permute requires exactly 2 inputs: data and permutation axes")

    data, perm = inputs

    return TensorNode(
        OpType.PERMUTE,
        data.shape,
        data.dtype,
        [data, perm],
        f"permute_{data.name}",
        backend=data.backend,
    )
