from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def reshape_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Reshape: Reshape tensor to new shape
    inputs[0]: Data tensor
    inputs[1]: Target shape (1D tensor of integers)
    """
    if len(inputs) != 2:
        raise ValueError("Reshape requires exactly 2 inputs: data and target shape")

    data, shape_tensor = inputs

    return TensorNode(
        OpType.RESHAPE,
        data.dtype,
        [data, shape_tensor],
        name=f"reshape_{data.name}",
        backend=data.backend,
    )
