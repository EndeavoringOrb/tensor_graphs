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

    # The output shape is dynamic based on the *values* in shape_tensor.
    # We initialize it to (None,) to indicate it needs inference.
    # ShapeInference will later resolve this using the actual values.

    return TensorNode(
        OpType.RESHAPE,
        (None,),
        data.dtype,
        [data, shape_tensor],
        f"reshape_{data.name}",
        backend=data.backend,
    )
