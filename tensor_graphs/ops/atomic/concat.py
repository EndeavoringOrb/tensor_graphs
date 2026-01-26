from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def concat_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Concat: A + B along axis
    inputs[0]: Tensor A
    inputs[1]: Tensor B
    inputs[2]: Axis (1-element 1D INT32 tensor)
    """
    if len(inputs) != 3:
        raise ValueError("Concat requires 3 inputs: two tensors and an axis")

    a, b, axis_tensor = inputs

    return TensorNode(
        OpType.CONCAT,
        a.shape,  # Shape of output tensor (same as input A)
        a.dtype,  # Same dtype as input A
        [a, b, axis_tensor],
        f"concat_{a.name}_{b.name}_{axis_tensor.name}",
    )
