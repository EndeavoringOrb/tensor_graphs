from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def fill_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    if len(inputs) != 2:
        raise ValueError("Fill requires exactly 2 inputs: [value_tensor, shape_tensor]")

    value_tensor = inputs[0]
    shape_tensor = inputs[1]

    return TensorNode(
        OpType.FILL,
        value_tensor.dtype,
        [value_tensor, shape_tensor],
        name=f"fill_{value_tensor.name}_{shape_tensor.name}",
        backend=value_tensor.backend,
    )
