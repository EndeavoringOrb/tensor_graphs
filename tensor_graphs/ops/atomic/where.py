from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic_types import OpType


def where_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Where: where(condition, x, y)
    inputs[0]: Condition tensor (bool or int32)
    inputs[1]: X tensor
    inputs[2]: Y tensor
    """
    if len(inputs) != 3:
        raise ValueError("Where requires exactly 3 inputs: condition, x, and y")

    condition, x, y = inputs

    return TensorNode(
        OpType.WHERE,
        x.shape,
        x.dtype,
        [condition, x, y],
        f"where_{x.name}",
        backend=x.backend,
    )
