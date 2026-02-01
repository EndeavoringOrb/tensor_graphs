from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def sum_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Sum: sum(x) or sum(x, axis=...)
    inputs[0]: Input tensor
    attrs["axis"]: Axis for reduction (optional)
    attrs["keepdims"]: Whether to keep reduced dimensions (optional, default: True)
    """
    if attrs is None:
        attrs = {}

    if len(inputs) == 1:
        x = inputs[0]
        parents = [x]
    else:
        raise ValueError("Sum requires 1 input")

    return TensorNode(
        OpType.SUM,
        x.dtype,
        parents,
        name=f"sum_{x.name}",
        attrs=attrs,
        backend=x.backend,
    )
