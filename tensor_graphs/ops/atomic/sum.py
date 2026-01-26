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
    if len(inputs) != 1:
        raise ValueError("Sum requires exactly 1 input")

    x = inputs[0]

    if attrs is None:
        attrs = {}

    return TensorNode(
        OpType.SUM,
        x.shape if not ("axis" in attrs and attrs["axis"] is not None) else (1,),
        x.dtype,
        [x],
        f"sum_{x.name}",
        attrs=attrs,
        backend=x.backend,
    )
