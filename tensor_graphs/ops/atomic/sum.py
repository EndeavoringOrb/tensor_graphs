from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def sum_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Sum: sum(x) or sum(x, axis=...)
    inputs[0]: Input tensor
    inputs[1]: Axis (optional)
    attrs["axis"]: Axis for reduction (optional)
    attrs["keepdims"]: Whether to keep reduced dimensions (optional, default: True)
    """
    if attrs is None:
        attrs = {}

    if len(inputs) == 1:
        x = inputs[0]
        parents = [x]
    elif len(inputs) == 2:
        x = inputs[0]
        axis_node = inputs[1]
        parents = [x, axis_node]
    else:
        raise ValueError("Sum requires 1 or 2 inputs")

    # Global sum (axis=None) or per-axis sum both produce (1,) when keepdims=True
    out_shape = (1,)

    return TensorNode(
        OpType.SUM,
        out_shape,
        x.dtype,
        parents,
        f"sum_{x.name}",
        attrs=attrs,
        backend=x.backend,
    )
