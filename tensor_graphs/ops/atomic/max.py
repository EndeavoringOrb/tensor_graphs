from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def max_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Max: max(x) or max(x, axis=...)
    inputs[0]: Input tensor
    inputs[1]: Axis (optional)
    attrs["axis"]: Axis for reduction (optional)
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
        raise ValueError("Max requires 1 or 2 inputs")

    out_shape = (None,)
    return TensorNode(OpType.MAX, out_shape, x.dtype, parents, f"max_{x.name}", attrs=attrs, backend=x.backend)
