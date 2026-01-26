from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def max_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Max: max(x) or max(x, axis=...)
    inputs[0]: Input tensor
    attrs["axis"]: Axis for reduction (optional)
    """
    if len(inputs) != 1:
        raise ValueError("Max requires exactly 1 input")

    x = inputs[0]

    if attrs is None:
        attrs = {}

    return TensorNode(
        OpType.MAX,
        x.shape if not ("axis" in attrs and attrs["axis"] is not None) else (1,),
        x.dtype,
        [x],
        f"max_{x.name}",
        attrs=attrs,
        backend=x.backend,
    )
