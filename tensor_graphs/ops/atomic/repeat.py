from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def repeat_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Repeat: Repeat elements along an axis
    inputs[0]: Data tensor
    inputs[1]: Repeats (optional)
    attrs["repeats"]: Number of repeats
    attrs["axis"]: Axis along which to repeat (default: 0)
    """
    if len(inputs) == 1:
        data = inputs[0]
        if attrs is None:
            attrs = {}
        parents = [data]
        node_attrs = attrs
    elif len(inputs) == 2:
        data = inputs[0]
        repeats = inputs[1]
        if attrs is None:
            attrs = {}
        parents = [data, repeats]
        node_attrs = attrs
    else:
        raise ValueError("Repeat requires 1 or 2 inputs")

    return TensorNode(
        OpType.REPEAT,
        data.shape,
        data.dtype,
        parents,
        f"repeat_{data.name}",
        attrs=node_attrs,
        backend=data.backend,
    )
