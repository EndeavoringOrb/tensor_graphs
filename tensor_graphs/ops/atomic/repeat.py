from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def repeat_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Repeat: Repeat elements along an axis
    inputs[0]: Data tensor
    attrs["repeats"]: Number of repeats
    attrs["axis"]: Axis along which to repeat (default: 0)
    """
    if len(inputs) != 1:
        raise ValueError("Repeat requires exactly 1 input: data tensor")

    data = inputs[0]

    if attrs is None:
        attrs = {}

    return TensorNode(
        OpType.REPEAT,
        data.shape,
        data.dtype,
        [data],
        f"repeat_{data.name}",
        attrs=attrs,
        backend=data.backend,
    )
