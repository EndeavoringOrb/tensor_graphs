from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def repeat_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Repeat.
    inputs: [Data Tensor]
    attrs['repeats']: int
    attrs['axis']: int (optional, default 0)
    """
    if len(inputs) != 1:
        raise ValueError("Repeat requires exactly 1 data input")

    if attrs is None or "repeats" not in attrs:
        raise ValueError("Repeat requires 'repeats' in attributes")

    data = inputs[0]

    return TensorNode(
        OpType.REPEAT,
        data.dtype,
        inputs,
        name=f"repeat_{data.name}",
        attrs=attrs,
        backend=data.backend,
    )
