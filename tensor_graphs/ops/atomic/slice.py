from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def slice_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Slice: Extract sub-tensor
    inputs[0]: Data tensor
    attrs["starts"]: List of start indices
    attrs["ends"]: List of end indices
    attrs["steps"]: List of step values (optional, default: 1 for all)
    """
    if len(inputs) != 1:
        raise ValueError("Slice requires exactly 1 input: data tensor")

    data = inputs[0]

    if attrs is None:
        attrs = {}

    return TensorNode(
        OpType.SLICE,
        data.shape,
        data.dtype,
        [data],
        f"slice_{data.name}",
        attrs=attrs,
        backend=data.backend,
    )
