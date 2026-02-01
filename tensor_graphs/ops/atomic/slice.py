from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def slice_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Slice.
    inputs: [Data Tensor]
    attrs['starts']: List[int]
    attrs['ends']: List[int]
    attrs['steps']: List[int] (optional)
    """
    if len(inputs) != 1:
        raise ValueError(
            "Slice requires exactly 1 data input. Params must be in attrs."
        )

    if attrs is None or "starts" not in attrs or "ends" not in attrs:
        raise ValueError("Slice requires 'starts' and 'ends' in attributes")

    data = inputs[0]

    return TensorNode(
        OpType.SLICE,
        data.dtype,
        inputs,
        name=f"slice_{data.name}",
        attrs=attrs,
        backend=data.backend,
    )
