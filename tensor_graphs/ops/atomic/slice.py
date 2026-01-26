from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def slice_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Slice: Extract sub-tensor
    inputs[0]: Data tensor
    inputs[1...3]: starts, ends, steps (optional)
    attrs["starts"]: List of start indices
    attrs["ends"]: List of end indices
    attrs["steps"]: List of step values
    """
    data = inputs[0]

    if len(inputs) == 4:
        # Explicit input nodes for slicing parameters
        starts, ends, steps = inputs[1], inputs[2], inputs[3]
        parents = [data, starts, ends, steps]
        node_attrs = {}
    elif len(inputs) == 1:
        if attrs is None:
            attrs = {}
        parents = [data]
        node_attrs = attrs
    else:
        raise ValueError("Slice requires either 1 input (with attrs) or 4 inputs")

    return TensorNode(
        OpType.SLICE,
        data.shape,
        data.dtype,
        parents,
        f"slice_{data.name}",
        attrs=node_attrs,
        backend=data.backend,
    )
