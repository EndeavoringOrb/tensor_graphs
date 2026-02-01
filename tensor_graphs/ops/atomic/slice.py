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

    # Simple static shape inference (approximation)
    # Ideally handled by ShapeInference, but we set what we can here
    out_shape = list(data.shape)
    starts = attrs["starts"]
    ends = attrs["ends"]
    steps = attrs.get("steps", [1] * len(starts))

    # Note: Logic to compute exact output shape is complex due to broadcasting/step logic
    # We leave exact calculation to ShapeInference or assume caller provided generic shape
    # For now, we preserve rank.

    return TensorNode(
        OpType.SLICE,
        tuple(out_shape),  # Placeholder, effectively
        data.dtype,
        inputs,
        f"slice_{data.name}",
        attrs=attrs,
        backend=data.backend,
    )
