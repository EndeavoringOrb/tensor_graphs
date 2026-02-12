from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def im2col_ref(inputs: List[TensorNode], attrs: Dict[str, Any] = {}) -> TensorNode:
    """
    Reference graph for Im2Col.
    inputs[0]: Input tensor [N, C, H, W]
    attrs:
        kernel_size: int
        stride: int
        padding: int
    """
    if len(inputs) != 1:
        raise ValueError("Im2Col requires exactly 1 input")

    x = inputs[0]
    kernel_size = attrs["kernel_size"]
    stride = attrs["stride"]
    padding = attrs["padding"]

    return TensorNode(
        OpType.IM2COL,
        x.dtype,
        [x],
        name=f"im2col_{x.name}",
        attrs={
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        },
        backend=x.backend,
    )
