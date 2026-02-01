from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def concat_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Concat: Concatenate list of tensors along axis.
    inputs: List[TensorNode] (Data tensors)
    attrs['axis']: int (Required)
    """
    if attrs is None or "axis" not in attrs:
        raise ValueError("Concat requires 'axis' in attributes")

    return TensorNode(
        OpType.CONCAT,
        inputs[0].dtype,
        inputs,
        name=f"concat_{len(inputs)}_inputs",
        attrs=attrs,
    )
