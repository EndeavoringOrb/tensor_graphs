from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic_types import OpType


def cast_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Cast: A -> B (cast from input dtype to target dtype)
    """
    if len(inputs) != 1:
        raise ValueError("Cast requires exactly 1 input")

    input_tensor = inputs[0]
    target_dtype = attrs.get("to", DType.FP32) if attrs else DType.FP32

    return TensorNode(
        OpType.CAST,
        target_dtype,
        [input_tensor],
        name=f"cast_{input_tensor.name}",
        attrs={"to": target_dtype},
        backend=input_tensor.backend,
    )
