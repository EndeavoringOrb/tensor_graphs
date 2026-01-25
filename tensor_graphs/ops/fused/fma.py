from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def fused_mul_add_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for FusedMulAdd: (A * B) + C
    """
    if len(inputs) != 3:
        raise ValueError("FusedMulAdd requires 3 inputs")

    a, b, c = inputs
    mul_node = TensorNode(OpType.MUL, a.shape, a.dtype, [a, b], f"decomp_mul_{a.name}")
    add_node = TensorNode(
        OpType.ADD, a.shape, a.dtype, [mul_node, c], f"decomp_fma_{a.name}"
    )
    return add_node
