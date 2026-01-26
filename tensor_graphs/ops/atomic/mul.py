from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def mul_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for Multiply: A * B
    inputs[0]: Tensor A
    inputs[1]: Tensor B
    """
    if len(inputs) != 2:
        raise ValueError("Mul requires 2 inputs")

    a, b = inputs

    return TensorNode(
        OpType.MUL,
        a.shape,
        a.dtype,
        [a, b],
        f"mul_{a.name}_{b.name}",
        backend=a.backend,
    )
