from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic_types import OpType


def rope_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    x, cos, sin = inputs
    D = x.shape[-1]
    half_d = D // 2 if D is not None else None

    # x1 = x[..., :half_d]
    x1_node = x[..., :half_d]

    # x2 = x[..., half_d:]
    x2_node = x[..., half_d:]

    neg_x2_node = TensorNode(
        op_type=OpType.NEGATE,
        shape=x2_node.shape,
        dtype=x2_node.dtype,
        parents=[x2_node],
        name=f"{x.name}_neg_x2",
    )

    rotated_shape = x.shape
    axis_node = TensorNode(
        OpType.CONSTANT, (1,), DType.INT32, [], f"{x.name}_axis", attrs={"value": [-1]}
    )

    rotated_node = TensorNode(
        op_type=OpType.CONCAT,
        shape=rotated_shape,
        dtype=x.dtype,
        parents=[neg_x2_node, x1_node, axis_node],
        name=f"{x.name}_rotated",
    )

    term1_node = TensorNode(
        op_type=OpType.MUL,
        shape=x.shape,
        dtype=x.dtype,
        parents=[x, cos],
        name=f"{x.name}_term1_mul_cos",
    )

    term2_node = TensorNode(
        op_type=OpType.MUL,
        shape=x.shape,
        dtype=x.dtype,
        parents=[rotated_node, sin],
        name=f"{x.name}_term2_mul_sin",
    )

    return TensorNode(
        op_type=OpType.ADD,
        shape=x.shape,
        dtype=x.dtype,
        parents=[term1_node, term2_node],
        name=f"{x.name}_rope_out",
    )
