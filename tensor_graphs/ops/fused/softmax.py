from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def softmax_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    x = inputs[0]
    axis = attrs.get("axis", -1) if attrs else -1

    reduced_shape = list(x.shape)
    reduced_shape[axis] = 1

    x_max = TensorNode(
        OpType.MAX,
        tuple(reduced_shape),
        x.dtype,
        [x],
        "max",
        attrs={"axis": axis, "keepdims": True},
    )

    shifted = TensorNode(
        OpType.ADD,
        x.shape,
        x.dtype,
        [x, TensorNode(OpType.NEGATE, x.shape, x.dtype, [x_max], "neg_max")],
        "shifted",
    )

    exps = TensorNode(OpType.EXP, x.shape, x.dtype, [shifted], "exps")

    sum_exps = TensorNode(
        OpType.SUM,
        tuple(reduced_shape),
        x.dtype,
        [exps],
        "sum_exps",
        attrs={"axis": axis, "keepdims": True},
    )

    return TensorNode(OpType.DIVIDE, x.shape, x.dtype, [exps, sum_exps], "softmax_out")
