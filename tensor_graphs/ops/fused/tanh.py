from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ..atomic_types import OpType


def tanh_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    x = inputs[0]
    # e^x
    exp_x = TensorNode(OpType.EXP, x.shape, x.dtype, [x], "exp_x")
    # e^(-x)
    neg_x = TensorNode(OpType.NEGATE, x.shape, x.dtype, [x], "neg_x")
    exp_neg_x = TensorNode(OpType.EXP, x.shape, x.dtype, [neg_x], "exp_neg_x")
    # e^x - e^(-x)
    numerator = TensorNode(
        OpType.ADD,
        x.shape,
        x.dtype,
        [
            exp_x,
            TensorNode(
                OpType.NEGATE,
                exp_neg_x.shape,
                exp_neg_x.dtype,
                [exp_neg_x],
                "neg_exp_neg_x",
            ),
        ],
        "numerator",
    )
    # e^x + e^(-x)
    denominator = TensorNode(
        OpType.ADD, x.shape, x.dtype, [exp_x, exp_neg_x], "denominator"
    )
    # (e^x - e^(-x)) / (e^x + e^(-x))
    return TensorNode(
        OpType.DIVIDE, x.shape, x.dtype, [numerator, denominator], "tanh_out"
    )
