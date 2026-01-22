from typing import List
import numpy as np
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic import OpType
from ..interface import CompositeOp
from ..registry import register_composite


@register_composite
class GELU(CompositeOp):
    op_type = "GELU"

    def decompose(self, inputs: List[TensorNode]) -> TensorNode:
        # 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = inputs[0]

        # Constants would be Input nodes in a real graph, simplified here
        c_cube = TensorNode(OpType.INPUT, (1,), x.dtype, [], "c_0.044")
        c_sqrt = TensorNode(OpType.INPUT, (1,), x.dtype, [], "c_sqrt_2_pi")
        c_half = TensorNode(OpType.INPUT, (1,), x.dtype, [], "c_0.5")
        c_one = TensorNode(OpType.INPUT, (1,), x.dtype, [], "c_1.0")

        # x^3
        x2 = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "x2")
        x3 = TensorNode(OpType.MUL, x.shape, x.dtype, [x2, x], "x3")

        # inner
        term1 = TensorNode(OpType.MUL, x.shape, x.dtype, [x3, c_cube], "term1")
        term2 = TensorNode(OpType.ADD, x.shape, x.dtype, [x, term1], "term2")
        inner = TensorNode(OpType.MUL, x.shape, x.dtype, [term2, c_sqrt], "inner")

        # tanh
        tanh_node = Tanh().decompose([inner])

        # outer
        one_plus = TensorNode(
            OpType.ADD, x.shape, x.dtype, [c_one, tanh_node], "one_plus"
        )
        half_x = TensorNode(OpType.MUL, x.shape, x.dtype, [x, c_half], "half_x")

        return TensorNode(OpType.MUL, x.shape, x.dtype, [half_x, one_plus], "gelu_out")


@register_composite
class Softmax(CompositeOp):
    op_type = "Softmax"

    def decompose(self, inputs: List[TensorNode]) -> TensorNode:
        x = inputs[0]
        # Max
        # We need an axis. Assuming last axis (-1)
        axis = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "axis")

        x_max = TensorNode(OpType.MAX, x.shape, x.dtype, [x, axis], "max")
        # Sub
        shifted = TensorNode(
            OpType.ADD,
            x.shape,
            x.dtype,
            [x, TensorNode(OpType.NEGATE, x.shape, x.dtype, [x_max], "neg_max")],
            "shifted",
        )

        # Exp
        exps = TensorNode(OpType.EXP, x.shape, x.dtype, [shifted], "exps")

        # Sum
        sum_exps = TensorNode(OpType.SUM, x.shape, x.dtype, [exps, axis], "sum_exps")

        # Div
        return TensorNode(
            OpType.DIVIDE, x.shape, x.dtype, [exps, sum_exps], "softmax_out"
        )


@register_composite
class Tanh(CompositeOp):
    op_type = "Tanh"

    def decompose(self, inputs: List[TensorNode]) -> TensorNode:
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
