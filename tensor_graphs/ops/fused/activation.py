from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ..atomic import OpType
from ..interface import CompositeOp
from ..registry import register_composite


@register_composite
class GELU(CompositeOp):
    op_type = "GELU"

    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
    ) -> TensorNode:
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = inputs[0]
        c_cube = TensorNode(
            OpType.CONSTANT,
            (1,),
            x.dtype,
            [],
            "c_0.044",
            attrs={"value": 0.044715},
        )
        c_sqrt = TensorNode(
            OpType.CONSTANT,
            (1,),
            x.dtype,
            [],
            "c_sqrt_2_pi",
            attrs={"value": np.sqrt(2 / np.pi)},
        )
        c_half = TensorNode(
            OpType.CONSTANT, (1,), x.dtype, [], "c_0.5", attrs={"value": 0.5}
        )
        c_one = TensorNode(
            OpType.CONSTANT, (1,), x.dtype, [], "c_1.0", attrs={"value": 1.0}
        )

        # x^3
        x2 = TensorNode(OpType.MUL, x.shape, x.dtype, [x, x], "x2")
        x3 = TensorNode(OpType.MUL, x.shape, x.dtype, [x2, x], "x3")

        # inner
        term1 = TensorNode(OpType.MUL, x.shape, x.dtype, [x3, c_cube], "term1")
        term2 = TensorNode(OpType.ADD, x.shape, x.dtype, [x, term1], "term2")
        inner = TensorNode(OpType.MUL, x.shape, x.dtype, [term2, c_sqrt], "inner")

        # tanh
        tanh_node = Tanh().decompose([inner], attrs)

        # outer
        one_plus = TensorNode(
            OpType.ADD, x.shape, x.dtype, [c_one, tanh_node], "one_plus"
        )
        half_x = TensorNode(OpType.MUL, x.shape, x.dtype, [x, c_half], "half_x")

        return TensorNode(OpType.MUL, x.shape, x.dtype, [half_x, one_plus], "gelu_out")

    def sample_inputs(self) -> List[Tuple[List[np.ndarray], Dict[str, Any]]]:
        # Case 1: Simple random
        x = np.random.randn(4, 4).astype(np.float32)
        return [([x], {})]


@register_composite
class Softmax(CompositeOp):
    op_type = "Softmax"

    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
    ) -> TensorNode:
        x = inputs[0]
        # Max
        # Use axis from attributes if provided, default to -1
        axis = attrs.get("axis", -1) if attrs else -1

        # Calculate shape for reduction (keepdims=True)
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
        sum_exps = TensorNode(
            OpType.SUM,
            tuple(reduced_shape),
            x.dtype,
            [exps],
            "sum_exps",
            attrs={"axis": axis, "keepdims": True},
        )

        # Div
        return TensorNode(
            OpType.DIVIDE, x.shape, x.dtype, [exps, sum_exps], "softmax_out"
        )

    def sample_inputs(self) -> List[Tuple[List[np.ndarray], Dict[str, Any]]]:
        x = np.random.randn(2, 5).astype(np.float32)
        return [([x], {"axis": -1})]


@register_composite
class Tanh(CompositeOp):
    op_type = "Tanh"

    def decompose(
        self, inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
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

    def sample_inputs(self) -> List[Tuple[List[np.ndarray], Dict[str, Any]]]:
        x = np.random.randn(10).astype(np.float32)
        return [([x], {})]
