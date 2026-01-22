import pytest
import numpy as np
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ...ops.atomic import OpType
from ...backend.reference import evaluate_graph


def test_mul_vec32():
    # Should use optimized kernel
    a = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "a")
    b = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "b")
    node = TensorNode(OpType.MUL, (32,), DType.FP32, [a, b], "mul")

    res = evaluate_graph(
        node,
        {
            "a": np.full(32, 2.0, dtype=np.float32),
            "b": np.full(32, 3.0, dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(res, np.full(32, 6.0, dtype=np.float32))
