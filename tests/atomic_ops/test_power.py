import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_power_basic():
    base = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "base")
    exp = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "exp")
    pow_node = TensorNode(OpType.POWER, (3,), DType.FP32, [base, exp], "pow")

    inputs = {
        "base": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "exp": np.array([3.0, 2.0, 0.5], dtype=np.float32),
    }

    res = evaluate_graph(pow_node, inputs)
    expected = np.array([8.0, 9.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(res, expected)
