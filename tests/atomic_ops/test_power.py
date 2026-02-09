import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_power_basic():
    base = TensorNode(OpType.INPUT, DType.FP32, [], (3,), "base")
    exp = TensorNode(OpType.INPUT, DType.FP32, [], (3,), "exp")
    pow_node = TensorNode(OpType.POWER, DType.FP32, [base, exp], (3,), "pow")

    inputs = {
        "base": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "exp": np.array([3.0, 2.0, 0.5], dtype=np.float32),
    }

    sess = GraphSession(pow_node)
    res = sess.run(inputs)
    expected = np.array([8.0, 9.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(res, expected)
