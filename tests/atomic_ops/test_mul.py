import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_mul_vec32():
    # Should use optimized kernel
    a = TensorNode(OpType.INPUT, DType.FP32, [], (32,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (32,), "b")
    node = TensorNode(OpType.MUL, DType.FP32, [a, b], (32,), "mul")

    sess = GraphSession(node)
    res = sess.run(
        {
            "a": np.full(32, 2.0, dtype=np.float32),
            "b": np.full(32, 3.0, dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(res, np.full(32, 6.0, dtype=np.float32))
