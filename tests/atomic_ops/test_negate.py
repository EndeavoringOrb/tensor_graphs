import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_negate_basic():
    """Test element-wise negation."""
    a = TensorNode(OpType.INPUT, DType.FP32, [], (4,), "a")
    neg_node = TensorNode(OpType.NEGATE, DType.FP32, [a], (4,), "neg")

    val_a = np.array([1.0, -2.0, 0.0, 4.5], dtype=np.float32)

    sess = GraphSession(neg_node)
    res = sess.run({"a": val_a})
    expected = np.array([-1.0, 2.0, -0.0, -4.5], dtype=np.float32)

    np.testing.assert_array_equal(res, expected)
