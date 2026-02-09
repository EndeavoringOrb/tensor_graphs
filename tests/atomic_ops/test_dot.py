import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_dot_2x2():
    # Should use 2x2 optimized kernel
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "b")
    node = TensorNode(OpType.DOT, DType.FP32, [a, b], (2, 2), "dot")

    mat = np.eye(2, dtype=np.float32)
    sess = GraphSession(node)
    res = sess.run({"a": mat, "b": mat})
    np.testing.assert_array_equal(res, mat)


def test_dot_broadcast_rhs_2d_with_batch_lhs():
    # A: (1, M, K), B: (K, N) -> result should be (1, M, N)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (1, 2, 3), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (3, 4), "b")
    node = TensorNode(OpType.DOT, DType.FP32, [a, b], (1, 2, 4), "dot_brd")

    a_val = np.arange(1 * 2 * 3, dtype=np.float32).reshape(1, 2, 3)
    b_val = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)

    sess = GraphSession(node)
    res = sess.run({"a": a_val, "b": b_val})
    # Reference using numpy.matmul (will broadcast B)
    expected = np.matmul(a_val, b_val)
    np.testing.assert_array_almost_equal(res, expected)
