import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_concat_vectors():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (3,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (2,), "b")
    # Axis is now an attribute
    concat_node = TensorNode(
        OpType.CONCAT, DType.FP32, [a, b], (5,), "concat", attrs={"axis": 0}
    )

    val_a = np.array([1, 2, 3], dtype=np.float32)
    val_b = np.array([4, 5], dtype=np.float32)

    sess = GraphSession(concat_node)
    res = sess.run({"a": val_a, "b": val_b})
    expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    np.testing.assert_array_equal(res, expected)


def test_concat_matrices_axis_0():
    """Test concatenating two matrices along axis 0 (rows)."""
    # (2, 3) + (1, 3) -> (3, 3)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (1, 3), "b")

    concat_node = TensorNode(
        OpType.CONCAT, DType.FP32, [a, b], (3, 3), "concat", attrs={"axis": 0}
    )

    val_a = np.zeros((2, 3), dtype=np.float32)
    val_b = np.ones((1, 3), dtype=np.float32)

    sess = GraphSession(concat_node)
    res = sess.run({"a": val_a, "b": val_b})

    expected = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    np.testing.assert_array_equal(res, expected)
    assert res.shape == (3, 3)


def test_concat_matrices_axis_1():
    """Test concatenating two matrices along axis 1 (columns)."""
    # (2, 2) + (2, 1) -> (2, 3)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (2, 1), "b")

    concat_node = TensorNode(
        OpType.CONCAT, DType.FP32, [a, b], (2, 3), "concat", attrs={"axis": 1}
    )

    val_a = np.full((2, 2), 1.0, dtype=np.float32)
    val_b = np.full((2, 1), 2.0, dtype=np.float32)

    sess = GraphSession(concat_node)
    res = sess.run({"a": val_a, "b": val_b})

    expected = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)

    np.testing.assert_array_equal(res, expected)
