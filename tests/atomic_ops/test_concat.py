import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_concat_vectors():
    a = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "a")
    b = TensorNode(OpType.INPUT, (2,), DType.FP32, [], "b")
    # Axis is now an attribute
    concat_node = TensorNode(OpType.CONCAT, (5,), DType.FP32, [a, b], "concat", attrs={"axis": 0})

    val_a = np.array([1, 2, 3], dtype=np.float32)
    val_b = np.array([4, 5], dtype=np.float32)

    res = evaluate_graph(concat_node, {"a": val_a, "b": val_b})
    expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    np.testing.assert_array_equal(res, expected)


def test_concat_matrices_axis_0():
    """Test concatenating two matrices along axis 0 (rows)."""
    # (2, 3) + (1, 3) -> (3, 3)
    a = TensorNode(OpType.INPUT, (2, 3), DType.FP32, [], "a")
    b = TensorNode(OpType.INPUT, (1, 3), DType.FP32, [], "b")
    axis = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "axis")

    concat_node = TensorNode(OpType.CONCAT, (3, 3), DType.FP32, [a, b, axis], "concat")

    val_a = np.zeros((2, 3), dtype=np.float32)
    val_b = np.ones((1, 3), dtype=np.float32)
    val_axis = np.array([0], dtype=np.int32)

    res = evaluate_graph(concat_node, {"a": val_a, "b": val_b, "axis": val_axis})

    expected = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    np.testing.assert_array_equal(res, expected)
    assert res.shape == (3, 3)


def test_concat_matrices_axis_1():
    """Test concatenating two matrices along axis 1 (columns)."""
    # (2, 2) + (2, 1) -> (2, 3)
    a = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "a")
    b = TensorNode(OpType.INPUT, (2, 1), DType.FP32, [], "b")
    axis = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "axis")

    concat_node = TensorNode(OpType.CONCAT, (2, 3), DType.FP32, [a, b, axis], "concat")

    val_a = np.full((2, 2), 1.0, dtype=np.float32)
    val_b = np.full((2, 1), 2.0, dtype=np.float32)
    val_axis = np.array([1], dtype=np.int32)

    res = evaluate_graph(concat_node, {"a": val_a, "b": val_b, "axis": val_axis})

    expected = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)

    np.testing.assert_array_equal(res, expected)
