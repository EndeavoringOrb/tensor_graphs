import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_div_generic_vector():
    """Test element-wise division of vectors."""
    a = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "b")
    div_node = TensorNode(OpType.DIVIDE, DType.FP32, [a, b], (10,), "div")

    val_a = np.full(10, 10.0, dtype=np.float32)
    val_b = np.full(10, 2.0, dtype=np.float32)

    res = evaluate_graph(div_node, {"a": val_a, "b": val_b})
    np.testing.assert_array_equal(res, np.full(10, 5.0, dtype=np.float32))


def test_div_scalar_broadcast():
    """Test Scalar / Matrix broadcasting."""
    s = TensorNode(OpType.INPUT, DType.FP32, [], (1,), "s")
    m = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "m")
    div_node = TensorNode(OpType.DIVIDE, DType.FP32, [s, m], (2, 2), "div")

    val_s = np.array([10.0], dtype=np.float32)
    val_m = np.full((2, 2), 2.0, dtype=np.float32)

    res = evaluate_graph(div_node, {"s": val_s, "m": val_m})
    # 10 / 2 = 5
    np.testing.assert_array_equal(res, np.full((2, 2), 5.0, dtype=np.float32))


def test_div_matrix_scalar():
    """Test Matrix / Scalar broadcasting."""
    m = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "m")
    s = TensorNode(OpType.INPUT, DType.FP32, [], (1,), "s")
    div_node = TensorNode(OpType.DIVIDE, DType.FP32, [m, s], (2, 2), "div")

    val_m = np.full((2, 2), 20.0, dtype=np.float32)
    val_s = np.array([4.0], dtype=np.float32)

    res = evaluate_graph(div_node, {"m": val_m, "s": val_s})
    # 20 / 4 = 5
    np.testing.assert_array_equal(res, np.full((2, 2), 5.0, dtype=np.float32))
