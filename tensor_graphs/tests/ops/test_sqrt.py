import pytest
import numpy as np
from ...ir.node import TensorNode
from ...ir.dtypes import DType
from ...ops.atomic import OpType
from ...backend.reference import evaluate_graph


def test_sqrt_basic():
    """Test element-wise square root."""
    a = TensorNode(OpType.INPUT, (4,), DType.FP32, [], "a")
    sqrt_node = TensorNode(OpType.SQRT, (4,), DType.FP32, [a], "sqrt")

    val_a = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)

    res = evaluate_graph(sqrt_node, {"a": val_a})
    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    
    np.testing.assert_array_equal(res, expected)

def test_sqrt_matrix():
    """Test square root on a matrix."""
    a = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "a")
    sqrt_node = TensorNode(OpType.SQRT, (2, 2), DType.FP32, [a], "sqrt")

    val_a = np.array([[0.25, 100], [0, 1]], dtype=np.float32)

    res = evaluate_graph(sqrt_node, {"a": val_a})
    expected = np.array([[0.5, 10], [0, 1]], dtype=np.float32)
    
    np.testing.assert_array_equal(res, expected)