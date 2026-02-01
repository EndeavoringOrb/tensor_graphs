import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_exp_basic():
    """Test element-wise exponential."""
    a = TensorNode(OpType.INPUT, DType.FP32, [], (4,), "a")
    exp_node = TensorNode(OpType.EXP, DType.FP32, [a], (4,), "exp")

    val_a = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)

    res = evaluate_graph(exp_node, {"a": val_a})
    expected = np.exp(val_a)

    np.testing.assert_allclose(res, expected, atol=1e-6)


def test_exp_matrix():
    """Test exponential on a matrix."""
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "a")
    exp_node = TensorNode(OpType.EXP, DType.FP32, [a], (2, 2), "exp")

    val_a = np.array([[0.0, 1.0], [0.5, -0.5]], dtype=np.float32)

    res = evaluate_graph(exp_node, {"a": val_a})
    expected = np.exp(val_a)

    np.testing.assert_allclose(res, expected, atol=1e-6)
