import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_exp_basic():
    """Test element-wise exponential."""
    a = TensorNode(OpType.INPUT, (4,), DType.FP32, [], "a")
    exp_node = TensorNode(OpType.EXP, (4,), DType.FP32, [a], "exp")

    val_a = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)

    res = evaluate_graph(exp_node, {"a": val_a})
    expected = np.exp(val_a)
    
    np.testing.assert_allclose(res, expected, atol=1e-6)

def test_exp_matrix():
    """Test exponential on a matrix."""
    a = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "a")
    exp_node = TensorNode(OpType.EXP, (2, 2), DType.FP32, [a], "exp")

    val_a = np.array([[0.0, 1.0], [0.5, -0.5]], dtype=np.float32)

    res = evaluate_graph(exp_node, {"a": val_a})
    expected = np.exp(val_a)
    
    np.testing.assert_allclose(res, expected, atol=1e-6)