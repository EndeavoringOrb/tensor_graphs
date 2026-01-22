import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_cos_basic():
    """Test element-wise cosine."""
    a = TensorNode(OpType.INPUT, (4,), DType.FP32, [], "a")
    cos_node = TensorNode(OpType.COS, (4,), DType.FP32, [a], "cos")

    val_a = np.array([0, np.pi/2, np.pi, 3*np.pi/2], dtype=np.float32)

    res = evaluate_graph(cos_node, {"a": val_a})
    expected = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    
    np.testing.assert_allclose(res, expected, atol=1e-6)