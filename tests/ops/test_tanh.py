import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.ops.fused.activation import Tanh
from tensor_graphs.backend.reference import evaluate_graph


def test_tanh_basic():
    """Test element-wise tanh."""
    a = TensorNode(OpType.INPUT, (4,), DType.FP32, [], "a")

    tanh_op = Tanh()
    decomposed_graph = tanh_op.decompose([a])

    val_a = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    res = evaluate_graph(decomposed_graph, {"a": val_a})
    expected = np.tanh(val_a)

    np.testing.assert_allclose(res, expected, atol=1e-6)
