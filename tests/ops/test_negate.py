import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_negate_basic():
    """Test element-wise negation."""
    a = TensorNode(OpType.INPUT, (4,), DType.FP32, [], "a")
    neg_node = TensorNode(OpType.NEGATE, (4,), DType.FP32, [a], "neg")

    val_a = np.array([1.0, -2.0, 0.0, 4.5], dtype=np.float32)

    res = evaluate_graph(neg_node, {"a": val_a})
    expected = np.array([-1.0, 2.0, -0.0, -4.5], dtype=np.float32)

    np.testing.assert_array_equal(res, expected)
