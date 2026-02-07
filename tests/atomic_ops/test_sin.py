import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_sin_basic():
    """Test element-wise sine."""
    a = TensorNode(OpType.INPUT, DType.FP32, [], (4,), "a")
    sin_node = TensorNode(OpType.SIN, DType.FP32, [a], (4,), "sin")

    val_a = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float32)

    res = evaluate_graph(sin_node, {"a": val_a})
    # Use close for float comparisons
    expected = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)

    np.testing.assert_allclose(res, expected, atol=1e-6)
