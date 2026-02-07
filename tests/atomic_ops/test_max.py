import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_max_reduce_all():
    x = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "x")
    max_node = TensorNode(OpType.MAX, DType.FP32, [x], (1, 1), "max")

    data = np.array([[1, 5, 2], [4, 6, 3]], dtype=np.float32)
    inputs = {"x": data}
    res = evaluate_graph(max_node, inputs)

    # Defaults to global max if no axis attr
    assert res == 6.0


def test_max_axis():
    x = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "x")
    max_node = TensorNode(OpType.MAX, DType.FP32, [x], (2, 1), "max", attrs={"axis": 1})

    data = np.array([[1, 5, 2], [4, 6, 3]], dtype=np.float32)
    inputs = {"x": data}
    res = evaluate_graph(max_node, inputs)

    expected = np.array([[5], [6]], dtype=np.float32)
    np.testing.assert_array_equal(res, expected)
