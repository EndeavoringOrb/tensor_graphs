import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_sum_global():
    x = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "x")
    sum_node = TensorNode(OpType.SUM, DType.FP32, [x], (1,), "sum")

    data = np.ones((2, 2), dtype=np.float32)
    inputs = {"x": data}
    res = evaluate_graph(sum_node, inputs)

    # Defaults to global sum, keepdims=True by default in atomic/sum.py if not specified
    assert res == 4.0


def test_sum_axis_keepdims():
    x = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "x")
    sum_node = TensorNode(
        OpType.SUM, DType.FP32, [x], (2, 1), "sum", attrs={"axis": 1, "keepdims": True}
    )

    data = np.ones((2, 3), dtype=np.float32)
    inputs = {"x": data}
    res = evaluate_graph(sum_node, inputs)

    expected = np.full((2, 1), 3.0, dtype=np.float32)
    np.testing.assert_array_equal(res, expected)
