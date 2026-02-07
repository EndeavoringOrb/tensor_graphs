import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_repeat():
    x = TensorNode(OpType.INPUT, DType.FP32, [], (2,), "x")
    rep_node = TensorNode(
        OpType.REPEAT, DType.FP32, [x], (6,), "rep", attrs={"repeats": 3, "axis": 0}
    )

    data = np.array([10, 20], dtype=np.float32)
    res = evaluate_graph(rep_node, {"x": data})

    expected = np.array([10, 10, 10, 20, 20, 20], dtype=np.float32)
    np.testing.assert_array_equal(res, expected)
