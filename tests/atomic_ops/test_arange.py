import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_arange_basic():
    start = TensorNode(OpType.INPUT, DType.INT32, [], (1,), "start")
    stop = TensorNode(OpType.INPUT, DType.INT32, [], (1,), "stop")
    step = TensorNode(OpType.INPUT, DType.INT32, [], (1,), "step")

    arange_node = TensorNode(
        OpType.ARANGE, DType.INT32, [start, stop, step], (None,), "arange"
    )

    inputs = {
        "start": np.array([0], dtype=np.int32),
        "stop": np.array([5], dtype=np.int32),
        "step": np.array([1], dtype=np.int32),
    }

    res = evaluate_graph(arange_node, inputs)
    expected = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    np.testing.assert_array_equal(res, expected)
