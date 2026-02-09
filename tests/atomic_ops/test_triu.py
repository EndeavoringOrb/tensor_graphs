import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_triu_k0():
    x = TensorNode(OpType.INPUT, DType.FP32, [], (3, 3), "x")
    triu_node = TensorNode(OpType.TRIU, DType.FP32, [x], (3, 3), "triu", attrs={"k": 0})

    data = np.ones((3, 3), dtype=np.float32)
    inputs = {"x": data}

    sess = GraphSession(triu_node)
    res = sess.run(inputs)

    expected = np.triu(data, 0)
    np.testing.assert_array_equal(res, expected)


def test_triu_k1():
    x = TensorNode(OpType.INPUT, DType.FP32, [], (3, 3), "x")
    triu_node = TensorNode(OpType.TRIU, DType.FP32, [x], (3, 3), "triu", attrs={"k": 1})

    data = np.ones((3, 3), dtype=np.float32)
    inputs = {"x": data}

    sess = GraphSession(triu_node)
    res = sess.run(inputs)

    expected = np.triu(data, 1)
    np.testing.assert_array_equal(res, expected)
