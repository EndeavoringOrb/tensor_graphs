import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_triu_k0():
    x = TensorNode(OpType.INPUT, (3, 3), DType.FP32, [], "x")
    triu_node = TensorNode(OpType.TRIU, (3, 3), DType.FP32, [x], "triu", attrs={"k": 0})

    data = np.ones((3, 3), dtype=np.float32)
    inputs = {"x": data, "k": np.array([0], dtype=np.int32)}
    res = evaluate_graph(triu_node, inputs)

    expected = np.triu(data, 0)
    np.testing.assert_array_equal(res, expected)


def test_triu_k1():
    x = TensorNode(OpType.INPUT, (3, 3), DType.FP32, [], "x")
    triu_node = TensorNode(OpType.TRIU, (3, 3), DType.FP32, [x], "triu", attrs={"k": 1})

    data = np.ones((3, 3), dtype=np.float32)
    inputs = {"x": data, "k": np.array([1], dtype=np.int32)}
    res = evaluate_graph(triu_node, inputs)

    expected = np.triu(data, 1)
    np.testing.assert_array_equal(res, expected)
