import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_add_generic_vector():
    # Shape (10,) matches (None,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (10,), "b")
    add_node = TensorNode(OpType.ADD, DType.FP32, [a, b], (10,), "add")

    val_a = np.ones(10, dtype=np.float32)
    val_b = np.ones(10, dtype=np.float32) * 2

    res = evaluate_graph(add_node, {"a": val_a, "b": val_b})
    np.testing.assert_array_equal(res, np.ones(10) * 3)


def test_add_vec32_optimized():
    # Shape (32,) matches (32,)
    a = TensorNode(OpType.INPUT, DType.FP32, [], (32,), "a")
    b = TensorNode(OpType.INPUT, DType.FP32, [], (32,), "b")
    add_node = TensorNode(OpType.ADD, DType.FP32, [a, b], (32,), "add")

    val_a = np.ones(32, dtype=np.float32)
    val_b = np.ones(32, dtype=np.float32)

    res = evaluate_graph(add_node, {"a": val_a, "b": val_b})
    np.testing.assert_array_equal(res, np.ones(32) * 2)


def test_add_broadcast():
    s = TensorNode(OpType.INPUT, DType.FP32, [], (1,), "s")
    m = TensorNode(OpType.INPUT, DType.FP32, [], (4, 4), "m")
    add_node = TensorNode(OpType.ADD, DType.FP32, [s, m], (4, 4), "add")

    val_s = np.array([10.0], dtype=np.float32)
    val_m = np.ones((4, 4), dtype=np.float32)

    res = evaluate_graph(add_node, {"s": val_s, "m": val_m})
    np.testing.assert_array_equal(res, np.ones((4, 4)) * 11)
