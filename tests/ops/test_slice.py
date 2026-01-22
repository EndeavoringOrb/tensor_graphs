import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_slice_basic_1d():
    """Test simple 1D slicing: data[2:6:2]"""
    data_node = TensorNode(OpType.INPUT, (10,), DType.FP32, [], "data")
    starts_node = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "starts")
    ends_node = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "ends")
    steps_node = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "steps")

    slice_node = TensorNode(
        OpType.SLICE,
        (2,),
        DType.FP32,
        [data_node, starts_node, ends_node, steps_node],
        "slice",
    )

    input_data = np.arange(10, dtype=np.float32)
    starts = np.array([2], dtype=np.int32)
    ends = np.array([6], dtype=np.int32)
    steps = np.array([2], dtype=np.int32)

    res = evaluate_graph(
        slice_node, {"data": input_data, "starts": starts, "ends": ends, "steps": steps}
    )

    expected = input_data[2:6:2]
    np.testing.assert_array_equal(res, expected)


def test_slice_multi_dim():
    """Test multi-dim slicing: data[0:2, 1:3]"""
    data_node = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], "data")
    starts_node = TensorNode(OpType.INPUT, (2,), DType.INT32, [], "starts")
    ends_node = TensorNode(OpType.INPUT, (2,), DType.INT32, [], "ends")
    steps_node = TensorNode(OpType.INPUT, (2,), DType.INT32, [], "steps")

    slice_node = TensorNode(
        OpType.SLICE,
        (2, 2),
        DType.FP32,
        [data_node, starts_node, ends_node, steps_node],
        "slice",
    )

    input_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    starts = np.array([0, 1], dtype=np.int32)
    ends = np.array([2, 3], dtype=np.int32)
    steps = np.array([1, 1], dtype=np.int32)

    res = evaluate_graph(
        slice_node, {"data": input_data, "starts": starts, "ends": ends, "steps": steps}
    )

    expected = input_data[0:2, 1:3]
    np.testing.assert_array_equal(res, expected)
