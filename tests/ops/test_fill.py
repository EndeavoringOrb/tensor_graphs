import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_fill_fp32():
    # Value input: Scalar 5.0
    val_node = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "val")

    # Shape input: Tensor containing [2, 3] -> Shape is (2,)
    shape_node = TensorNode(OpType.INPUT, (2,), DType.INT32, [], "shape_tensor")

    # Fill node: Output shape is (2, 3)
    fill_node = TensorNode(
        OpType.FILL, (2, 3), DType.FP32, [val_node, shape_node], "fill"
    )

    val_input = np.array([5.0], dtype=np.float32)
    shape_input = np.array([2, 3], dtype=np.int32)

    res = evaluate_graph(fill_node, {"val": val_input, "shape_tensor": shape_input})

    expected = np.full((2, 3), 5.0, dtype=np.float32)
    np.testing.assert_array_equal(res, expected)


def test_fill_int32():
    # Value input: Scalar 7
    val_node = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "val")

    # Shape input: Tensor containing [4] -> Shape is (1,)
    shape_node = TensorNode(OpType.INPUT, (1,), DType.INT32, [], "shape_tensor")

    # Fill node: Output shape is (4,)
    fill_node = TensorNode(
        OpType.FILL, (4,), DType.INT32, [val_node, shape_node], "fill"
    )

    val_input = np.array([7], dtype=np.int32)
    shape_input = np.array([4], dtype=np.int32)

    res = evaluate_graph(fill_node, {"val": val_input, "shape_tensor": shape_input})

    expected = np.full((4,), 7, dtype=np.int32)
    np.testing.assert_array_equal(res, expected)
