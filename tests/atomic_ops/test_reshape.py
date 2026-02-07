import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_reshape_vector_to_matrix():
    """Test reshaping a 1D vector (6,) to 2D (2, 3) using generic kernel."""
    data_node = TensorNode(OpType.INPUT, DType.FP32, [], (6,), "data")
    shape_node = TensorNode(OpType.INPUT, DType.INT32, [], (2,), "target_shape")

    reshape_node = TensorNode(
        OpType.RESHAPE, DType.FP32, [data_node, shape_node], (2, 3), "reshape"
    )

    input_data = np.arange(6, dtype=np.float32)
    target_shape_val = np.array([2, 3], dtype=np.int32)

    res = evaluate_graph(
        reshape_node, {"data": input_data, "target_shape": target_shape_val}
    )

    expected = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    np.testing.assert_array_equal(res, expected)
    assert res.shape == (2, 3)


def test_reshape_3d_to_flattened():
    """Test reshaping 3D (2,2,2) -> 1D (8,) using same generic kernel."""
    data_node = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2, 2), "data")
    shape_node = TensorNode(OpType.INPUT, DType.INT32, [], (1,), "target_shape")

    reshape_node = TensorNode(
        OpType.RESHAPE, DType.FP32, [data_node, shape_node], (8,), "reshape"
    )

    input_data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    target_shape_val = np.array([8], dtype=np.int32)

    res = evaluate_graph(
        reshape_node, {"data": input_data, "target_shape": target_shape_val}
    )

    expected = np.arange(8, dtype=np.float32)
    np.testing.assert_array_equal(res, expected)


def test_reshape_with_inference():
    """Test reshape using -1 (inference)."""
    data_node = TensorNode(OpType.INPUT, DType.FP32, [], (2, 3), "data")
    shape_node = TensorNode(OpType.INPUT, DType.INT32, [], (2,), "target_shape")

    # We want (3, 2)
    reshape_node = TensorNode(
        OpType.RESHAPE, DType.FP32, [data_node, shape_node], (3, 2), "reshape"
    )

    input_data = np.zeros((2, 3), dtype=np.float32)
    # Target shape: [-1, 2] -> Should become [3, 2]
    target_shape_val = np.array([-1, 2], dtype=np.int32)

    res = evaluate_graph(
        reshape_node, {"data": input_data, "target_shape": target_shape_val}
    )

    assert res.shape == (3, 2)
