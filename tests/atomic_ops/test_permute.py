import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_permute_matrix_transpose():
    """Test transposing a 2D matrix (2, 3) -> (3, 2)."""
    data_node = TensorNode(OpType.INPUT, (2, 3), DType.FP32, [], "data")
    perm_node = TensorNode(OpType.INPUT, (2,), DType.INT32, [], "perm")

    # Expected output shape (3, 2)
    permute_node = TensorNode(
        OpType.PERMUTE, (3, 2), DType.FP32, [data_node, perm_node], "permute"
    )

    input_data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    perm_val = np.array([1, 0], dtype=np.int32)

    res = evaluate_graph(permute_node, {"data": input_data, "perm": perm_val})

    expected = input_data.T
    np.testing.assert_array_equal(res, expected)
    assert res.shape == (3, 2)


def test_permute_3d_reorder():
    """Test reordering 3D dimensions (0, 1, 2) -> (2, 0, 1)."""
    # Input Shape: (2, 3, 4)
    data_node = TensorNode(OpType.INPUT, (2, 3, 4), DType.FP32, [], "data")
    perm_node = TensorNode(OpType.INPUT, (3,), DType.INT32, [], "perm")

    # Output Shape: (4, 2, 3) based on perm (2, 0, 1)
    permute_node = TensorNode(
        OpType.PERMUTE, (4, 2, 3), DType.FP32, [data_node, perm_node], "permute"
    )

    input_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    perm_val = np.array([2, 0, 1], dtype=np.int32)

    res = evaluate_graph(permute_node, {"data": input_data, "perm": perm_val})

    expected = np.transpose(input_data, (2, 0, 1))
    np.testing.assert_array_equal(res, expected)
