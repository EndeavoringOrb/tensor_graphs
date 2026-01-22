import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_gather_embedding():
    """Test Gather acting as an embedding lookup"""
    vocab_size = 10
    embed_dim = 4

    # Weight Matrix (10, 4)
    w = TensorNode(OpType.INPUT, (vocab_size, embed_dim), DType.FP32, [], "w")

    # Indices (3,)
    idx = TensorNode(OpType.INPUT, (3,), DType.INT32, [], "idx")

    # Gather -> (3, 4)
    gather_node = TensorNode(
        OpType.GATHER, (3, embed_dim), DType.FP32, [w, idx], "gather"
    )

    val_w = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    val_idx = np.array([0, 2, 5], dtype=np.int32)

    res = evaluate_graph(gather_node, {"w": val_w, "idx": val_idx})

    expected = val_w[val_idx]
    np.testing.assert_array_equal(res, expected)


def test_gather_multidim_indices():
    """Test Gather with multi-dimensional indices"""
    vocab_size = 10
    embed_dim = 4

    w = TensorNode(OpType.INPUT, (vocab_size, embed_dim), DType.FP32, [], "w")

    # Indices (2, 2)
    idx = TensorNode(OpType.INPUT, (2, 2), DType.INT32, [], "idx")

    # Gather -> (2, 2, 4)
    gather_node = TensorNode(
        OpType.GATHER, (2, 2, embed_dim), DType.FP32, [w, idx], "gather"
    )

    val_w = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    val_idx = np.array([[0, 1], [2, 3]], dtype=np.int32)

    res = evaluate_graph(gather_node, {"w": val_w, "idx": val_idx})

    expected = val_w[val_idx]
    np.testing.assert_array_equal(res, expected)
