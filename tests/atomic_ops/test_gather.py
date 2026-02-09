import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_gather_embedding():
    """Test Gather acting as an embedding lookup"""
    vocab_size = 10
    embed_dim = 4

    # Weight Matrix (10, 4)
    w = TensorNode(OpType.INPUT, DType.FP32, [], (vocab_size, embed_dim), "w")

    # Indices (3,)
    idx = TensorNode(OpType.INPUT, DType.INT32, [], (3,), "idx")

    # Gather -> (3, 4)
    gather_node = TensorNode(
        OpType.GATHER, DType.FP32, [w, idx], (3, embed_dim), "gather"
    )

    val_w = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    val_idx = np.array([0, 2, 5], dtype=np.int32)

    sess = GraphSession(gather_node)
    res = sess.run({"w": val_w, "idx": val_idx})

    expected = val_w[val_idx]
    np.testing.assert_array_equal(res, expected)


def test_gather_multidim_indices():
    """Test Gather with multi-dimensional indices"""
    vocab_size = 10
    embed_dim = 4

    w = TensorNode(OpType.INPUT, DType.FP32, [], (vocab_size, embed_dim), "w")

    # Indices (2, 2)
    idx = TensorNode(OpType.INPUT, DType.INT32, [], (2, 2), "idx")

    # Gather -> (2, 2, 4)
    gather_node = TensorNode(
        OpType.GATHER, DType.FP32, [w, idx], (2, 2, embed_dim), "gather"
    )

    val_w = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    val_idx = np.array([[0, 1], [2, 3]], dtype=np.int32)

    sess = GraphSession(gather_node)
    res = sess.run({"w": val_w, "idx": val_idx})

    expected = val_w[val_idx]
    np.testing.assert_array_equal(res, expected)
