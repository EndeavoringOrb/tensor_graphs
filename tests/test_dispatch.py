import unittest
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dispatch import resolve_dispatch
import tensor_graphs.ops.fused.llm  # Register RoPE and Embedding


class TestDispatch(unittest.TestCase):
    def test_autocast_inplace(self):
        # Scenario:
        # Input A: FP32 Scalar (1,)
        # Input B: FP8 Matrix (4, 4)
        # Op: Add
        node_a_fp32 = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "scalar_fp32")
        node_b_fp8 = TensorNode(OpType.INPUT, (4, 4), DType.FP8E4M3, [], "matrix_fp8")

        # User tries to create an Add node.
        target_node = TensorNode(
            OpType.ADD, (4, 4), DType.FP32, [node_a_fp32, node_b_fp8], "add_op"
        )

        original_id = id(target_node)
        executable_node = resolve_dispatch(target_node)

        # Verify in-place mutation
        self.assertEqual(id(executable_node), original_id)
        self.assertEqual(target_node.op_type, OpType.ADD)

        # Verify that the FP8 input was cast to FP32 (autocast heuristic)
        # target_node.parents[0] is node_a_fp32 (already FP32)
        # target_node.parents[1] should be a Cast node
        self.assertEqual(target_node.parents[1].op_type, "Cast")
        self.assertEqual(target_node.parents[1].dtype, DType.FP32)
        self.assertEqual(target_node.parents[1].parents[0], node_b_fp8)

    def test_rope_decomposition_inplace(self):
        # RoPE inputs: x, cos, sin
        # Use FP8 to ensure no kernel matches and it decomposes
        x = TensorNode(OpType.INPUT, (1, 128), DType.FP8E4M3, [], "x")
        cos = TensorNode(OpType.INPUT, (1, 128), DType.FP8E4M3, [], "cos")
        sin = TensorNode(OpType.INPUT, (1, 128), DType.FP8E4M3, [], "sin")

        rope_node = TensorNode("RoPE", (1, 128), DType.FP32, [x, cos, sin], "rope")
        original_id = id(rope_node)

        executable_node = resolve_dispatch(rope_node)

        # Verify in-place mutation
        self.assertEqual(id(executable_node), original_id)

        # RoPE decomposes to Add as root (based on its implementation in llm.py)
        # It might be wrapped or further resolved, but resolve_dispatch with decomposition
        # should eventually result in an atomic op if it can't find a fused kernel.
        self.assertEqual(rope_node.op_type, OpType.ADD)

    def test_embedding_decomposition(self):
        # Use a signature that won't match the fused kernel to force decomposition
        # Fused kernel expects (None, None) for both indices and weights.
        # Here indices is (10,)
        indices = TensorNode(OpType.INPUT, (10,), DType.INT32, [], "indices")
        weights = TensorNode(OpType.INPUT, (100, 128), DType.FP32, [], "weights")

        emb_node = TensorNode(
            "Embedding", (10, 128), DType.FP32, [indices, weights], "emb"
        )
        original_id = id(emb_node)

        executable_node = resolve_dispatch(emb_node)

        self.assertEqual(id(executable_node), original_id)
        # Embedding decomposes to Gather
        self.assertEqual(emb_node.op_type, OpType.GATHER)


if __name__ == "__main__":
    unittest.main()
