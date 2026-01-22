import unittest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.optim.fusion import fuse_graph
from tensor_graphs.backend.reference import evaluate_graph


class TestFusion(unittest.TestCase):
    def test_mul_add_fusion_structure(self):
        # Create: y = (x * w) + b
        x = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "x")
        w = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "w")
        b = TensorNode(OpType.INPUT, (32,), DType.FP32, [], "b")

        mul_node = TensorNode(OpType.MUL, (32,), DType.FP32, [x, w], "mul")
        add_node = TensorNode(OpType.ADD, (32,), DType.FP32, [mul_node, b], "add")

        # Run Optimizer
        optimized_node = fuse_graph(add_node)

        self.assertEqual(optimized_node.op_type, OpType.FUSED_MUL_ADD)
        self.assertEqual(len(optimized_node.parents), 3)

    def test_fusion_numerical_equivalence(self):
        # Shape: (2,)
        x = TensorNode(OpType.INPUT, (2,), DType.FP32, [], "x")
        w = TensorNode(OpType.INPUT, (2,), DType.FP32, [], "w")
        b = TensorNode(OpType.INPUT, (2,), DType.FP32, [], "b")

        mul_node = TensorNode(OpType.MUL, (2,), DType.FP32, [x, w], "mul")
        original_graph = TensorNode(OpType.ADD, (2,), DType.FP32, [mul_node, b], "add")

        fused_graph = fuse_graph(original_graph)

        inputs = {
            "x": np.array([2.0, 3.0], dtype=np.float32),
            "w": np.array([4.0, 5.0], dtype=np.float32),
            "b": np.array([1.0, 1.0], dtype=np.float32),
        }

        res_orig = evaluate_graph(original_graph, inputs)
        res_fused = evaluate_graph(fused_graph, inputs)

        np.testing.assert_array_equal(res_orig, res_fused)
        np.testing.assert_array_equal(
            res_fused, np.array([9.0, 16.0], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
