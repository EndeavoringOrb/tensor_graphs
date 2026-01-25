import unittest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.ops.fused.norm import RMSNorm
from tensor_graphs.ops.fused.activation import GELU
from tensor_graphs.optim.fusion import fuse_graph


class TestOpFusion(unittest.TestCase):
    def test_rmsnorm_fusion(self):
        # Create a graph that matches RMSNorm.decompose
        x = TensorNode(OpType.INPUT, (1, 10), DType.FP32, [], "x")
        scale = TensorNode(OpType.INPUT, (10,), DType.FP32, [], "scale")
        eps = TensorNode(
            OpType.CONSTANT, (1,), DType.FP32, [], "eps", attrs={"value": 1e-6}
        )
        # Decompose RMSNorm to get the atomic graph
        rmsnorm = RMSNorm()
        decomposed = rmsnorm.decompose([x, scale, eps], attrs={"axis": -1})

        # Run Optimizer
        optimized_node = fuse_graph(decomposed)

        # Should be fused back to RMSNorm
        self.assertEqual(optimized_node.op_type, "RMSNorm")
        self.assertEqual(len(optimized_node.parents), 3)
        self.assertEqual(optimized_node.parents[0], x)
        self.assertEqual(optimized_node.parents[1], scale)
        self.assertEqual(optimized_node.parents[2], eps)

    def test_gelu_fusion(self):
        # Create a graph that matches GELU.decompose
        x = TensorNode(OpType.INPUT, (1, 10), DType.FP32, [], "x")

        # Decompose GELU
        gelu = GELU()
        decomposed = gelu.decompose([x])

        # Run Optimizer
        optimized_node = fuse_graph(decomposed)

        # Should be fused back to GELU
        self.assertEqual(optimized_node.op_type, "GELU")
        self.assertEqual(len(optimized_node.parents), 1)
        self.assertEqual(optimized_node.parents[0], x)


if __name__ == "__main__":
    unittest.main()
