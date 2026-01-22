import unittest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.ops.fused.math import FusedMulAdd
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

        # Check against the Composite Op Type string
        self.assertEqual(optimized_node.op_type, FusedMulAdd.op_type)
        self.assertEqual(len(optimized_node.parents), 3)


if __name__ == "__main__":
    unittest.main()
