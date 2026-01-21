import unittest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph

class TestAtomicOps(unittest.TestCase):
    def test_basic_math(self):
        # Build graph: y = (a + b) * c
        # Shapes: (1,)
        a = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "a")
        b = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "b")
        c = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "c")
        
        add_node = TensorNode(OpType.ADD, (1,), DType.FP32, [a, b], "add")
        mul_node = TensorNode(OpType.MUL, (1,), DType.FP32, [add_node, c], "mul")
        
        inputs = {
            "a": np.array([2.0], dtype=np.float32),
            "b": np.array([3.0], dtype=np.float32),
            "c": np.array([4.0], dtype=np.float32)
        }
        
        result = evaluate_graph(mul_node, inputs)
        self.assertEqual(result[0], 20.0)

    def test_matrix_ops(self):
        # A @ B
        # Shapes: (2, 2)
        A = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "A")
        B = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "B")
        dot_node = TensorNode(OpType.DOT, (2, 2), DType.FP32, [A, B], "dot")
        
        data_a = np.eye(2, dtype=np.float32)
        data_b = np.array([[1, 2], [3, 4]], dtype=np.float32)
        
        res = evaluate_graph(dot_node, {"A": data_a, "B": data_b})
        
        np.testing.assert_array_equal(res, data_b)

if __name__ == "__main__":
    unittest.main()
