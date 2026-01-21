import unittest
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ir.graph import topological_sort, get_inputs
from tensor_graphs.ops.atomic import OpType

class TestGraphBuild(unittest.TestCase):
    def test_topo_sort(self):
        a = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "A")
        b = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "B")
        mul = TensorNode(OpType.MUL, (1,), DType.FP32, [a, b], "Mul")
        
        sorted_nodes = topological_sort(mul)
        
        self.assertIn(a, sorted_nodes)
        self.assertIn(b, sorted_nodes)
        self.assertEqual(sorted_nodes[-1], mul)
        
    def test_get_inputs(self):
        a = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "A")
        b = TensorNode(OpType.INPUT, (1,), DType.FP32, [], "B")
        mul = TensorNode(OpType.MUL, (1,), DType.FP32, [a, b], "Mul")
        
        inputs = get_inputs(mul)
        self.assertEqual(len(inputs), 2)
        self.assertIn(a, inputs)

if __name__ == "__main__":
    unittest.main()