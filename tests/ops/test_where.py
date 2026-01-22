import unittest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph


class TestWhereOp(unittest.TestCase):
    def test_where_fp32(self):
        # Condition, X, Y all FP32
        cond = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "cond")
        x = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "x")
        y = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "y")

        where_node = TensorNode(OpType.WHERE, (3,), DType.FP32, [cond, x, y], "where")

        inputs = {
            "cond": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "y": np.array([-10.0, -20.0, -30.0], dtype=np.float32),
        }

        result = evaluate_graph(where_node, inputs)
        expected = np.array([10.0, -20.0, 30.0], dtype=np.float32)

        np.testing.assert_allclose(result, expected)

    def test_where_bool(self):
        # Condition is BOOL
        cond = TensorNode(OpType.INPUT, (3,), DType.BOOL, [], "cond")
        x = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "x")
        y = TensorNode(OpType.INPUT, (3,), DType.FP32, [], "y")

        where_node = TensorNode(OpType.WHERE, (3,), DType.FP32, [cond, x, y], "where")

        inputs = {
            "cond": np.array([True, False, True], dtype=bool),
            "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "y": np.array([-10.0, -20.0, -30.0], dtype=np.float32),
        }

        result = evaluate_graph(where_node, inputs)
        expected = np.array([10.0, -20.0, 30.0], dtype=np.float32)

        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
