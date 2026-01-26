import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph


def test_dot_2x2():
    # Should use 2x2 optimized kernel
    a = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "a")
    b = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "b")
    node = TensorNode(OpType.DOT, (2, 2), DType.FP32, [a, b], "dot")

    mat = np.eye(2, dtype=np.float32)
    res = evaluate_graph(node, {"a": mat, "b": mat})
    np.testing.assert_array_equal(res, mat)
