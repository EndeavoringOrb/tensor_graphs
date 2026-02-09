import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_cast_int_to_float():
    x = TensorNode(OpType.INPUT, DType.INT32, [], (3,), "x")
    cast_node = TensorNode(
        OpType.CAST, DType.FP32, [x], (3,), "cast", attrs={"to": DType.FP32}
    )

    inputs = {"x": np.array([1, 2, 3], dtype=np.int32)}
    sess = GraphSession(cast_node)
    res = sess.run(inputs)

    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(res, expected)
    assert res.dtype == np.float32
