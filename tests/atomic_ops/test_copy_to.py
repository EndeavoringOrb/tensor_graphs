import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, Backend
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.session import GraphSession


def test_copy_to_cpu_numpy():   
    x = TensorNode(OpType.INPUT, DType.FP32, [], (2, 2), "x")
    copy_node = TensorNode(
        OpType.COPY_TO,
        DType.FP32,
        [x],
        (2, 2),
        "copy",
        attrs={"target_backend": Backend.CPU_NUMPY.value},
    )

    data = np.random.randn(2, 2).astype(np.float32)
    inputs = {"x": data}
    sess = GraphSession(copy_node)
    res = sess.run(inputs)

    np.testing.assert_array_equal(res, data)
