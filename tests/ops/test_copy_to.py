import pytest
import numpy as np
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, Backend
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph


def test_copy_to_cpu_numpy():
    x = TensorNode(OpType.INPUT, (2, 2), DType.FP32, [], "x")
    copy_node = TensorNode(
        OpType.COPY_TO,
        (2, 2),
        DType.FP32,
        [x],
        "copy",
        attrs={"target_backend": Backend.CPU_NUMPY.value},
    )

    data = np.random.randn(2, 2).astype(np.float32)
    inputs = {"x": data}
    res = evaluate_graph(copy_node, inputs)

    np.testing.assert_array_equal(res, data)
