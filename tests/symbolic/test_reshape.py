from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.compiler.dirty_propagation import DirtyPropagator


def test_reshape_forward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 5), "a")
    res = TensorNode(OpType.RESHAPE, DType.FP32, [a], (10,), "reshape")

    a.dirty_region = [(slice(0, 1), slice(0, 1))]
    out_dirty = DirtyPropagator.propagate(res)
    assert out_dirty == [(slice(0, 1),)]


def test_reshape_backward():
    a = TensorNode(OpType.INPUT, DType.FP32, [], (2, 5), "a")
    shape = TensorNode(OpType.CONSTANT, DType.INT32, [], (1,), attrs={"value": 10})
    res = TensorNode(OpType.RESHAPE, DType.FP32, [a, shape], (10,), "reshape")

    in_slices = DirtyPropagator.get_input_slices(res, [(slice(2, 5),)])
    assert in_slices[0] == [(slice(0, 1), slice(2, 5))]